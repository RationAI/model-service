import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torchvision.ops import MLP

from xai.nfm.configuration import Config
from xai.nfm.layers import FeedForward, RoPE


class Attention(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, rope_theta: float, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.head_dim = dim // num_heads

        self.rope = RoPE(self.head_dim, theta=rope_theta)
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

    def forward(
        self, tgt: Tensor, src: Tensor, tgt_pos: Tensor, src_pos: Tensor
    ) -> Tensor:
        self.tgt = tgt
        self.src = src
        self.tgt_pos = tgt_pos
        self.src_pos = src_pos

        q = rearrange(self.q(tgt), "b n (h d) -> b h n d", d=self.head_dim)
        k, v = rearrange(
            self.kv(src), "b n (two h d) -> two b h n d", two=2, d=self.head_dim
        )
        q = self.q_norm(q)
        k = self.k_norm(k)

        self.attn_map = (
            self.rope(q, tgt_pos) @ self.rope(k, src_pos).transpose(-1, -2)
        ) / (self.head_dim**0.5)
        attn = F.softmax(self.attn_map, dim=-1)

        x = attn @ v
        x = rearrange(x, "b h n d -> b n (h d)")

        return self.wo(x)

    def relprop(self, r: Tensor) -> tuple[Tensor, Tensor]:
        """Compute relevance propagation for Attention.

        Args:
            r: Relevance of output tokens [b, n, d]

        Returns:
            Relevance for target and source tokens.
        """
        # Simplified Attention LRP: R_i = sum_j (alpha_ij * R_j)
        # We want to return the contribution of each source token to each target token.
        # But standard relprop returns per-token relevance of the same shape as input.

        # If we want output tokens w.r.t input tokens, we are looking for a [n, m] matrix.
        # However, the relprop chain usually flows back to the original input.

        attn = F.softmax(self.attn_map, dim=-1)  # [b, h, n, m]

        # Weighted attention across heads
        # For simplicity, we average across heads and weight by output relevance magnitude
        r_mag = r.abs().mean(dim=-1, keepdim=True)  # [b, n, 1]

        # Contribution from source tokens to target relevance
        # R_src_per_tgt = attn * r_mag?
        # Actually, let's return the relevance for src that will be used by the caller.

        # Backprop through attention: R_src = attn^T @ R_tgt
        # (Actually attn is [n, m], so R_src [m, d] = attn^T [m, n] @ R_tgt [n, d])

        # Since r is [b, n, d], we want r_src as [b, m, d]
        # We average attention over heads for simplicity
        avg_attn = attn.mean(dim=1)  # [b, n, m]

        r_src = torch.bmm(avg_attn.transpose(-1, -2), r)  # [b, m, d]
        r_tgt = r  # The tgt relevance stays (simplified)

        return r_tgt, r_src


class CrossLayer(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.self_attn = Attention(
            dim=config.dim, num_heads=config.num_heads, rope_theta=config.rope_theta
        )
        self.cross_attn = Attention(
            dim=config.dim, num_heads=config.num_heads, rope_theta=config.rope_theta
        )
        self.ffn = FeedForward(config.dim, config.hidden_dim)

        self.pre_self_attn_norm = nn.RMSNorm(config.dim)
        self.pre_cross_attn_norm = nn.RMSNorm(config.dim)
        self.pre_ffn_norm = nn.RMSNorm(config.dim)

    def forward(
        self, tgt: Tensor, src: Tensor, tgt_pos: Tensor, src_pos: Tensor
    ) -> Tensor:
        self.tgt_in = tgt
        self.src_in = src
        y = self.pre_cross_attn_norm(tgt)
        self.cross_attn_out = self.cross_attn(y, src, tgt_pos, src_pos)
        tgt = tgt + self.cross_attn_out

        y = self.pre_self_attn_norm(tgt)
        self.self_attn_out = self.self_attn(y, y, tgt_pos, tgt_pos)
        tgt = tgt + self.self_attn_out

        y = self.pre_ffn_norm(tgt)
        self.ffn_out = self.ffn(y)
        return tgt + self.ffn_out

    def relprop(self, r: Tensor) -> tuple[Tensor, Tensor]:
        # Residual LRP: R_in = R_out * (x / (x + f(x)))
        # For simplicity, we split relevance 50/50 or proportional to magnitude

        r_ffn = self.ffn.relprop(r * 0.5)
        r = r * 0.5 + r_ffn

        r_self, _ = self.self_attn.relprop(r * 0.5)
        r = r * 0.5 + r_self

        r_cross, r_src = self.cross_attn.relprop(r * 0.5)
        r_tgt = r * 0.5 + r_cross

        return r_tgt, r_src


class Layer(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.self_attn = Attention(
            dim=config.dim, num_heads=config.num_heads, rope_theta=config.rope_theta
        )
        self.ffn = FeedForward(config.dim, config.hidden_dim)

        self.pre_self_attn_norm = nn.RMSNorm(config.dim)
        self.pre_ffn_norm = nn.RMSNorm(config.dim)

    def forward(self, x: Tensor, pos: Tensor) -> Tensor:
        self.x_in = x
        y = self.pre_self_attn_norm(x)
        self.self_attn_out = self.self_attn(y, y, pos, pos)
        x = x + self.self_attn_out

        y = self.pre_ffn_norm(x)
        self.ffn_out = self.ffn(y)
        return x + self.ffn_out

    def relprop(self, r: Tensor) -> Tensor:
        r_ffn = self.ffn.relprop(r * 0.5)
        r = r * 0.5 + r_ffn

        r_self, _ = self.self_attn.relprop(r * 0.5)
        r = r * 0.5 + r_self
        return r


class Transformer(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.bn = nn.BatchNorm1d(4 * config.efd_order, affine=False)
        self.polygon_proj = nn.Linear(4 * config.efd_order, config.dim)

        self.cross_layers = nn.ModuleList(
            CrossLayer(config) for _ in range(config.num_cross_layers)
        )
        self.self_layers = nn.ModuleList(
            Layer(config) for _ in range(config.num_self_layers)
        )
        self.norm = nn.RMSNorm(config.dim)

    def forward(self, src: Tensor, tgt_pos: Tensor, src_pos: Tensor) -> Tensor:
        """Forward pass of the Transformer model.

        Args:
            src: Source sequence of shape (b, m, d)
            tgt_pos: Target positions of shape (b, n, 2)
            src_pos: Source positions of shape (b, m, 2)
        """
        # Ignore zero tokens as they are padded polygons
        src_flatten = src.flatten(0, 1)
        non_zero = src_flatten.abs().sum(dim=-1) != 0
        if non_zero.any():
            src_flatten[non_zero] = self.bn(src_flatten[non_zero])

        src = self.polygon_proj(src)

        tgt = torch.ones(src.shape[0], tgt_pos.shape[1], src.shape[2]).to(src)

        for layer in self.cross_layers:
            tgt = layer(tgt, src, tgt_pos, src_pos)

        for layer in self.self_layers:
            tgt = layer(tgt, tgt_pos)

        return self.norm(tgt)

    def relprop(self, r: Tensor) -> Tensor:
        # Propagation through self layers
        for layer in reversed(self.self_layers):
            r = layer.relprop(r)

        # Propagation through cross layers
        r_src_total = torch.zeros_like(self.cross_layers[0].src_in)
        for layer in reversed(self.cross_layers):
            r, r_src = layer.relprop(r)
            r_src_total += r_src

        return r_src_total


class NucleiGraphEncoder(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.backbone = Transformer(config)

        self.proj = MLP(
            config.dim,
            hidden_channels=[
                config.proj_hidden_dim,
                config.proj_hidden_dim,
                config.proj_dim,
            ],
            norm_layer=nn.BatchNorm1d,
        )
        self.final_norm = nn.BatchNorm1d(config.proj_dim, affine=False)

    def forward(
        self, src: Tensor, tgt_pos: Tensor, src_pos: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Forward pass of the Transformer model.

        Args:
            src: Source sequence of shape (b, m, d)
            tgt_pos: Target positions of shape (b, n, 2)
            src_pos: Source positions of shape (b, m, 2)
        """
        embed = self.backbone(src, tgt_pos, src_pos)
        # 2. Global Average Pooling
        return embed, self.final_norm(self.proj(embed.mean(dim=1)))
