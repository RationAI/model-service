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
        self,
        tgt: Tensor,
        src: Tensor,
        tgt_pos: Tensor,
        src_pos: Tensor,
        return_attn: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
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

        if return_attn:
            return self.wo(x), attn.max(dim=1).values
        return self.wo(x), None


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
        self,
        tgt: Tensor,
        src: Tensor,
        tgt_pos: Tensor,
        src_pos: Tensor,
        return_attn: bool = False,
    ) -> tuple[Tensor, list[Tensor] | None]:
        self.tgt_in = tgt
        self.src_in = src

        y = self.pre_cross_attn_norm(tgt)
        self.cross_attn_out, cross_attn = self.cross_attn(
            y, src, tgt_pos, src_pos, return_attn=return_attn
        )
        tgt = tgt + self.cross_attn_out

        y = self.pre_self_attn_norm(tgt)
        self.self_attn_out, self_attn = self.self_attn(
            y, y, tgt_pos, tgt_pos, return_attn=return_attn
        )
        tgt = tgt + self.self_attn_out

        y = self.pre_ffn_norm(tgt)
        self.ffn_out = self.ffn(y)
        out = tgt + self.ffn_out

        if return_attn:
            return out, (cross_attn, self_attn)
        return out, None


class Layer(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.self_attn = Attention(
            dim=config.dim, num_heads=config.num_heads, rope_theta=config.rope_theta
        )
        self.ffn = FeedForward(config.dim, config.hidden_dim)

        self.pre_self_attn_norm = nn.RMSNorm(config.dim)
        self.pre_ffn_norm = nn.RMSNorm(config.dim)

    def forward(
        self, x: Tensor, pos: Tensor, return_attn: bool = False
    ) -> tuple[Tensor, Tensor | None]:
        self.x_in = x
        y = self.pre_self_attn_norm(x)
        self.self_attn_out, attn = self.self_attn(
            y, y, pos, pos, return_attn=return_attn
        )
        x = x + self.self_attn_out

        y = self.pre_ffn_norm(x)
        self.ffn_out = self.ffn(y)
        out = x + self.ffn_out
        return out, attn


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

    def forward(
        self,
        src: Tensor,
        tgt_pos: Tensor,
        src_pos: Tensor,
        return_attn: bool = False,
    ) -> tuple[Tensor, list[Tensor] | None]:
        """Forward pass of the Transformer model.

        Args:
            src: Source sequence of shape (b, m, d)
            tgt_pos: Target positions of shape (b, n, 2)
            src_pos: Source positions of shape (b, m, 2)
            return_attn: Whether to return attention maps.
        """
        # Ignore zero tokens as they are padded polygons
        src_flatten = src.flatten(0, 1)
        non_zero = src_flatten.abs().sum(dim=-1) != 0
        if non_zero.any():
            src_flatten[non_zero] = self.bn(src_flatten[non_zero])

        src = self.polygon_proj(src)

        tgt = torch.ones(src.shape[0], tgt_pos.shape[1], src.shape[2]).to(src)

        self_attn_maps = []
        cross_attn_maps = []
        for layer in self.cross_layers:
            tgt, attn = layer(tgt, src, tgt_pos, src_pos, return_attn=return_attn)
            if return_attn:
                cross_attn_maps.append(attn[0])
                self_attn_maps.append(attn[1])

        for layer in self.self_layers:
            tgt, attn = layer(tgt, tgt_pos, return_attn=return_attn)
            if return_attn:
                self_attn_maps.append(attn)

        out = self.norm(tgt)
        if return_attn:
            return out, {"self": self_attn_maps, "cross": cross_attn_maps}
        return out, None


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
        self, src: Tensor, tgt_pos: Tensor, src_pos: Tensor, return_attn: bool = False
    ) -> tuple[Tensor, Tensor, list[Tensor] | None]:
        """Forward pass of the Transformer model.

        Args:
            src: Source sequence of shape (b, m, d)
            tgt_pos: Target positions of shape (b, n, 2)
            src_pos: Source positions of shape (b, m, 2)
            return_attn: Whether to return attention maps.
        """
        embed, attn = self.backbone(src, tgt_pos, src_pos, return_attn=return_attn)
        # 2. Global Average Pooling
        return embed, self.final_norm(self.proj(embed.mean(dim=1))), attn
