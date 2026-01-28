import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# from fastapi import FastAPI
# from ray import serve
from xai.nfm.configuration import Config
from xai.nfm.transformer import NucleiGraphEncoder


# fastapi = FastAPI()


def cluster_reorder(centroids: np.ndarray, cluster_size: int = 4096):
    """Reorder centroids and efd so that sequential blocks form spatial clusters."""
    from sklearn.cluster import MiniBatchKMeans

    n_clusters = max(1, len(centroids) // cluster_size)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, n_init=3, random_state=42)
    labels = kmeans.fit_predict(centroids)

    return np.argsort(labels)


# @serve.deployment(num_replicas="auto")
# @serve.ingress(fastapi)
class NFM:
    def __init__(self) -> None:
        data = torch.load("pytorch_model.bin", map_location=torch.device("cuda"))
        state_dict = {k.replace("model.", "", 1): v for k, v in data.items()}

        self.model = NucleiGraphEncoder(Config()).cuda()
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def predict(
        self, src: torch.Tensor, tgt_pos: torch.Tensor, src_pos: torch.Tensor
    ) -> torch.Tensor:
        src = src.unsqueeze(0).cuda()
        tgt_pos = tgt_pos.unsqueeze(0).cuda()
        src_pos = src_pos.unsqueeze(0).cuda()

        print(src.shape, tgt_pos.shape, src_pos.shape)

        with torch.no_grad():
            output, _ = self.model(src, tgt_pos, src_pos)
        return output

    # @fastapi.post("/get_spatial_registers")
    def get_spatial_registers(self, slide_path: str):
        from ratiopath.openslide import OpenSlide

        from xai.nfm.misc import (
            elliptic_fourier_descriptors,
            sample_spatial_registers,
        )

        data = pd.read_parquet(
            f"/mnt/projects/nuclei_based_wsi_analysis/nuclei_segmentation/tile_level_annotations/slide_id={Path(slide_path).stem}",
            columns=["polygon"],
        )
        polygons = np.stack(data.polygon).reshape(-1, 64, 2)

        with OpenSlide(slide_path) as slide:
            level = slide.closest_level(0.5)
            mpp_x, mpp_y = slide.slide_resolution(level)
            polygons[..., 0] *= mpp_x
            polygons[..., 1] *= mpp_y

        efd = elliptic_fourier_descriptors(polygons.astype(np.float64), order=16)
        efd = efd.astype(np.float32)
        centroids = polygons.mean(axis=1).astype(np.float32)

        indices = cluster_reorder(centroids)
        efd = efd[indices]
        centroids = centroids[indices]

        n_spatial_reg = centroids.shape[0] // 16

        # return spatial_registers
        embed = torch.empty((1, n_spatial_reg, 384), dtype=torch.float32).cuda()
        spatial_registers = torch.empty((n_spatial_reg, 2), dtype=torch.float32)

        for i in range(0, len(polygons), 4096):
            j = i // 16
            spatial_registers[j : j + 256] = torch.from_numpy(
                sample_spatial_registers(
                    centroids[i : i + 4096],
                    n_samples=len(centroids[i : i + 4096]) // 16,
                )
            )

            embed[:, j : j + 256] = self.predict(
                src=torch.from_numpy(efd[i : i + 4096]).reshape(-1, 64),
                tgt_pos=spatial_registers[j : j + 256],
                src_pos=torch.from_numpy(centroids[i : i + 4096]),
            )

        session_id = hashlib.sha256(slide_path.encode()).hexdigest()
        return embed, spatial_registers.numpy(), centroids

        # session_id = str(uuid.uuid4())

    # @fastapi.post("/explain")
    async def explain(
        self,
        src: list[list[float]],
        tgt_pos: list[list[float]],
        src_pos: list[list[float]],
    ) -> dict:
        """Compute LRP for each output token with respect to input tokens."""
        src_t = torch.tensor(src).unsqueeze(0).float()
        tgt_pos_t = torch.tensor(tgt_pos).unsqueeze(0).float()
        src_pos_t = torch.tensor(src_pos).unsqueeze(0).float()

        # 1. Forward pass to store activations
        embed, _ = self.model(src_t, tgt_pos_t, src_pos_t)

        # 2. Backward pass with LRP
        # Initialize relevance as the identity for each output token
        # This gives us attribution for each output token separately
        num_output_tokens = embed.shape[1]
        num_input_tokens = src_t.shape[1]

        # We can compute this in a loop or batch it if memory allows
        # For now, let's compute the attribution matrix
        attribution_matrix = []

        for i in range(num_output_tokens):
            # Relevance starts at token i
            r = torch.zeros_like(embed)
            r[0, i, :] = embed[0, i, :]  # Start with the activation of that token

            # Propagate back through the transformer
            r_back = self.model.backbone.relprop(r)

            # Magnitude of relevance per input token
            r_per_token = r_back.abs().mean(dim=-1).squeeze(0)  # [m]
            attribution_matrix.append(r_per_token.tolist())

        return {
            "attribution_matrix": attribution_matrix,  # [num_output_tokens, num_input_tokens]
            "num_output_tokens": num_output_tokens,
            "num_input_tokens": num_input_tokens,
        }


# app = NFM.bind()

nfm = NFM()

r = nfm.get_spatial_registers(
    "/mnt/data/MOU/prostate/tile_level_annotations/P-2016_0383-12-0.mrxs"
)


import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA


embed, spatial_registers, centroids = nfm.get_spatial_registers(
    "/mnt/data/MOU/prostate/tile_level_annotations/P-2016_0383-12-0.mrxs"
)

# Squeeze the batch dimension and move to CPU
embeddings = embed.squeeze(0).cpu().detach().numpy()
pca_color = PCA(n_components=1).fit_transform(embeddings).flatten()

# Plot the spatial registers colored by their embedding features
plt.figure(figsize=(12, 10))

# Plot all centroids as small points in the background
# plt.scatter(
#     centroids[:, 0],
#     centroids[:, 1],
#     c="lightgray",
#     alpha=0.3,
#     s=5,
#     label="All Centroids",
# )

scatter = plt.scatter(
    spatial_registers[:, 0],
    spatial_registers[:, 1],
    c=pca_color,
    cmap="viridis",
    alpha=0.7,
    s=2,
    label="Spatial Registers",
)
plt.legend()
plt.colorbar(scatter, label="First Principal Component (Feature Variance)")
plt.title("Spatial Distribution of Embeddings")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.gca().invert_yaxis()  # Invert Y to match image/slide coordinate systems
plt.grid(True, linestyle="--", alpha=0.3)
plt.savefig("spatial_embeddings_map.png", dpi=300)
