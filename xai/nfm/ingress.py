import hashlib
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ray import serve


fastapi = FastAPI()

fastapi.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows any website (including your xOpat) to connect
    allow_credentials=True,
    allow_methods=["*"],  # Allows POST, OPTIONS, etc.
    allow_headers=["*"],
)


def get_cluster_indices(centroids: np.ndarray, cluster_size: int = 4096):
    """Get blocks of indices that form spatial clusters using MiniBatchKMeans."""
    from sklearn.cluster import MiniBatchKMeans

    n_clusters = max(1, len(centroids) // cluster_size)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, n_init=3, random_state=42)
    labels = kmeans.fit_predict(centroids)

    indices = np.argsort(labels)
    return [indices[i : i + cluster_size] for i in range(0, len(indices), cluster_size)]


@serve.deployment(num_replicas="auto")
@serve.ingress(fastapi)
class NFM:
    device = "cuda"
    mpp = 0.25
    efd_order = 16
    output_dir = Path("/mnt/projects/nuclei_foundational_model/xai_masks")

    def __init__(self) -> None:
        import torch

        from xai.nfm.configuration import Config
        from xai.nfm.transformer import NucleiGraphEncoder

        data = torch.load(
            "/mnt/projects/nuclei_foundational_model/model.bin",
            map_location=torch.device(self.device),
        )
        state_dict = {k.replace("model.", "", 1): v for k, v in data.items()}
        self.config = Config()
        self.model = NucleiGraphEncoder(self.config).to(self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        self.temp_dir = Path("/tmp/nfm_attention")
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    @fastapi.post("/get_spatial_registers")
    def get_spatial_registers(self, slide_path: str):
        import pandas as pd
        import torch
        from ratiopath.openslide import OpenSlide

        from xai.nfm.misc import elliptic_fourier_descriptors, sample_spatial_registers

        session_id = hashlib.sha256(slide_path.encode()).hexdigest()
        session_path = self.temp_dir / session_id
        session_path.mkdir(exist_ok=True, parents=True)

        data = pd.read_parquet(
            f"/mnt/projects/nuclei_based_wsi_analysis/nuclei_segmentation/tile_level_annotations/slide_id={Path(slide_path).stem}",
            columns=["polygon"],
        )
        polygons = np.stack(data.polygon).reshape(-1, 64, 2)
        # Save polygons for mask generation later
        np.save(session_path / "polygons.npy", polygons)

        with OpenSlide(slide_path) as slide:
            level = slide.closest_level(self.mpp)
            mpp_x, mpp_y = slide.slide_resolution(level)
            extent_x, extent_y = slide.level_dimensions[level]

        with open(session_path / "metadata.json", "w") as f:
            json.dump(
                {
                    "extent_x": extent_x,
                    "extent_y": extent_y,
                    "mpp_x": mpp_x,
                    "mpp_y": mpp_y,
                },
                f,
            )

        polygons[..., 0] *= mpp_x
        polygons[..., 1] *= mpp_y
        efd = elliptic_fourier_descriptors(
            polygons.astype(np.float64), order=self.efd_order
        ).astype(np.float32)
        centroids = polygons.mean(axis=1)

        all_attn_data = []
        all_embeddings = []
        spatial_registers_list = []
        current_reg_idx = 0

        for indices in get_cluster_indices(centroids):
            n_reg_chunk = len(indices) // 16
            if n_reg_chunk == 0:
                continue

            chunk_centroids = centroids[indices]
            chunk_efd = efd[indices]

            chunk_spatial_registers = sample_spatial_registers(
                chunk_centroids, n_samples=n_reg_chunk
            )

            src = torch.from_numpy(chunk_efd).view(1, -1, 64).to(self.device)
            tgt_pos = (
                torch.from_numpy(chunk_spatial_registers)
                .unsqueeze(0)
                .to(self.device)
                .float()
            )
            src_pos = (
                torch.from_numpy(chunk_centroids).unsqueeze(0).to(self.device).float()
            )

            with torch.no_grad():
                embed, _, attn = self.model(src, tgt_pos, src_pos, return_attn=True)

            all_attn_data.append(
                {
                    "data": torch.stack(attn["cross"]).squeeze(1),
                    "reg_start": current_reg_idx,
                    "nuclei_idx": torch.from_numpy(indices).long().to(self.device),
                }
            )

            all_embeddings.append(embed.squeeze(0).cpu().numpy())
            spatial_registers_list.append(chunk_spatial_registers)
            current_reg_idx += n_reg_chunk

        final_indices = []
        final_values = []

        for chunk in all_attn_data:
            # Find non-zero values in the chunk
            # Adjust threshold (e.g., > 1e-4) if you want it even smaller/sparser
            nz_layers, nz_regs, nz_nuclei_local = torch.where(chunk["data"] > 1e-3)

            # Map local indices to global dimensions
            global_regs = nz_regs + chunk["reg_start"]
            global_nuclei = chunk["nuclei_idx"][nz_nuclei_local]

            # Stack into (Layer, Head, Reg, Nuclei)
            chunk_indices = torch.stack([nz_layers, global_regs, global_nuclei])

            final_indices.append(chunk_indices)
            final_values.append(chunk["data"][nz_layers, nz_regs, nz_nuclei_local])

        # Combine everything into one 4D sparse tensor
        total_nuclei = len(polygons)
        total_reg = total_nuclei // 16
        num_layers = self.config.num_cross_layers + self.config.num_self_layers

        full_sparse_attn = torch.sparse_coo_tensor(
            torch.cat(final_indices, dim=1),
            torch.cat(final_values),
            size=(num_layers, total_reg, total_nuclei),
        ).coalesce()

        torch.save(full_sparse_attn.cpu(), session_path / "attention_3d.pt")

        spatial_registers = np.concatenate(spatial_registers_list, axis=0)
        spatial_registers[..., 0] /= mpp_x
        spatial_registers[..., 1] /= mpp_y
        return {
            "session_id": session_id,
            "embeddings": np.concatenate(all_embeddings, axis=0).tolist(),
            "spatial_registers": spatial_registers.tolist(),
        }

    @fastapi.post("/generate_mask")
    def generate_mask(
        self, session_id: str, register_ids: list[int], layers: list[int]
    ):
        import pyvips
        import torch
        from PIL import Image, ImageDraw

        session_path = self.temp_dir / session_id
        if not session_path.exists():
            return {"error": "Session not found or attention maps expired"}

        with open(session_path / "metadata.json") as f:
            metadata = json.load(f)

        polygons = np.load(session_path / "polygons.npy")
        extent_x = metadata["extent_x"]
        extent_y = metadata["extent_y"]

        attention = torch.load(session_path / "attention_3d.pt")
        attention = torch.index_select(attention, 0, torch.tensor(layers))
        attention = torch.index_select(attention, 1, torch.tensor(register_ids))

        dense_attention = attention.to_dense()

        # Average over registers, then cumulative product over layers
        # Result shape: [nuclei]
        total_attention = dense_attention.sum(0).mean(dim=0).numpy()

        # Normalize attention scores (0-255)
        total_attention = (total_attention / total_attention.max() * 255).astype(
            np.uint8
        )

        # vips_img = pyvips.Image.black(extent_x, extent_y, bands=1)

        img = Image.new("L", (extent_x // 2, extent_y // 2), 0)
        draw = ImageDraw.Draw(img)
        for i, poly in enumerate(polygons):
            score = int(total_attention[i])
            if score > 0:
                draw.polygon(poly / 2, fill=score)

        vips_img = pyvips.Image.new_from_array(img)

        # for i, poly in enumerate(polygons):
        #     score = int(total_attention[i])
        #     if score > 0:
        #         x_min, y_min = np.floor(poly.min(axis=0)).astype(int)
        #         x_max, y_max = np.ceil(poly.max(axis=0)).astype(int)
        #         img = Image.new("L", (x_max - x_min, y_max - y_min), 0)
        #         draw = ImageDraw.Draw(img)
        #         draw.polygon(poly - [x_min, y_min], fill=score)

        #         vips_patch = pyvips.Image.new_from_array(np.array(img))
        #         vips_img = vips_img.draw_image(vips_patch, x_min, y_min)

        output_path = self.output_dir / f"{session_id}_attention_rollout.tif"
        vips_img.tiffsave(
            output_path,
            bigtiff=True,
            compression=pyvips.enums.ForeignTiffCompression.DEFLATE,
            tile=True,
            tile_width=512,
            tile_height=512,
            xres=1000 / metadata["mpp_x"] * 2,
            yres=1000 / metadata["mpp_y"] * 2,
            pyramid=True,
        )

        return {"mask_path": output_path}


app = NFM.bind()
