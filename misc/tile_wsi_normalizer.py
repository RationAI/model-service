import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyvips


class TileWSINormalizer:
    def __init__(
        self, extent_x: int, extent_y: int, mpp_x: float, mpp_y: float
    ) -> None:
        self.extent_x = extent_x
        self.extent_y = extent_y
        self.mpp_x = mpp_x
        self.mpp_y = mpp_y

        # Create temporary files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.result_path = Path(self.temp_dir.name) / "result.raw"
        self.count_path = Path(self.temp_dir.name) / "count.raw"

        self.result = np.memmap(
            str(self.result_path),
            dtype=np.uint8,
            mode="w+",
            shape=(self.extent_y, self.extent_x, 3),
        )

        self.count = np.memmap(
            str(self.count_path),
            dtype=np.uint8,
            mode="w+",
            shape=(self.extent_y, self.extent_x),
        )

    def update(self, tile: np.ndarray[Any, Any], x: int, y: int) -> None:
        # Model returns CHW uint8; buffer is HWC uint8.
        if tile.ndim == 3 and tile.shape[0] == 3:
            tile = tile.transpose(1, 2, 0)


        h = max(0, min(tile.shape[0], self.extent_y - y))
        w = max(0, min(tile.shape[1], self.extent_x - x))
        if h == 0 or w == 0:
            return
        tile = tile[:h, :w]

        region = self.result[y : y + h, x : x + w]
        count = self.count[y : y + h, x : x + w]

        overlap = count > 0
        if overlap.any():
            n = count[:, :, np.newaxis].astype(np.float32)
            blended = np.where(
                overlap[:, :, np.newaxis],
                (region.astype(np.float32) * n + tile) / (n + 1),
                tile,
            )
            self.result[y : y + h, x : x + w] = np.clip(blended, 0, 255).astype(
                np.uint8
            )
        else:
            self.result[y : y + h, x : x + w] = tile

        self.count[y : y + h, x : x + w] = count + 1

    def flush(self) -> None:
        self.result.flush()
        self.count.flush()

    def save(self, output_path: str, tile_width: int, tile_height: int) -> None:
        result_img = pyvips.Image.rawload(
            str(self.result_path), self.extent_x, self.extent_y, 3
        )
        result_img = result_img.copy(interpretation=pyvips.Interpretation.SRGB)

        count_img = pyvips.Image.rawload(
            str(self.count_path), self.extent_x, self.extent_y, 1
        )
        mask = count_img > 0

        # White background for tiles that were never written (non-tissue regions).
        white = (
            pyvips.Image.black(self.extent_x, self.extent_y, bands=3) + 255
        ).cast(pyvips.BandFormat.UCHAR)
        final_img = mask.ifthenelse(result_img, white)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        final_img.tiffsave(
            output_path,
            bigtiff=True,
            compression=pyvips.enums.ForeignTiffCompression.DEFLATE,
            tile=True,
            tile_width=tile_width,
            tile_height=tile_height,
            xres=1000 / self.mpp_x,
            yres=1000 / self.mpp_y,
            pyramid=True,
        )

    def cleanup(self) -> None:
        if hasattr(self, "result"):
            del self.result
        if hasattr(self, "count"):
            del self.count

        self.temp_dir.cleanup()

    def __del__(self) -> None:
        self.cleanup()
