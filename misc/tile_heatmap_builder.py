import tempfile
from pathlib import Path

import numpy as np
import pyvips


class TileHeatmapBuilder:
    def __init__(
        self, extent_x: int, extent_y: int, mpp_x: float, mpp_y: float
    ) -> None:
        self.extent_x = extent_x
        self.extent_y = extent_y
        self.mpp_x = mpp_x
        self.mpp_y = mpp_y

        # Create temporary files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.image_path = Path(self.temp_dir.name) / "image.dat"
        self.count_path = Path(self.temp_dir.name) / "count.dat"

        self.image = np.memmap(
            str(self.image_path),
            dtype=np.float32,
            mode="w+",
            shape=(self.extent_y, self.extent_x),
        )

        self.count = np.memmap(
            str(self.count_path),
            dtype=np.uint8,
            mode="w+",
            shape=(self.extent_y, self.extent_x),
        )

    def update(self, tile: np.ndarray, x: int, y: int) -> None:
        mm_y, mm_x = self.image[y : y + tile.shape[0], x : x + tile.shape[1]].shape
        self.image[y : y + mm_y, x : x + mm_x] += tile[:mm_y, :mm_x]
        self.count[y : y + mm_y, x : x + mm_x] += 1

    def flush(self) -> None:
        self.image.flush()
        self.count.flush()

    def save(self, output_path: str, tile_width: int, tile_height: int) -> None:
        image_vips = pyvips.Image.new_from_array(self.image)
        count_vips = pyvips.Image.new_from_array(self.count)

        image_vips /= count_vips
        image_vips *= 255
        image_vips = image_vips.cast(pyvips.BandFormat.UCHAR)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        image_vips.tiffsave(
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
        if hasattr(self, "image"):
            del self.image
        if hasattr(self, "count"):
            del self.count

        self.temp_dir.cleanup()

    def __del__(self) -> None:
        self.cleanup()
