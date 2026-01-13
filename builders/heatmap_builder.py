import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypedDict

from fastapi import FastAPI
from ray import serve


class Config(TypedDict):
    num_threads: int
    max_concurrent_tasks: int


fastapi = FastAPI()


@serve.deployment(num_replicas="auto")
@serve.ingress(fastapi)
class HeatmapBuilder:
    num_threads: int
    max_concurrent_tasks: int

    async def reconfigure(self, config: Config) -> None:
        self.num_threads = config["num_threads"]
        self.max_concurrent_tasks = config["max_concurrent_tasks"]

    @fastapi.post("/")
    async def root(
        self,
        model_id: str,
        slide_path: str,
        tissue_mask_path: str,
        output_path: str,
        stride_fraction: float,
        output_bigtiff_tile_height: int,
        output_bigtiff_tile_width: int,
    ) -> str:
        from ratiopath.openslide import OpenSlide
        from ratiopath.tiling import grid_tiles

        from misc.fetch_tissue_tile import fetch_tissue_tile
        from misc.tile_heatmap_builder import TileHeatmapBuilder

        model = serve.get_app_handle(model_id)
        model_config = await model.get_config.remote()
        stride: int = round(stride_fraction * model_config["tile_size"])

        loop = asyncio.get_running_loop()
        tasks: set[asyncio.Task[Any]] = set()

        with (
            OpenSlide(slide_path) as slide,
            OpenSlide(tissue_mask_path) as tissue_slide,
            ThreadPoolExecutor(max_workers=self.num_threads) as executor,
        ):
            level = slide.closest_level(model_config["mpp"])
            mpp_x, mpp_y = slide.slide_resolution(level)
            extent_x, extent_y = slide.level_dimensions[level]

            tissue_level = tissue_slide.closest_level(model_config["mpp"])
            tissue_extent_x, tissue_extent_y = tissue_slide.level_dimensions[
                tissue_level
            ]
            scale_x = tissue_extent_x / extent_x
            scale_y = tissue_extent_y / extent_y

            mask_builder = TileHeatmapBuilder(
                extent_x=extent_x, extent_y=extent_y, mpp_x=mpp_x, mpp_y=mpp_y
            )

            async def process_tile(x: int, y: int) -> None:
                tile = await loop.run_in_executor(
                    executor,
                    fetch_tissue_tile,
                    slide,
                    tissue_slide,
                    x,
                    y,
                    level,
                    scale_x,
                    scale_y,
                    tissue_level,
                    model_config["tile_size"],
                )
                if tile is not None:
                    prediction = await model.predict.remote(tile)
                    mask_builder.update(prediction, x, y)

            for x, y in grid_tiles(
                slide_extent=(extent_x, extent_y),
                tile_extent=(model_config["tile_size"], model_config["tile_size"]),
                stride=(stride, stride),
            ):
                if len(tasks) >= self.max_concurrent_tasks:
                    _, tasks = await asyncio.wait(
                        tasks, return_when=asyncio.FIRST_COMPLETED
                    )

                tasks.add(asyncio.create_task(process_tile(x, y)))

            await asyncio.wait(tasks)

        mask_builder.save(
            output_path,
            tile_height=output_bigtiff_tile_height,
            tile_width=output_bigtiff_tile_width,
        )
        mask_builder.cleanup()
        return output_path


app = HeatmapBuilder.bind()  # type: ignore[attr-defined]
