import icechunk as ic
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_zeros, spatial_dims
from shapely import box
from pyproj import CRS, Transformer
import numpy as np
import asyncio
from icechunk.xarray import to_icechunk


def transform_bounds(
    initial_crs: str | CRS,
    target_crs: str | CRS,
    bounds: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Transforms bounds from one CRS to another."""
    if isinstance(initial_crs, str):
        initial_crs = CRS.from_string(initial_crs)
    if isinstance(target_crs, str):
        target_crs = CRS.from_string(target_crs)
    transformer = Transformer.from_crs(initial_crs, target_crs, always_xy=True)
    xmin, ymin = transformer.transform(bounds[0], bounds[1])
    xmax, ymax = transformer.transform(bounds[2], bounds[3])
    return xmin, ymin, xmax, ymax


def get_geobox(bbox: box, proj_crs: str | CRS, shape: tuple[int, int]) -> GeoBox:
    """Returns a geobox in proj_crs from a bounding box, with a target projected CRS and resolution to create the pixel grid."""
    xmin, ymin, xmax, ymax = transform_bounds("EPSG:4326", proj_crs, bbox.bounds)

    return GeoBox.from_bbox((xmin, ymin, xmax, ymax), crs=proj_crs, shape=shape)


async def get_written_chunk_set(
    repo: ic.Repository, band_name: str = "foo"
) -> set[tuple[int, ...]]:
    """Returns a set of chunk indices that have been written to the repository."""
    session = repo.readonly_session("main")
    chunk_set = set()

    written_chunks = session.chunk_coordinates(f"/{band_name}")

    async for chunk in written_chunks:
        chunk_set.add(chunk)

    return chunk_set


async def main():
    bbox = box(35.44, -16.12, 35.84, -15.74)

    geobox = get_geobox(bbox, "EPSG:3857", (500, 500))

    template = xr_zeros(geobox, chunks=-1, dtype=np.float32)  # type: ignore

    y_dim, x_dim = spatial_dims(template)

    # we define some arbritrary band for encoding + chunks
    band_dict = {
        "foo": ((y_dim, x_dim), template.data, {"grid_mapping": "spatial_ref"})
    }

    coords = {
        "spatial_ref": template.spatial_ref,
        y_dim: template.coords[y_dim].data,
        x_dim: template.coords[x_dim].data,
    }

    band_encoding = {"foo": {"chunks": (50, 50), "write_empty_chunks": True}}

    ds = xr.Dataset(band_dict, coords=coords)

    storage = ic.in_memory_storage()
    repo = ic.Repository.create(storage)

    session = repo.writable_session("main")

    print(f"Writing template to repository")
    print(f"Template:\n{ds}")

    ds.to_zarr(
        session.store,
        compute=False,
        mode="w",
        encoding=band_encoding,
        consolidated=False,
    )

    session.commit("Write template")

    # now we get the written chunks, check the length
    written_chunk_set = await get_written_chunk_set(repo)
    print(f"Number of written chunks: {len(written_chunk_set)}")

    session_2 = repo.writable_session("main")

    # now we can write one chunk of purely NaNs
    nan_array = np.full(template.data.shape, np.nan)

    new_ds = xr.Dataset(
        {
            "foo": (
                (y_dim, x_dim),
                nan_array,
                {"grid_mapping": "spatial_ref"},
            )
        },
        coords=coords,
    )

    region = {y_dim: slice(0, 500), x_dim: slice(0, 500)}

    to_icechunk(
        new_ds.drop_vars("spatial_ref", errors="ignore"), session_2, region=region
    )

    session_2.commit("Write NaNs")

    written_chunk_set_2 = await get_written_chunk_set(repo)
    print(f"Number of written chunks: {len(written_chunk_set_2)}")


if __name__ == "__main__":
    asyncio.run(main())
