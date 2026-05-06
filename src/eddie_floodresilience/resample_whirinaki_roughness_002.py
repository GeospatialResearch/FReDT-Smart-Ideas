from pathlib import Path
from src.eddie_floodresilience.preprocessing.terrain_data_manipulator import TerrainFilter
from src.eddie_floodresilience.preprocessing.terrain_attributes_generator import TerrainAttributesGenerator


import xarray

roughness_4m_path = Path(r"D:\Digital_Twin_data\roughness")


terrain_filter = TerrainFilter(
    roughness_4m_path
)

terrain_filter.filter_dem_for_wflow()



stream_topology_data = TerrainAttributesGenerator(
    roughness_4m_path,
    'dem',
    10,
    1000
)

stream_topology_data.raster_resampling('nn')

stream_topology_data.raster_fill_depression(0.0001)

stream_topology_data.d8_pointer_generator()

stream_topology_data.d8_stream_generator(catchment_area=True)

stream_topology_data.strahler_stream_order_generator()




# -----------------------------------------------------------------


import rioxarray as rxr
from rasterio.enums import Resampling

main_dir = r"D:\Digital_Twin_data\roughness"

roughness = rxr.open_rasterio(fr"{main_dir}\roughness_split.tif")


# Explicitly set NoData
roughness = roughness.rio.write_nodata(-9999)

# Resample to 8 m
roughness_8m = roughness.rio.reproject(
    roughness.rio.crs,      # keep same CRS
    resolution=8,           # target resolution (8 m)
    resampling=Resampling.nearest  # good for roughness
)

# Save output
roughness_8m.rio.to_raster(r"D:\Digital_Twin_data\roughness\roughness_8m.tif")


# ---------------------------------------------------------------------------------

import geopandas as gpd
from rasterio.features import shapes

strahler = rxr.open_rasterio(fr"{main_dir}\strahler_d8.tif").squeeze()

# # Mask: True where Strahler is 3 or 4
# strahler_mask = (strahler == 3) | (strahler == 4)
#
# strahler_filtered = strahler.where(strahler_mask)
#
# strahler_filtered.rio.write_nodata(-9999, inplace=True)
# strahler_filtered.rio.to_raster(fr"{main_dir}\strahler_3_4.tif")
#
# # Convert mask to integer (needed for shapes)
# mask_int = strahler_mask.astype("uint8")
#
# results = (
#     {
#         "type": "Feature",
#         "geometry": geom,
#         "properties": {"value": int(value)},
#     }
#     for geom, value in shapes(
#         mask_int.values,
#         mask=mask_int.values,
#         transform=strahler.rio.transform()
#     )
#     if value == 1
# )
#
# gdf = gpd.GeoDataFrame.from_features(list(results), crs=strahler.rio.crs)
#
# gdf.to_file(fr"{main_dir}\strahler_3_4.shp")








from rasterio.enums import Resampling

strahler_8m = strahler.rio.reproject_match(roughness_8m, resampling=Resampling.nearest)

strahler_mask_8m = (strahler_8m == 3) | (strahler_8m == 4)


# Convert roughness to manning's n
import xarray as xr
import numpy as np

def roughness_to_manning(
    roughness: xr.DataArray,
    h: float = 1
) -> None:
    """
    Convert raster of roughness to manning's n

    Parameters
    ----------
    roughness : Any
        A raster of roughness data
    h : float = 1
        Value of depth. Default is 1
    """
    # Avoid division by zero / negative values
    roughness_safe = roughness.where(roughness > 1e-6)

    # Set up ratio
    ratio = h / roughness_safe

    # Avoid invalid log inputs
    ratio_h_roughness = ratio.where(ratio > 1)

    numerator = 0.41 * (h ** (1 / 6)) * (ratio_h_roughness - 1)
    denominator = np.sqrt(9.80665) * (1 + ratio_h_roughness * (np.log(ratio_h_roughness) - 1))
    manning_n = numerator / denominator

    return manning_n

manning = roughness_to_manning(
    roughness_8m,
    0.3
)

# Filter out unreasonable manning
manning_clean = manning.clip(min=0.01, max=0.2)

manning_updated = manning_clean.where(~strahler_mask_8m, 0.06)

manning_updated.rio.to_raster(fr"{main_dir}\roughness_8m_rivers_0p06.tif")