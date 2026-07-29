# -*- coding: utf-8 -*-
# Copyright © 2021-2026 Geospatial Research Institute Toi Hangarau
# LICENSE: https://github.com/GeospatialResearch/Digital-Twins/blob/master/LICENSE
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Shared generic raster operations for polygonization."""
import geopandas as gpd
import numpy as np
import rasterio as rio
import shapely
import xarray as xr


def polygonize_raster(
    raster: xr.DataArray, mask: np.ndarray | None = None, column_name: str | None = None
) -> gpd.GeoDataFrame:
    """
    Take a raster and identify polygons that all share the same value.

    Parameters
    ----------
    raster : xr.DataArray
        Raster to search for polygons.
    mask : np.ndarray | None = None
        A mask of the area to look for polygons, if None then search the whole raster.
    column_name : str | None
        The column in the output data frame to share the value of each polygon. If None then this column will not exist.

    Returns
    -------
    gpd.GeoDataFrame
        The polygons that all share the same value, with optional column "column_name" set to the value of each polygon.
    """
    polygons = rio.features.shapes(raster, mask=mask, transform=raster.rio.transform())

    polygons_records = []
    # Add each polygon to a list in a form ready to be ingested into a GeoDataFrame to be returned
    for polygon, val in polygons:
        new_row = {"geometry": shapely.geometry.shape(polygon)}
        if column_name is not None:
            new_row[column_name] = val
        polygons_records.append(new_row)

    gdf = gpd.GeoDataFrame(polygons_records, crs=raster.rio.crs.wkt)
    return gdf
