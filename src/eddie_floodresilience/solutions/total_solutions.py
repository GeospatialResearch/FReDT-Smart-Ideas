from osgeo import gdal
import rioxarray as rxr
import xarray as xr
from pathlib import Path
import geopandas as gpd
from rasterio.features import rasterize

import numpy as np


class LandCoverSolution():
    """This class is to change the land cover based on polygons"""

    def __init__(
            self,
            hydro_combination_path: Path,
            hydromt_path: Path,
            polygons: str = None
    ) -> None:
        """
        Change the land cover based on polygons

        Parameters
        ----------
        hydro_combination_path : Path
            Directory to folder storing all necessary data
        hydromt_path : Path
            A directory to where all necessary files are stored to run wflow model
        polygons : str = None
            Polygons that are used to change the landcover information
        """
        self.hydro_combination_path = hydro_combination_path
        self.hydromt_path = hydromt_path
        self.polygons = polygons

    def rasterize_polygons(
            self,
            current_landcover: xr.DataArray,
            polygons: gpd.GeoDataFrame
    ) -> xr.DataArray:
        """
        This is to apply values to each polygon under raster format

        Parameters
        ----------
        current_landcover : xr.DataArray
            Raster of current land cover from LCDB-converted Global cover
        polygons : gpd.GeoDataFrame
            Polygons that are used to change the landcover information.
            This polygon dataframe has 'landcover' column with new values

        Returns
        -------
        modified_landcover : xr.DataArray
            Raster of land cover that is modified
        """
        # Copy original land cover data to not be affected by the change
        modified_landcover = current_landcover.copy()

        # Create rasterization shapes
        shapes = [
            (geom, value)
            for geom, value in zip(
                polygons.geometry,
                polygons["landcover"]
            )
        ]

        # Rasterize all polygons at once
        polygon_raster = rasterize(
            shapes=shapes,
            out_shape=current_landcover.shape,
            transform=current_landcover.rio.transform(),
            fill=0,
            dtype='uint8'
        )

        # Applyg changes
        mask = polygon_raster != 0
        modified_landcover.values[mask] = polygon_raster[mask]

        return modified_landcover


    def landcover_change(self):
        """This is to change the landcover based on polygons"""
        # Read current land cover data
        with rxr.open_rasterio(self.hydromt_path / r'globcover_origin.tif') as current_landcover:
            current_landcover = current_landcover.squeeze().load()

        # Read polygons
        polygons_path = self.hydro_combination_path / self.polygons
        polygons = gpd.read_file(polygons_path)

        # Convert crs
        # This step will be removed in future
        polygons_crs = polygons.to_crs(4326)

        # Rasterize and apply new values to current land cover
        modified_landcover = self.rasterize_polygons(
            current_landcover,
            polygons_crs
        )

        # Write out new land cover
        modified_landcover.rio.to_raster(
            self.hydromt_path / "globcover.tif",
            compress="LZW",
            tiled=True,
            BIGTIFF="IF_SAFER"
        )

