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

"""Classes to create input configuration files for LISFLOOD-FP."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import rioxarray as rxr
import xarray as xr
from rasterio.enums import Resampling
from scipy.ndimage import binary_dilation

from src.eddie_floodresilience.preprocessing.terrain_data_manipulator import TerrainFilter
from src.eddie_floodresilience.preprocessing.terrain_attributes_generator import TerrainAttributesGenerator


class TerrainFloodModelGenerator:
    """This class is to generate terrain data (DEM and friction) for flood model"""

    def __init__(
        self,
        flood_model_path: Path,
        hydromt_path: Path,
        river_name: str,
        terrain_crs_clipped: xr.Dataset,
        adjust_manning: bool,
        vectors: pd.DataFrame = None,
        crs: int = 2193
    ) -> None:
        """
        Generate terrain data includes DEM and friction for flood model

        Parameters
        ----------
        flood_model_path : Path
            Directory to folder storing flood model data
        hydromt_path : Path
            A directory to where all necessary files are stored to run wflow model
        river_name: str
            Name of directory to where the river information files are stored
        terrain_crs_clipped : xr.Dataset
            Clipped terrain data with projection
        adjust_manning : bool
            True means adjusting Manning's n by resampling 4m Manning's n
            False means no Mannning's n adjustment
        vectors : pd.DataFrame = None
            Name of vector file that is used to change the elevation information.
            This vector dataframe has 'value' column to specify increasing or decreasing elevation,
            and 'distance' column to specify how smooth to decrease elevation.
        crs : int = 2193
            Targeted crs. The default is 2193 for NZTM.
        """
        self.flood_model_path = flood_model_path
        self.hydromt_path = hydromt_path
        self.river_name = river_name
        self.terrain_crs_clipped = terrain_crs_clipped
        self.adjust_manning = adjust_manning
        self.vectors = vectors
        self.crs = crs

    def fill_nan_and_write_nodata(
        self,
        terrain_variable: xr.Dataset
    ) -> xr.Dataset:
        """
        Fill NaN and write nodata for a specific terrain data

        Parameters
        ----------
        terrain_variable : xr.Dataset
            Specific terrain data that needs filling with -9999 as nodata

        Returns
        -------
        xr.Dataset
            Specific terrain data that are filled with -9999 as nodata
        """
        # Fill NaN with -9999
        terrain_variable = terrain_variable.fillna(-9999)

        # Write nodata as -9999
        terrain_variable = terrain_variable.rio.write_nodata(-9999)

        return terrain_variable

    def remove_sea(
            self,
            terrain_variable: xr.Dataset
    ) -> xr.Dataset:
        """
        Remove sea from terrain data

        Parameters
        ----------
        terrain_variable : xr.Dataset
            Variable name could be 'z' (DEM) and 'zo' (roughness length)

        Returns
        -------
        terrain_vavriable : xr.Dataset
            Variable name could be 'z' (DEM) and 'zo' (roughness length).
            These terrains have the sea removed
        """
        # Get nz land polygon
        land = gpd.read_file(self.hydromt_path / 'nz_coastline.shp')

        # Remove sea using nz land polygon
        terrain_variable_no_sea = terrain_variable.rio.clip(
            land.geometry,
            crs=2193,
            drop=False
        )

        return terrain_variable_no_sea

    def format_terrain_data_pipeline(
        self,
        variable_name: str
    ) -> xr.Dataset:
        """
        Format terrain data to be used by flood model, the steps are:
            - drop spatial reference
            - make sure data have crs
            - fill nan and write nodata
            - round up values

        Parameters
        ----------
        variable_name : str
            Variable name could be 'z' (DEM) and 'zo' (roughness length)

        Returns
        -------
        terrain_variable : xr.Dataset
            Specific terrain data that are formatted
        """
        # Get specific terrain
        terrain_variable = self.terrain_crs_clipped[f'{variable_name}']

        # Drop spatial ref
        terrain_variable = terrain_variable.drop_vars("spatial_ref")

        # Make sure terrain has crs
        terrain_variable = terrain_variable.rio.write_crs(self.crs)

        # Remove sea
        terrain_variable = self.remove_sea(terrain_variable)

        # Fill NaN with -9999 and write it as nodata value
        terrain_variable = self.fill_nan_and_write_nodata(terrain_variable)

        return terrain_variable

    def strahler_for_manning_generator(self) -> None:
        """Generate Strahler order streams raster to filter river in Manning's n"""
        # Split terrain data to collect roughness raster
        TerrainFilter(
            path=self.flood_model_path,
            hydromt_path=self.hydromt_path,
            river_name=self.river_name,
            origin_filename='4m_geofabric'
        ).filter_dem_for_wflow()

        # Set up class to generate Strahler order streams
        strahler_for_manning = TerrainAttributesGenerator(
            self.flood_model_path,
            'dem',
            10,
            1000,
            origin_filename='4m_geofabric'
        )

        # Resample resolution (from 4m to 10m)
        # This can be the problem in the future if the terrain is too large,
        # the memory will be broken
        strahler_for_manning.raster_resampling_using_gdal('nn')

        # Fill depression (default is 0.0001)
        strahler_for_manning.raster_fill_depression()

        # Generate D8 pointers
        strahler_for_manning.d8_pointer_generator()

        # Generate streams for whole catchment
        strahler_for_manning.d8_stream_generator()

        # Generate Strahler order streams for manning's n
        strahler_for_manning.strahler_stream_order_generator()

    def resample_roughness(self) -> xr.DataArray:
        """
        Resample roughness from 4m to 8m.

        Returns
        -------
        xr.DataArray
            Roughness resampled to 8m
        """
        # Get roughness path
        roughness_path = self.flood_model_path / "4m_geofabric_roughness_split.tif"

        # Auto chunk size from native block size
        with rio.open(roughness_path) as src:
            block_y, block_x = src.block_shapes[0] if src.block_shapes else (4096, 4096)

        # Read roughness raster
        roughness = rxr.open_rasterio(
            roughness_path,
            chunks={'x': block_x, 'y': block_y}
        )

        # Convert -9999 to 0.004
        roughness = roughness.where(roughness != -9999, 0.004)

        # Write nodata
        roughness = roughness.rio.write_nodata(-9999)

        # Write crs
        roughness = roughness.rio.write_crs("EPSG:2193")

        # Resample to 8m
        roughness_8m = roughness.rio.reproject(
            2193,
            resolution=8,
            resampling=Resampling.nearest
        )

        # Save roughness 8m out
        roughness_outpath = self.flood_model_path / "roughness_8m.tif"
        roughness_8m.rio.to_raster(
            roughness_outpath,
            tiled=True,
            windowed=True,
            lock=False,
        )

        return roughness_8m

    def strahler_filter_generator(
        self,
        roughness: xr.DataArray
    ) -> xr.DataArray:
        """
        Generate filtered Strahler Order stream raster

        Parameters
        ----------
        roughness : xr.DataArray
            Roughness raster

        Returns
        -------
        xr.DataArray
            Mask to choose only 3 and 4 orders
        """
        # Get strahler path
        strahler_path = self.flood_model_path / "4m_geofabric_strahler_d8.tif"

        # Read strahler
        strahler = rxr.open_rasterio(strahler_path)

        # Assign crs
        strahler = strahler.rio.write_crs("EPSG:2193")

        # Reproject and rescale strahler to roughness 8m
        strahler_8m = strahler.rio.reproject_match(
            roughness,
            resampling=Resampling.nearest
        )

        # Find max order
        max_order = int(strahler_8m.max().item())

        # Filter to choose only 3 and 4 orders
        strahler_mask_8m = (strahler_8m >= 3) & (strahler_8m <= max_order)

        return strahler_mask_8m

    def roughness_to_manning(
        self,
        roughness: xr.DataArray,
        h: float = 1
    ) -> xr.DataArray:
        """
        Convert raster of roughness length to Manning's n

        Parameters
        ----------
        roughness : xr.DataArray
            Roughness length raster
        h : float = 1
            Value of depth. Default is 1

        Returns
        -------
        manning : xr.DataArray
            Manning's n raster converted from roughness length raster
        """
        # Avoid zero division
        roughness = roughness.clip(min=1e-6)

        # Avoid invalid log inputs
        ratio = h / roughness
        ratio_h_roughness = ratio.where(ratio > 1, 1)

        # Convert roughness length to Manning's n
        numerator = 0.41 * (h ** (1 / 6)) * (ratio_h_roughness - 1)
        denominator = np.sqrt(9.80665) * (1 + ratio_h_roughness * (np.log(ratio_h_roughness) - 1))
        manning_n = numerator / denominator

        # Avoid unreasonable Manning's n
        manning_n = manning_n.clip(min=1e-4, max=0.8)

        return manning_n

    def manning_adjustment(
        self,
        strahler_mask_raster: bool,
        manning: xr.DataArray
    ) -> xr.DataArray:
        """
        Filter out unreasonable Manning's n and adjust river Manning's n

        Parameters
        ----------
        strahler_mask_raster : bool
            Strahler at 8m and being filtered (only 3 and 4 orders)
        manning : xr.DataArray
            Manning's n raster converted from the roughness length raster

        Returns
        -------
        adjusted_manning : xr.DataArray
            Manning's n raster that is filtered out unreasonable values with adjusted river
        """
        # Filter out unreasonable Manning's n
        manning_filtered = manning.clip(
            min=1e-4,
            max=0.8
        )

        # Buffer the mask by 1-pixel wide to 2-pixel wide
        mask = strahler_mask_raster.values.astype(bool)
        buffered_mask = binary_dilation(mask, iterations=1)

        # Convert back to DataArray
        buffered_mask_da = xr.DataArray(
            buffered_mask,
            coords=strahler_mask_raster.coords,
            dims=strahler_mask_raster.dims
        )

        # Adjust river Manning's n.
        # At the moment, the value is kept at 0.06
        manning_river_adjusted = manning_filtered.where(
            ~buffered_mask_da,
            0.08
        )

        return manning_river_adjusted

    def manning_generator(self) -> None:
        """Generate clipped Manning's n from 4m to 8m with adjusted river"""
        if self.adjust_manning:
            # Generate Strahler Order streams and roughness length at 4m
            self.strahler_for_manning_generator()

            # Resample roughness to 8m
            roughness_8m = self.resample_roughness()

            # Filter Strahler Order streams
            strahler_mask_8m = self.strahler_filter_generator(roughness_8m)

            # Convert roughness length to Manning's n
            manning = self.roughness_to_manning(
                roughness_8m,
                1
            ) * 2

            # Adjust Manning's n
            manning_for_flood = self.manning_adjustment(
                strahler_mask_8m,
                manning
            )

        else:
            # Roughness path
            roughness_path = self.flood_model_path.parents[1] / "terrain/8m_geofabric_roughness_split.tif"

            # Read roughness raster
            roughness = rxr.open_rasterio(roughness_path)

            # Convert -9999 to 0.004
            roughness = roughness.where(roughness != -9999, 0.004)

            # Write nodata before converting
            roughness = roughness.rio.write_nodata(-9999)

            # Convert roughness length to Manning's n
            manning_for_flood = self.roughness_to_manning(
                roughness,
                1
            ) * 2

        # Clip Manning's n
        clipped_manning_for_flood = manning_for_flood.rio.clip_box(
            *self.terrain_crs_clipped.rio.bounds()
        )

        # Remove sea
        clipped_manning_for_flood = self.remove_sea(clipped_manning_for_flood)

        # Write out
        manning_outpath = self.flood_model_path / "manning.asc"
        clipped_manning_for_flood.rio.to_raster(manning_outpath)

    def write_out_terrain_data(
        self,
        variable_name: str,
        terrain_variable: xr.Dataset
    ) -> None:
        """
        Write out terrain data as ASCII file

        Parameters
        ----------
        variable_name : str
            Variable name could be 'z' (DEM) and 'zo' (roughness length)
        terrain_variable : xr.DataArray
            Specific terrain data that needs writing out
        """
        # Set up path
        terrain_variable_path = self.flood_model_path / f"{variable_name}.asc"

        # Write out as ASCII file
        terrain_variable.rio.to_raster(
            terrain_variable_path,
            dtype="float32"
        )

    def terrain_data_generator(
        self,
        variable_name: str
    ) -> None:
        """
        Generate speific terrain data

        Parameters
        ----------
        variable_name : str
            Variable name could be 'z' (DEM) and 'zo' (roughness length)
        """
        # Format terrain data
        terrain_variable = self.format_terrain_data_pipeline(variable_name)

        # Write out terrain data
        self.write_out_terrain_data(
            variable_name,
            terrain_variable
        )

    def execute_terrain_data_generator(self) -> None:
        """Generate specific terrain data"""
        if self.vectors is None:
            # Generate DEM
            self.terrain_data_generator('z')

        # Generate manning
        self.manning_generator()
