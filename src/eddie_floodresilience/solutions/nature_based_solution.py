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

"""Apply interventions and solutions for flooding that will from scenarios."""
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
from whitebox.whitebox_tools import WhiteboxTools
from whitebox_workflows import WbEnvironment


log = logging.getLogger(__name__)

GLOBCOVER_CLASSES: dict[str, int] = {
    "Dense Deciduous Forest": 50,
    "Evergreen Forest": 40,
    "Deciduous Forest": 60,
    "Needleleaf Forest": 70,
    "Pasture Mosaic": 120,
    "Shrubland": 130,
    "Pasture": 140,
    "Sparse Vegetation": 150,
    "Wetland": 160,
    "Bare Land": 200,
}

LCDB_CLASSES: dict[str, int] = {
    "High producing Exotic Grassland": 40,
    "Low Producing Grassland": 41,
    "Herbaceous Freshwater Vegetation": 45,
    "Manuka and/or Kanuka": 52,
    "Broadleaved Indigenous Hardwoods": 54,
    "Forest - Harvested": 64,
    "Deciduous Hardwoods": 68,
    "Indigenous Forest": 69,
    "Exotic Forest (needleleaf forest)": 71,
}

wbe = WbEnvironment()
wbe.verbose = True
wbe.max_procs = -1

wbt = WhiteboxTools()


class NatureBasedSolution:
    """This class is to change the land cover based on polygons"""

    def __init__(
        self,
        hydromt_path: Path,
        scenario_and_id_folder: Path,
        landcover: str = 'globcover',
        polygons: gpd.GeoDataFrame | None = None
    ) -> None:
        """
        Change the land cover based on polygons.
        This class relates to functions:
        - landcover_section in wflow_data_catalog_generator.py
        - par_generator in lisflood_parameters_generator.py
        - hydrological_and_hydrodynamic_simulation_generator in hydrological_and_hydrodynamic_pipeline.py

        Parameters
        ----------
        hydromt_path : Path
            A directory to where all necessary files are stored to run wflow model
        landcover : str = 'globcover'
            Name of land cover dataset. Default is 'globcover'
        scenario_and_id_folder : Path
            Directory to the scenario folder name with ID
        polygons : gpd.GeoDataFrame = None
            Polygons that are used to change the landcover information
        """
        self.hydromt_path = hydromt_path
        self.scenario_and_id_folder = scenario_and_id_folder
        self.landcover = landcover
        self.polygons = polygons

    def rasterize_polygons(
        self,
        current_landcover: xr.DataArray,
        polygons: gpd.GeoDataFrame
    ) -> xr.DataArray:
        """
        Apply values to each polygon under raster format

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
        if "landcover" not in polygons.columns:
            polygons["landcover"] = polygons["landcover_name"].map(LCDB_CLASSES)

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

    def apply_landcover_solution(self) -> Path:
        """
        Change the landcover based on polygons.

        Returns
        -------
        Path
            Directory to the modified landcover.
        """
        # Set up land cover features based on chosen land cover
        if self.landcover.startswith('globcover'):
            original_landcover = 'original_globcover.tif'
            crs = 4326
            folder_landcover = 'globcover'
        else:
            original_landcover = 'original_lcdb.tif'
            crs = 2193
            folder_landcover = 'lcdb'

        # Read current land cover data
        with rxr.open_rasterio(self.hydromt_path / original_landcover) as current_landcover:
            current_landcover = current_landcover.squeeze().load()

        # Convert crs
        # This step will be removed in future
        polygons_crs = self.polygons.to_crs(crs)

        # Rasterize and apply new values to current land cover
        modified_landcover = self.rasterize_polygons(
            current_landcover,
            polygons_crs
        )

        # self.hydromt_path may not be writable on linux, so write to self.hydro_combination_path
        globcover_dir = self.scenario_and_id_folder / 'hydrological_process' / folder_landcover
        globcover_dir.mkdir(parents=True, exist_ok=True)

        # Set up the path for new land cover with scenario and ID
        output_path = globcover_dir / f"{folder_landcover}_{self.scenario_and_id_folder.name}.tif"

        # Write out new land cover
        modified_landcover.rio.to_raster(
            output_path,
            compress="LZW",
            tiled=True,
            BIGTIFF="IF_SAFER"
        )

        return output_path
