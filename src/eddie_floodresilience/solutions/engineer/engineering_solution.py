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

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt

from src.eddie_floodresilience.solutions.engineer.drainage import GenerateDrainageElevation, GenerateDrainageGeometry


class EngineeringSolution:
    """This class is to change the elevation"""

    def __init__(
        self,
        flood_model: str,
        scenario_and_id_folder: Path,
        vectors: gpd.GeoDataFrame | None = None
    ) -> None:
        """
        Change the elevation based on the vector.
        This class relates to functions:
        - flood_model_executor in lisflood_simulations_generator.py
        - par_generator in lisflood_parameters_generator.py
        - hydrological_and_hydrodynamic_simulation_generator in hydrological_and_hydrodynamic_pipeline.py

        Parameters
        ----------
        flood_model : str
            Either "lisflood-fp" or "bg-flood"
        scenario_and_id_folder : Path
            Directory to the scenario folder name with ID
        vectors : gpd.GeoDataFrame | None = None
            GeoDataframe that contains 'vector_path', 'value', 'distance' columns:
            - 'vector_path': Column that stores directories to specific vectors
            - 'value: Column that stores value of the vectors used to increase/decrease elevation
            - 'distance': Column that stores value to smooth the decreased elevation
        """
        self.vectors = vectors
        self.scenario_and_id_folder = scenario_and_id_folder
        self.flood_model = flood_model

        # Read terrain data
        if flood_model == "lisflood-fp":
            z_file = r"original_scenario/hydrodynamic_process/z.asc"
            with rxr.open_rasterio(self.scenario_and_id_folder.parent / z_file) as dem:
                self.dem = dem.squeeze().load()

        else:
            terrain_file = r"original_scenario/hydrodynamic_process/8m_geofabric_clipped.nc"
            with xr.open_dataset(self.scenario_and_id_folder.parent / terrain_file) as terrain_data:
                self.dem = terrain_data.z.squeeze()
                self.roughness_length = terrain_data.zo.squeeze()

    def generate_drainage(self):
        """Modify drainage"""
        # Set up function to generate DEM with drainage
        generate_dem_with_drainage = GenerateDrainageElevation(
            self.vectors,
            self.dem
        )

        # Generate new DEM with drainage and drainage elevation
        drainage_elevation, new_dem = generate_dem_with_drainage.generate_elevation_for_drainage_line()

        # Generate indices of drainage
        (horizontal_drainage_indices,
         vertical_drainage_indices) = generate_dem_with_drainage.generate_ordinal_number_of_pixels()

        # Set up function to modify drainage geometry - surface width, base width, and slope
        generate_dem_with_drainage_geometry = GenerateDrainageGeometry(
            self.dem,
            new_dem,
            self.vectors,
            horizontal_drainage_indices,
            vertical_drainage_indices,
            drainage_elevation,
            self.vectors['base_width'],
            self.vectors['surface_width'],
            self.vectors['slope']
        )

        # Generate DEM with drainage geometry
        dem_drainage_geometry = generate_dem_with_drainage_geometry.generate_dem_with_drainage_geometry()

