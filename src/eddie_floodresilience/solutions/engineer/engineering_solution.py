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
"""This script is to apply engineering solution"""

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rioxarray as rxr
import xarray as xr

from src.eddie_floodresilience.solutions.engineer.drainage import GenerateFullDrainage
from src.eddie_floodresilience.solutions.engineer.stopbank import GenerateFullStopbank

log = logging.getLogger(__name__)


class EngineeringSolution:
    """This class is to apply engineering solution"""

    def __init__(
        self,
        flood_model: str,
        scenario_and_id_folder: Path,
        vectors: gpd.GeoDataFrame | None = None
    ) -> None:
        """
        Apply engineering solution by changing the elevation based on the vector.
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

    @staticmethod
    def generate_dem_with_full_drainage(
        drainage_vector: pd.Series,
        dem_need_modification: xr.DataArray
    ) -> xr.DataArray:
        """
        Generate DEM with full drainage

        Parameters
        ----------
        dem_need_modification : xr.DataArray
            DEM that needs modification
        drainage_vector : pd.Series
            Drainage information

        Returns
        -------
        dem_with_full_drainage : xr.DataArray
            DEM with full drainage information
        """
        # Set up drainage function
        generate_dem_with_full_drainage = GenerateFullDrainage(
            dem_need_modification,
            drainage_vector
        )

        # Generate DEM with full drainage
        dem_with_full_drainage = generate_dem_with_full_drainage.generate_full_drainage()

        return dem_with_full_drainage

    @staticmethod
    def generate_dem_with_full_stopbank(
        dem_need_modification: xr.DataArray,
        stopbank_vector: pd.Series
    ) -> xr.DataArray:
        """
        Generate DEM with full stopbank

        Parameters
        ----------
        dem_need_modification : xr.DataArray
            DEM that needs modification
        stopbank_vector : pd.Series
            Stopbank information

        Returns
        -------
        dem_with_full_stopbank : xr.DataArray
            DEM with full stopbank information
        """
        # Set up stopbank function
        generate_dem_with_full_stopbank = GenerateFullStopbank(
            dem_need_modification,
            stopbank_vector
        )

        # Generate DEM with full stopbank
        dem_with_full_stopbank = generate_dem_with_full_stopbank.change_elevation()

        return dem_with_full_stopbank

    def apply_engineering_solution(self) -> None:
        """Apply engineering solution to elevation data"""
        # Set up log for engineering solution
        log.info("Applying engineering solution")

        # Set up original DEM that needs modification
        modified_dem = self.dem

        # Apply engineering solution to the DEM
        for i in range(self.vectors.shape[0]):
            if self.vectors.iloc[i]['type'] == 'drainage':
                modified_dem = self.generate_dem_with_full_drainage(
                    self.vectors.iloc[i],
                    modified_dem
                )
            else:
                modified_dem = self.generate_dem_with_full_stopbank(
                    modified_dem,
                    self.vectors.iloc[i]
                )

        # Hydrodynamic process path
        hydrodynamic_process_path = self.scenario_and_id_folder / "hydrodynamic_process"

        if self.flood_model == "lisflood-fp":
            # Write out
            modified_dem.rio.to_raster(
                hydrodynamic_process_path / "z.asc",
                compress="LZW",
                tiled=True
            )

        else:
            # Merge z and zo
            terrain_data = xr.Dataset({
                "z": modified_dem,
                "zo": self.roughness_length
            })

            # Add CRS
            terrain_data.rio.write_crs("EPSG:2193", inplace=True)

            # Write out
            terrain_data.to_netcdf(hydrodynamic_process_path / "8m_geofabric_clipped.nc")
