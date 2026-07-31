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
from whitebox.whitebox_tools import WhiteboxTools
from whitebox_workflows import WbEnvironment


class EngineeringSolution:
    """This class is to change the elevation"""

    def __init__(
        self,
        flood_model: str,
        scenario_and_id_folder: Path,
        vectors: pd.DataFrame | None = None
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
        vectors : pd.DataFrame | None = None
            Dataframe that contains 'vector_path', 'value', 'distance' columns:
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

    def rasterize_vector(self, vector_path: str) -> xr.DataArray:
        """
        Rasterize vector

        Parameters
        ----------
        vector_path : str
            A directory to vector

        Returns
        -------
        vector_raster : xr.DataArray
            Rasterized vector
        """
        # Read vector
        vector = gpd.read_file(vector_path)

        # Get the shape of vector
        shapes = [(geom, 1) for geom in vector.geometry]

        # Rasterize the vector
        vector_raster = rasterize(
            shapes,
            out_shape=self.dem.shape,
            transform=self.dem.rio.transform(),
            fill=0,
            dtype="uint8"
        )

        return vector_raster

    def increase_elevation(
        self,
        dem: xr.DataArray,
        mask: xr.DataArray,
        value: float
    ) -> xr.DataArray:
        """
        Increase the elevation

        Parameters
        ----------
        dem : xr.DataArray
            Elevation data read by whitebox tool
        mask : xr.DataArray
            Rasterized vector
        value : float
            A value to increase the elevation

        Returns
        -------
        increased_elevation : xr.DataArray
            Modified elevation data
        """
        # Create a copy of DEM to modify
        increased_dem = dem.copy()

        # Increase elevation data values
        increased_dem.values[mask == 1] += value

        return increased_dem

    def decrease_elevation(
        self,
        dem: xr.DataArray,
        mask: xr.DataArray,
        value: float,
        distance: float = 0
    ) -> xr.DataArray:
        """
        Decrease the elevation

        Parameters
        ----------
        dem : xr.DataArray
            Elevation data read by whitebox tool
        mask : xr.DataArray
            Rasterized vector
        value : float
            Value to decrease the elevation
        distance : float
            Rate to control the sharpness of changing elevation

        Returns
        -------
        decreased_elevation : xr.DataArray
            Modified elevation data
        """
        # Create a copy of DEM to modify
        decreased_dem = dem.copy()

        # If no smoothing by distance, sharply change
        if distance <= 0:
            decreased_dem.values[mask == 1] -= abs(value)
            return decreased_dem

        else:
            # Smoothing using distance
            # Here we create a smoothing transition weights.
            # It will mask out area (close to vector) that should be affected (close to 1)
            # and area (far away from the vector) that should not be affected (close to 0)
            dist = distance_transform_edt(mask == 0)
            weight = np.clip(1 - dist / distance, 0, 1)
            decreased_dem.values -= abs(value) * weight

        return decreased_dem

    def change_elevation(self) -> xr.DataArray:
        """
        Change the elevation

        Returns
        -------
        modified_dem : xr.DataArray
            The DEM that has its elevation changed
        """
        # Copy DEM to work separately
        modified_dem = self.dem

        # Change elevation based on each vector
        for _, row in self.vectors.iterrows():

            # Extract information for each vector
            vector_path = row["vector_path"]
            value = row["value"]
            distance = row["distance"]

            # Rasterize vector
            rasterized_vector = self.rasterize_vector(vector_path)

            # Increase elevation
            if value > 0:
                modified_dem = self.increase_elevation(
                    modified_dem,
                    rasterized_vector,
                    value
                )

            # Decrease elevation
            else:
                modified_dem = self.decrease_elevation(
                    modified_dem,
                    rasterized_vector,
                    value,
                    distance
                )

        return modified_dem

    def apply_elevation_solution(self) -> None:
        """Apply solution to elevation data"""
        # Change elevation data
        modified_dem = self.change_elevation()

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
