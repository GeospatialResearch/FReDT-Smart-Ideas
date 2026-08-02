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
"""This script is for modifying stopbank"""

import logging

import numpy as np
import pandas as pd
import xarray as xr
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
from whitebox.whitebox_tools import WhiteboxTools
from whitebox_workflows import WbEnvironment

log = logging.getLogger(__name__)

# pylint: disable=duplicate-code
wbe = WbEnvironment()
wbe.verbose = True
wbe.max_procs = -1

wbt = WhiteboxTools()


class GenerateFullStopbank:
    """This class is to change the elevation"""

    def __init__(
        self,
        dem: xr.DataArray,
        vector: pd.Series | None = None
    ) -> None:
        """
        Change the stopbank elevation.
        This class relates to functions:
        - flood_model_executor in lisflood_simulations_generator.py
        - par_generator in lisflood_parameters_generator.py
        - hydrological_and_hydrodynamic_simulation_generator in hydrological_and_hydrodynamic_pipeline.py

        Parameters
        ----------
        dem : xr.DataArray
            DEM
        vector : pd.Series | None = None
            Dataframe that contains 'vector_path', 'value', 'distance' columns:
            - 'vector_path': Column that stores directories to specific vectors
            - 'value: Column that stores value of the vectors used to increase/decrease elevation
            - 'distance': Column that stores value to smooth the decreased elevation
        """
        self.vector = vector
        self.dem = dem

    def rasterize_vector(self) -> xr.DataArray:
        """
        Rasterize vector

        Returns
        -------
        vector_raster : xr.DataArray
            Rasterized vector
        """
        # Rasterize the vector
        vector_raster = rasterize(
            [(self.vector.geometry, 1)],
            out_shape=self.dem.shape,
            transform=self.dem.rio.transform(),
            fill=0,
            dtype="uint8"
        )

        return vector_raster

    @staticmethod
    def increase_elevation(
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
        modified_dem = self.dem.copy(deep=True)

        # Rasterize vector
        rasterized_vector = self.rasterize_vector()

        # Increase elevation
        if self.vector['value'] > 0:
            modified_dem = self.increase_elevation(
                modified_dem,
                rasterized_vector,
                self.vector['value']
            )

        # Decrease elevation
        else:
            modified_dem = self.decrease_elevation(
                modified_dem,
                rasterized_vector,
                self.vector['value'],
                self.vector['distance']
            )

        return modified_dem
