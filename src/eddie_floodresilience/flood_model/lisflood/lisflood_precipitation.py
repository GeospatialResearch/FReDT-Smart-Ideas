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

"""This script formats and writes precipitation data for LISFLOOD-FP inputs."""

import logging

import xarray as xr

from eddie.digitaltwin.utils import setup_logging, LogLevel
from ..flood_model_precipitation import BasePrecipitationFloodModelGenerator, BasePrecipitationGenerator

setup_logging(LogLevel.DEBUG)
log = logging.getLogger(__name__)


class LisfloodPrecipitationGenerator(BasePrecipitationGenerator):
    """This class is to generate precipitation for LISFLOOD-FP"""

    def combine_precipitation_timesteps(self) -> xr.Dataset:
        """
        Read and write out all precipitation timesteps into one precipitation data

        Returns
        -------
        combined_precipitation_timestep : xr.Dataset
            Precipitation that combines all timesteps
        """  # pylint: disable=duplicate-code
        log.info("Combining all precipitation timesteps")
        # Collect all files of precipitation timesteps
        precipitation_timesteps_files = self.collect_precipitation_timesteps()

        # Read/combine all files of precipitation timesteps
        combined_precipitation_timesteps = xr.open_mfdataset(
            precipitation_timesteps_files,
            combine='nested',
            concat_dim='time'
        )

        return combined_precipitation_timesteps


class LisfloodPrecipitationFloodModelGenerator(BasePrecipitationFloodModelGenerator):
    """This class is to generate precipitation for LISFLOOD-FP"""

    precipitation_var_name = "rainfall_depth"
    precipitation_var_units = "mm"
    time_var_units = "hour"
    time_var_scale_factor = 1
