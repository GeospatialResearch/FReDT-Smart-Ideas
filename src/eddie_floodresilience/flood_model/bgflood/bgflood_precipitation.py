# -*- coding: utf-8 -*-
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

"""This script formats and writes precipitation data for BG-Flood inputs."""
import logging

import xarray as xr

from ..flood_model_precipitation import BasePrecipitationFloodModelGenerator, PrecipitationGenerator

log = logging.getLogger(__name__)


class BGFloodPrecipitationGenerator(PrecipitationGenerator):
    """This class is to generate precipitation"""

    def combine_precipitation_timesteps(self) -> xr.Dataset:
        """
        Read and write out all precipitation timesteps into one precipitation data

        Returns
        -------
        combined_precipitation_timestep : xr.Dataset
            Precipitation that combines all timesteps
        """
        combined_precipitation_timesteps = super().combine_precipitation_timesteps()

        # Change variable name from rainfall_depth to depth
        combined_precipitation_timesteps = combined_precipitation_timesteps.rename({
            "rainfall_depth": "depth"
        })

        # remove spatial_ref (important)
        if "spatial_ref" in combined_precipitation_timesteps:
            combined_precipitation_timesteps = combined_precipitation_timesteps.drop_vars("spatial_ref")

        # Add necessary attribute
        combined_precipitation_timesteps.attrs["Conventions"] = "CF-1.5"

        return combined_precipitation_timesteps


class BGFloodPrecipitationFloodModelGenerator(BasePrecipitationFloodModelGenerator):
    """This class is to generate precipitation for flood model"""

    precipitation_var_name = "depth"
    precipitation_var_units = "kg m-2"
    time_var_units = "second"
    time_var_scale_factor = 60 * 60  # Hours to seconds
