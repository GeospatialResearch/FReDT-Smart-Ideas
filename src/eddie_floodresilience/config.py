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

"""Collection of utils that are used for system and environment configuration."""

import pathlib

from eddie.config import EnvVariable as EnvVarBase


class EnvVariable(EnvVarBase):  # pylint: disable=too-few-public-methods
    """Encapsulates all environment variable fetching, ensuring proper defaults and types."""

    NIWA_API_KEY = EnvVarBase._get_env_variable("NIWA_API_KEY")

    DATA_DIR = pathlib.Path(EnvVarBase._get_env_variable("DATA_DIR"))
    DATA_DIR_MODEL_OUTPUT = pathlib.Path(EnvVarBase._get_env_variable("DATA_DIR_MODEL_OUTPUT"))
    FLOOD_MODEL_DIR = pathlib.Path(EnvVarBase._get_env_variable("FLOOD_MODEL_DIR"))

    HYDRO_COMBINATION_PATH = pathlib.Path(EnvVarBase._get_env_variable("HYDRO_COMBINATION_PATH"))
    OUTLET_GAUGE_LOCATIONS_FILENAME = EnvVarBase._get_env_variable("OUTLET_GAUGE_LOCATIONS_FILENAME")

    # NewZealidar config that we must ensure have values.
    _LIDAR_DIR = EnvVarBase._get_env_variable("LIDAR_DIR")
    _DEM_DIR = EnvVarBase._get_env_variable("DEM_DIR")
    _LAND_FILE = EnvVarBase._get_env_variable("LAND_FILE", allow_empty=True)
    _INSTRUCTIONS_FILE = EnvVarBase._get_env_variable("INSTRUCTIONS_FILE")
