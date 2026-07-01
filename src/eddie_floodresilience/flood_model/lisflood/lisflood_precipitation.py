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

from eddie.digitaltwin.utils import setup_logging, LogLevel
from ..flood_model_precipitation import BasePrecipitationFloodModelGenerator

setup_logging(LogLevel.DEBUG)
log = logging.getLogger(__name__)


class LisfloodPrecipitationFloodModelGenerator(BasePrecipitationFloodModelGenerator):
    """This class is to generate precipitation for LISFLOOD-FP"""

    precipitation_var_name = "rainfall_depth"
    precipitation_var_units = "mm"
    time_var_units = "hour"
    time_var_scale_factor = 1
