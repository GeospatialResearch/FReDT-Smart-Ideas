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

"""This script sets parameter files for flood model runs."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import pandas as pd
from shapely.geometry import Polygon

from eddie.digitaltwin.utils import LogLevel, setup_logging

setup_logging(LogLevel.DEBUG)
log = logging.getLogger(__name__)


class FloodModelParametersGenerator(ABC):
    """This class is to generate parameter files for flood models"""  # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        flood_model_path: Path,
        hydromt_path: Path,
        terrain_bounding_box: Polygon,
        start_time: datetime,
        end_time: datetime,
        polygons: str = None,
        vectors: pd.DataFrame = None
    ) -> None:
        """
        Generate parameter files for flood model

        Parameters
        ----------
        flood_model_path : Path
            Directory to folder storing flood model data
        hydromt_path : Path
            Directory to folder storing necessary data
        terrain_bounding_box : Polygon
            Bounding's box of terrain data
        start_time : datetime
            Starting time details. Format is "yyyy-mm-ddThh:mm:ss"
        end_time : datetime
            Ending time details.
        polygons : str = None
            Name of polygon file that is used to change the landcover information.
            This polygon dataframe has 'landcover' column with new values
        vectors : pd.DataFrame = None
            Name of vector file that is used to change the elevation information.
            This vector dataframe has 'value' column to specify increasing or decreasing elevation,
            and 'distance' column to specify how smooth to decrease elevation.
        """
        self.flood_model_path = flood_model_path
        self.hydromt_path = hydromt_path
        self.terrain_bounding_box = terrain_bounding_box
        self.start_time = start_time
        self.end_time = end_time
        self.polygons = polygons
        self.vectors = vectors
        self.injection_points_flow = pd.read_csv(self.flood_model_path / "injection_points_flow.csv")
        self.seconds = int((end_time - start_time).total_seconds())

    @abstractmethod
    def parameter_files_generator(self) -> Path:
        """
        Generate parameter files for flood model

        Returns
        -------
        Path
            Directory of output files as configured in parameter files
        """

    @abstractmethod
    def write_injection_point_files(self) -> None:
        """Write injection point files for flood model"""

    @abstractmethod
    def write_flood_model_parameter_file(self) -> Path:
        """
        Write parameter files for flood model.

        Returns
        -------
        Path
            The directory the parameter was created in.
        """

    @staticmethod
    def move_points_inside_aoi(
        aoi_coords: list[float],
        xy_coords: list[float],
        buffer_distance: float,
        tolerance: float
    ) -> tuple[float, float]:
        """
        Move points inside aoi

        Parameters
        ----------
        aoi_coords : list[float]
            Coordinates of area of interest
        xy_coords : list[float]
            Coordinates of x and y. X and y of each injection point
        buffer_distance : float
            Amount that the points are moved inside
        tolerance : float
            How much differences in distance between injection points and coordinates of area of interest

        Returns
        -------
        xy_coords : tuple[float]
            New coordinates of x and y. X and y of each injection point
        """
        # Extract each x and y
        x, y = xy_coords

        # Extract xmin, ymin, xmax, ymax
        xmin, ymin, xmax, ymax = aoi_coords

        # Move points inside DEM
        if abs(y - ymax) <= tolerance:
            y -= buffer_distance

        elif abs(y - ymin) <= tolerance:
            y += buffer_distance

        elif abs(x - xmax) <= tolerance:
            x -= buffer_distance

        elif abs(x - xmin) <= tolerance:
            x += buffer_distance

        return x, y

    def optional_output_generator(self) -> Path:
        """
        Set up options for outputs according to sceanrios
        and create output directory for flood modelling outputs

        Returns
        -------
        output_directory : Path
            Directory of flood model outputs
        """
        # Set up flood model output path
        output = self.flood_model_path / "output"

        # Create output (if not available)
        output.mkdir(parents=True, exist_ok=True)

        return output
