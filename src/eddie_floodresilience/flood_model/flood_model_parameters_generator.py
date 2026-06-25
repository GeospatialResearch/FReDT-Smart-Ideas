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

from eddie.digitaltwin.setup_environment import get_database
from eddie.digitaltwin.utils import LogLevel, setup_logging
from src.eddie_floodresilience.tables import FloodModelOutput

setup_logging(LogLevel.DEBUG)
log = logging.getLogger(__name__)


class FloodModelParametersGenerator(ABC):
    """This class is to generate parameter files for flood models"""

    def __init__(
        self,
        flood_model_path: Path,
        terrain_bounding_box: Polygon,
        start_time: datetime,
        end_time: datetime,
        polygons: str = None,
        vectors: str = None
    ) -> None:
        """
        Generate parameter files for flood model

        Parameters
        ----------
        flood_model_path : Path
            Directory to folder storing flood model data
        terrain_bounding_box : Polygon
            Bounding's box of terrain data
        start_time : datetime
            Starting time details. Format is "yyyy-mm-ddThh:mm:ss"
        end_time : datetime
            Ending time details.
        polygons : str = None
            Name of polygon file that is used to change the landcover information.
            This polygon dataframe has 'landcover' column with new values
        vectors : str = None
            Name of vector file that is used to change the elevation information.
            This vector dataframe has 'value' column to specify increasing or decreasing elevation,
            and 'distance' column to specify how smooth to decrease elevation.
        """
        self.flood_model_path = flood_model_path
        self.terrain_bounding_box = terrain_bounding_box
        self.start_time = start_time
        self.end_time = end_time
        self.polygons = polygons
        self.vectors = vectors
        self.injection_points_flow = pd.read_csv(self.flood_model_path / "injection_points_flow.csv")
        self.seconds = int((end_time - start_time).total_seconds())

    @staticmethod
    def move_points_inside_aoi(
        aoi_coords: list[float],
        xy_coords: list[float],
        buffer_distance: float,
        tolerance: float
    ) -> tuple[float]:
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
        xy_coords : tuple[float, gloat]
            New coordinates of x and y. X and y of each injection point
        """
        log.info("Moving points inside aoi")
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

    def reserve_flood_model_id(self) -> int:
        """
        Find the next output ID to be used in database and directory and reserve it.

        Returns
        -------
        int
            The new flood model ID
        """
        engine = get_database()
        with engine.connect() as conn:
            # Create the 'flood_model_output' table in the database if it doesn't exist
            create_table(conn, FloodModelOutput)
            # Create a new query object to reserve the flood model id
            query = insert(FloodModelOutput).values(geometry=self.terrain_bounding_box.wkt)
            # Execute the query to store the BG Flood model output metadata in the database while retrieving id
            result = conn.execute(query)
        model_id = result.inserted_primary_key[0]
        # Log a message indicating the successful storage of Flood model output metadata in the database
        log.info(f"Flood model ID {model_id} reserved.")
        return model_id

    def file_increment_generator(
        self,
        filename: str
    ) -> Path:
        """
        Generate increasing files

        Parameters
        ----------
        filename : str
            Filename accords to scenario and order of that scenario

        Returns
        -------
        file_directory : Path
            A directory to the file which has just been named
        """
        number_ids = [
            int(f.stem.split("_")[-1])
            for f in Path(self.flood_model_path).glob(f"{filename}_*")
        ]

        # Get next output
        file_number = max(number_ids, default=0) + 1

        # Output path
        file_directory = Path(self.flood_model_path) / f"{filename}_{file_number:03d}"

        return file_directory

    def output_folder_generator(self) -> Path:
        """
        Set up options for outputs according to scenarios
        and create output directory for flood modelling outputs

        Returns
        -------
        output_directory : Path
            Directory of flood model outputs
        """
        model_id = self.reserve_flood_model_id()
        log.info("Generating output folder")
        if self.polygons is not None and self.vectors is not None:
            output_name = "output_landcover_elevation"
        elif self.polygons is not None:
            output_name = "output_landcover"
        elif self.vectors is not None:
            output_name = "output_elevation"
        else:
            output_name = "output"

        output_dir = self.flood_model_path / f"{output_name}_{model_id:03d}"
        output_dir.mkdir(exist_ok=False, parents=True)
        return output_dir

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
    def write_injection_point_files(self, output_dir: Path) -> None:
        """
        Write injection point files for flood model.

        Parameters
        ----------
        output_dir : Path
            The output directory for the flood model output.
        """

    @abstractmethod
    def write_flood_model_parameter_file(self) -> Path:
        """Write parameter files for flood model."""
