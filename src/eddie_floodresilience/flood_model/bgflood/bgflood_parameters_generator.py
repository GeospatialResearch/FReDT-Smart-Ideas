# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 15:16:47 2026

@author: mng42
"""

from osgeo import gdal  # Import gdal before rasterio

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, Point

import pandas as pd

from datetime import datetime

from pathlib import Path

from textwrap import dedent

import shutil
import logging
from eddie.digitaltwin.utils import setup_logging, LogLevel

setup_logging(LogLevel.DEBUG)
log = logging.getLogger(__name__)


class ParametersFloodModelGenerator:
    """This class is to generate parameter files for flood model"""

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

        self.seconds = int((end_time - start_time).total_seconds())

        self.polygons = polygons
        self.vectors = vectors

        # Read injection points flow csv
        self.injection_points_flow = pd.read_csv(self.flood_model_path / "injection_points_flow.csv")
        # Add column that converts time to seconds
        self.injection_points_flow['time_in_second'] = np.arange(
            0,
            3600 * (len(self.injection_points_flow)),
            3600
        )

        self.injection_points = gpd.read_file(self.flood_model_path / "injection_points.shp")

    def output_folder_generator(self):
        """Generate output folder"""
        log.info("Generating output folder")
        if self.polygons is not None and self.vectors is not None:
            output_name = "output_landcover_elevation"
        elif self.polygons is not None:
            output_name = "output_landcover"
        elif self.vectors is not None:
            output_name = "output_elevation"
        else:
            output_name = "output"

        prefix = output_name + "_"

        next_id = (
            max(
                [
                    int(f.name.split("_")[-1])
                    for f in self.flood_model_path.iterdir()
                    if f.is_dir()
                       and f.name.startswith(prefix)
                       and f.name.split("_")[-1].isdigit()
                ],
                default=0
            )
            + 1
        )

        self.output_folder = self.flood_model_path / f"{output_name}_{next_id:03d}"
        self.output_folder.mkdir(exist_ok=False)

    def flow_text_file_generator(self):
        """Generate flow text data for BG-Flood"""
        log.info("Generating flow text data")
        # Create a loop to generate flow text data for BG-Flood
        for col in self.injection_points_flow.columns:

            # Choose only flow text columns
            if col.startswith("Q_"):
                # Remove the Q before the flow ID
                fid = col.split("_")[1]  # e.g. Q_97 → 97

                # Extract time
                times = self.injection_points_flow["time_in_second"].values

                # Extract values
                values = self.injection_points_flow[col].values

                # Stack them up
                out = np.column_stack([times, values])

                # Write out into text data
                # <i want to use the output folder path here>

                np.savetxt(
                    self.output_folder / f"river{fid}.txt",
                    out,
                    fmt="%.6f",
                    delimiter="\t"
                )

    def tide_text_file_design(
        self,
        direction: str
    ):
        """
        Design codes for generating tide text data

        Parameters
        ----------
        direction : str
            Four edges of DEM - top, bottom, left, and right
        """
        with open(self.flood_model_path / f"{self.output_folder}/{direction}_bnd.txt", "w") as f:
            # write header
            f.write("# Water level boundary\n")

            # loop through your dataframe
            for t in self.injection_points_flow["time_in_second"]:
                f.write(f"{int(t)}\t0.0\n")

    def tide_text_file_generator(self):
        """Generate tide text data for four edges for BG-Flood"""
        log.info("Writing tide boundary configuration.")
        self.tide_text_file_design('top')
        self.tide_text_file_design('bottom')
        self.tide_text_file_design('left')
        self.tide_text_file_design('right')

    def move_points_inside_aoi(
        self,
        aoi_coords,
        xy_coords,
        buffer_distance,
        tolerance
    ):
        """
        Move points inside aoi

        Parameters
        ----------
        aoi_coords : list
            Coordinates of area of interest
        xy_coords : list
            Coordinates of x and y. X and y of each injection point
        buffer_distance : float
            Amount that the points are moved inside
        tolerance : float
            How much differences in distance between injection points and coordinates of area of interest

        Returns
        -------
        xy_coords : list
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

    def pixel_bounds_from_centroid(
        self,
        point
    ) -> tuple[float, float, float, float]:
        """
        Identify pixel bounds for flow through centroid

        Parameters
        ----------
        point : Point
            Centroid

        Returns
        -------
        pixel_bounds: tuple[float, float, float, float]
            Pixel bounds as (xmin, xmax, ymin, ymax)
        """
        # Get coordinates of centroids
        x, y = point.x, point.y

        # Move x, y inside
        x, y = self.move_points_inside_aoi(
            list(self.terrain_bounding_box.bounds),
            [x, y],
            buffer_distance=12,
            tolerance=5
        )

        # Calculate pixel bounds
        pixel_bounds = (
            x - 4,  # xmin
            x + 4,  # xmax
            y - 4,  # ymin
            y + 4  # ymax
        )

        return pixel_bounds

    def flow_text_data_design(
        self,
        flow_id: str,
        pixel_bounds: tuple[float, float, float, float]
    ) -> str:
        """
        Design river text data for BG Flood parameter file

        Parameters
        ----------
        flow_id : str
            ID of the flow data
        pixel_bounds: tuple[float, float, float, float]
            Pixel bounds as (xmin, xmax, ymin, ymax)

        Returns
        -------
        flow_text : str
            Name of flow data and its coordinates
        """
        # Format flow text data
        xmin, xmax, ymin, ymax = pixel_bounds

        # Design flow text
        flow_text = (
            f"river = river{flow_id[2:]}.txt,"
            f"{xmin:.1f},{xmax:.1f},{ymin:.1f},{ymax:.1f};"
        )

        return flow_text

    def flow_text_data_generator(
        self
    ) -> list:
        """
        Generate river text data for BG-Flood parameter file

        Returns
        -------
        flow_lines : list
            A lsit of flow information for BG-Flood parameter file
        """
        log.info("Generate flow BG Flood parameter file")
        # Calculate pixel bounds for each injection points
        self.injection_points["bounds"] = self.injection_points.geometry.apply(
            self.pixel_bounds_from_centroid
        )

        # Each line of text represents the flow
        flow_lines = [
            self.flow_text_data_design(row.FID, row.bounds)
            for _, row in self.injection_points.iterrows()
        ]

        return flow_lines

    def bg_param_file_generator(self):
        """Generate BG-Flood param file"""

        if self.vectors is not None:
            terrain_name = str(max(
                Path(self.flood_model_path).glob("8m_geofabric_clipped_elevation_*.nc"),
                default=Path(self.flood_model_path) / "8m_geofabric_clipped_elevation_001.nc"
            ))
        else:
            terrain_name = str(self.flood_model_path / "8m_geofabric_clipped.nc")

        param_text = dedent(f"""\
        topo = {terrain_name}?z;
        dx = 16.0;
        outputtimestep = 7200;
        endtime = {self.seconds};
        mask = 9999;
        gpudevice = 1;
        smallnc = 0;
        zsoffset = 1.500000;
        zsinit = -2.500000;
        frictionmodel = 1;
        cfmap = {terrain_name}?zo;
        outfile = output.nc;
        outvars = hmax;
        top = top_bnd.txt,2;
        bottom = bottom_bnd.txt,2;
        right = right_bnd.txt,2;
        left = left_bnd.txt,2;
        """)

        # rainfile = {str(self.flood_model_path / "precipitation_dynamic.nc")}?depth;

        # Set up output path
        output_path = self.flood_model_path / f"{self.output_folder}/BG_param.txt"

        with open(output_path, "w") as f:
            f.write(param_text)
            for line in self.flow_text_data_generator():
                f.write(line + "\n")

    def parameter_files_generator(self):
        """Generate parameter files to run BG-Flood"""

        # Create output folder
        self.output_folder_generator()

        # Generate flow files
        self.flow_text_file_generator()

        # Generate tide files
        self.tide_text_file_generator()

        # Generate param files
        self.bg_param_file_generator()
