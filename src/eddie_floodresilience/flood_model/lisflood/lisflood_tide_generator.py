# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 22:33:14 2026

@author: mng42
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 08:51:54 2026

@author: mng42
"""

from datetime import date, timedelta
import pandas as pd
from pyproj import Transformer
import geopandas as gpd
from datetime import datetime
from pathlib import Path
import requests
import math
from shapely.geometry import LineString, Point, box
import numpy as np


class TidalDataGenerator:
    """Generate tidal data"""

    def __init__(
            self,
            flood_model_path: Path,
            hydromt_path: Path,
            flood_aoi_boundary: list,
            start_time: str,
            end_time: str
    ) -> None:
        """
        This class is to generate tidal data

        Parameters
        ----------
        flood_model_path : Path
            Directory to folder storing terrain data
        hydromt_path : Path
            Directory to folder storing necessary data
        flood_aoi_boundary: list
            Boundaries' coordinates of area of interest.
            Format is [xmin, ymin, xmax, ymax]
        start_time : str
            Starting time of simulation
        end_time : str
            Ending time of simulation
        """
        self.flood_model_path = flood_model_path
        self.hydromt_path = hydromt_path
        self.flood_aoi_boundary = flood_aoi_boundary
        self.start_time = start_time
        self.end_time = end_time

        # Get land polygon
        self.land = gpd.read_file(self.hydromt_path / 'nz_coastline.shp')

        # Extra distance
        self.extra_distance = 200
        self.spacing = 500

        # Set up url and headers
        self.url = "https://api.niwa.co.nz/tides/data"
        self.headers = {"x-apikey": "MLnPC2sBzVkbvk6mdEyRxGNcTZDRv8y4"}

    def land_bufferer(self) -> gpd.GeoDataFrame:
        """
        Buffer land polygon as the distance between tide and coastline

        Returns
        -------
        buffered_whole_land_boundary : gpd.GeoDataFrame
            Boundary of land after being buffered

        """
        # Merge all land polygons into one
        whole_land = self.land.dissolve().geometry.iloc[0]

        # Buffer by extra distance
        buffered_whole_land = whole_land.buffer(self.extra_distance)

        # Boundary
        buffered_whole_land_boundary = buffered_whole_land.boundary

        return buffered_whole_land_boundary

    def intersected_coastline_generator(
            self,
            buffered_whole_land_boundary: gpd.GeoDataFrame
    ) -> LineString:
        """
        Intersect DEM with buffered coastline

        Parameters
        ----------
        buffered_whole_land_boundary : gpd.GeoDataFrame
            Boundary of land after being buffered

        Returns
        -------
        intersected_coastline : LineString
            Coastline that is clipped within DEM boundary

        """
        # Set DEM boundary
        self.dem_boundary = box(
            self.flood_aoi_boundary[0],
            self.flood_aoi_boundary[1],
            self.flood_aoi_boundary[2],
            self.flood_aoi_boundary[3]
        )

        # Clip coastline using DEM boundary
        clipped_dem_coastline = buffered_whole_land_boundary.intersection(self.dem_boundary)

        # Conver to GeoDataFrame
        clipped_dem_coastline_gdf = gpd.GeoDataFrame(
            geometry=[clipped_dem_coastline],
            crs=2193
        )

        # Extract intersected line only
        intersected_coastline = clipped_dem_coastline_gdf.geometry.iloc[0]

        return intersected_coastline

    def linestring_to_points_processor(
            self,
            intersected_coastline: LineString,
            spacing: float = 50
    ) -> list:
        """
        Process to convert linestrings to points

        Parameters
        ----------
        intersected_coastline : LineString
            Coastline that is clipped within DEM boundary

        Returns
        -------
        points_list : list
            List of points that are converted
        """
        # Set up points list
        points_list = []

        # Loop through line by line to convert to points
        for each_line in intersected_coastline:

            # Get length of each line
            length = each_line.length

            # Skip lines that are too short
            if length <= 2 * spacing:
                continue

            # Get distances between points except the start/end points
            distances = np.arange(
                spacing,
                length - spacing + 1e-9,
                spacing
            )

            # Create points excluding start/end points
            points_list.extend(
                [each_line.interpolate(distance) for distance in distances]
            )

        return points_list

    def linestring_to_points_generator(
            self,
            intersected_coastline: LineString
    ) -> gpd.GeoDataFrame:
        """
        Generate points from linestrings

        Parameters
        ----------
        intersected_coastline : LineString
            Coastline that is clipped within DEM boundary

        Returns
        -------
        points_gpd : gpd.GeoDataFrame
            GeoDataFrame of points that are converted
        """
        # Make LineString and MultiLineString behave the same
        intersected_coastlines = (
            [intersected_coastline]
            if isinstance(intersected_coastline, LineString)
            else intersected_coastline.geoms
        )

        # Create points based on intersected coastline
        points_list = self.linestring_to_points_processor(
            intersected_coastlines,
            spacing=self.spacing
        )

        # Convert to GeoDataFrame
        points_gdf = gpd.GeoDataFrame(
            geometry=points_list,
            crs=2193
        )

        # Reproject points from 2193 to 4326
        points_gdf = points_gdf.to_crs(4326)

        return points_gdf

    def tidal_points_query_requester(
            self,
            points_gdf: gpd.GeoDataFrame
    ):
        """
        Request tidal points from NIWA API

        Parameters
        ----------
        points_gpd : gpd.GeoDataFrame
            GeoDataFrame of points that are converted

        Returns
        -------
        all_tidal_df : list
            List of all tidal dataframe
        """
        # Set up list of all tidal df
        all_tidal_df_list = []

        for i, geom in enumerate(points_gdf.geometry):
            # Query parameters design
            query_params = {
                "lat": geom.y,
                "long": geom.x,
                "numberOfDays": (self.end_time - self.start_time).days,
                "startDate": self.start_time.strftime("%Y-%m-%d"),
                "datum": "MSL",
                "interval": 10
            }

            # Request points
            points_request = requests.get(
                self.url,
                params=query_params,
                headers=self.headers
            )

            # Get tidal raw data from request
            tidal_raw_data = points_request.json()

            # Get tidal series data
            tidal_series_data = tidal_raw_data.get(
                "values",
                None
            )

            # Convert to dataframe with only time and value
            tidal_df = pd.DataFrame(tidal_series_data)[['time', 'value']]

            # Rename tidal columns
            tidal_df = tidal_df.rename(columns={'value': f"p{i}"})

            # Collect all tidal df
            all_tidal_df_list.append(tidal_df)

        return all_tidal_df_list

    def tidal_points_cleaner(
            self,
            all_tidal_df_list: list
    ) -> pd.DataFrame:
        """
        Clean the requested tidal point dataframes

        Parameters
        ----------
        all_tidal_df_list : list
            List of all tidal dataframe

        Returns
        -------
        all_tidal_df : pd.DataFrame
            All tidal dataframe
        """
        # Mege all tidal point dataframes
        all_tidal_df = pd.concat(all_tidal_df_list, axis=0)

        # Group by time
        all_tidal_df = all_tidal_df.groupby("time", as_index=False).mean()

        # Sort out by time
        all_tidal_df = all_tidal_df.sort_values("time")

        return all_tidal_df

    def tidal_data_generator(self):
        """
        Generate tidal data from NIWA API


        Returns
        -------
        all_tidal_df : pd.DataFrame
            All tidal dataframe
        """
        # Buffer land boundary to intersect with DEM boundary
        buffered_whole_land_boundary = self.land_bufferer()

        # Generate intersected coastline
        intersected_coastline = self.intersected_coastline_generator(
            buffered_whole_land_boundary
        )

        # Convert intersected coastline to points
        points_gdf = self.linestring_to_points_generator(
            intersected_coastline
        )

        # Request tidal values for tidal points from NIWA API
        all_tidal_df_list = self.tidal_points_query_requester(
            points_gdf
        )

        # Clean the requested data
        all_tidal_df = self.tidal_points_cleaner(
            all_tidal_df_list
        )

        return all_tidal_df


# Check

# tide_generator = TidalDataGenerator(
#     Path(r"D:\Digital_Twin_data\hydrological_hydrodynamic_riverton_path_001"),
#     Path(r"D:\data_checking_tide"),
#     [1209556, 4849980, 1222804, 4864908],
#     datetime.fromisoformat("2020-02-03T00:00:00"),
#     datetime.fromisoformat("2020-02-05T00:00:00")
# )
#
# tide_df = tide_generator.tidal_data_generator()
