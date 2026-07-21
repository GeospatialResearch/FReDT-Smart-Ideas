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

"""Classes to create tide files for LISFLOOD-FP."""

import logging
import math
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point, Polygon

from src.eddie_floodresilience import config

log = logging.getLogger(__name__)


# pylint: disable=too-many-instance-attributes
class TidalDataGenerator:
    """Generate tidal data"""

    def __init__(
            self,
            flood_model_path: Path,
            hydromt_path: Path,
            start_time: datetime,
            end_time: datetime,
            terrain_bounding_box: Polygon
    ) -> None:
        """
        Generate tidal data

        Parameters
        ----------
        flood_model_path : Path
            Directory to folder storing terrain data
        hydromt_path : Path
            Directory to folder storing necessary data
        start_time : str
            Starting time of simulation
        end_time : str
            Ending time of simulation
        terrain_bounding_box : Polygon
            Bounding's box of terrain data
        """
        self.flood_model_path = flood_model_path
        self.hydromt_path = hydromt_path
        self.start_time = start_time
        self.end_time = end_time
        self.terrain_bounding_box = terrain_bounding_box

        # River outlet
        self.river_outlet = gpd.read_file(
            self.flood_model_path.parents[1] / 'terrain/river_outlet.shp'
        )
        self.river_outlet_geom = self.river_outlet.geometry.iloc[0]

        self.land = gpd.read_file(
            self.hydromt_path / 'nz_coastline.shp'
        )

        # Merge all land edges into one
        self.land_boundary_union = self.land.boundary.unary_union  # All edges are merged into one

        # Generate nearest coastline point
        self.nearest_point = self.land_boundary_union.interpolate(
            self.land_boundary_union.project(self.river_outlet_geom)
        )

    def design_tidal_point_from_inland_river_outlet(self) -> Point:
        """
        Generate tidal point from inland river outlet.
        If the river outlet is inland, make it as tidal point.
        If it is on the coastline, nudge 5m further inland, and make it as tidal point.

        Returns
        -------
        tidal_point_from_inland_river_outlet : Point
            Tidal point generated from inland river outlet
        """
        # Calculate the current distance between the nearest point and river outlet
        dir_x = self.river_outlet_geom.x - self.nearest_point.x
        dir_y = self.river_outlet_geom.y - self.nearest_point.y
        current_distance = math.sqrt(dir_x ** 2 + dir_y ** 2)

        if current_distance != 0:
            # If the river outlet is inland, make it as tidal point
            tidal_point_from_inland_river_outlet = self.river_outlet_geom

        else:
            # If the river outlet is on the coastline, nudge 5m further inland
            # and make it tidal point
            # Scale x and y direction into 1 for easy modififcation
            unit_x = dir_x / current_distance
            unit_y = dir_y / current_distance

            # Write out tidal point and nudge 5m further inland at the same time
            tidal_point_from_inland_river_outlet = Point(
                self.river_outlet_geom.x + unit_x * 5,
                self.river_outlet_geom.y + unit_y * 5
            )

        return tidal_point_from_inland_river_outlet

    def design_tidal_point_from_offshore_river_outlet(self) -> Point:
        """
        Generate tidal point from offshore river outlet.
        If the river outlet is offshore, choose the symmetrically opposite inland point as tidal point

        Returns
        -------
        tidal_point_from_offshore_river_outlet : Point
            Tidal point generated from offshore river outlet
        """
        # Generate tidal point generated from offshore river outlet
        # by choosing the symmetrically opposite inland point
        tidal_point_from_offshore_river_outlet = Point(
            2 * self.nearest_point.x - self.river_outlet_geom.x,
            2 * self.nearest_point.y - self.river_outlet_geom.y
        )

        return tidal_point_from_offshore_river_outlet

    def generate_tidal_point_from_river_outlet(self) -> gpd.GeoDataFrame:
        """
        Check if the river outlet is inland or offshore then generate the tidal point

        Returns
        -------
        tidal_point_gdf : gdp.GeoDataFrame
            Tidal point generated from river outlet
        """
        # Merge all land polygons into one
        land_union = self.land.unary_union

        # Check if the river outlet is inland or offshore
        if land_union.contains(self.river_outlet_geom):
            # If the river outlet is inland
            tidal_point = self.design_tidal_point_from_inland_river_outlet()
        else:
            # If the river outlet is offshore
            tidal_point = self.design_tidal_point_from_offshore_river_outlet()

        # Convert tidal point into GeoDataFrame
        tidal_point_gdf = gpd.GeoDataFrame(
            geometry=[tidal_point],
            crs=2193
        )

        # Write out
        tidal_point_gdf.to_file(
            self.flood_model_path / "tidal_point.shp"
        )

        return tidal_point_gdf

    def design_query_params(
            self,
            tidal_point_gdf: gpd.GeoDataFrame
    ) -> dict:
        """
        Design query with parameters to extract tidal data from NIWA API

        Parameters
        ----------
        tidal_point_gdf : gdp.GeoDataFrame
            Tidal point generated from river outlet

        Returns
        -------
        params_dict : dict
            Dictionary of parameters used for extracting tidal data from NIWA API
        """
        # Convert tidal point from crs 2193 to 4326
        tidal_point_gdf_4326 = tidal_point_gdf.to_crs(4326)
        tidal_point_4326 = tidal_point_gdf_4326.geometry.iloc[0]

        # Get time
        number_of_days = math.ceil((self.end_time - self.start_time).total_seconds() / 86400)
        start_date = self.start_time.strftime("%Y-%m-%d")

        # Design query params
        params_dict = {
            "apikey": config.EnvVariable.NIWA_API_KEY,
            "lat": float(tidal_point_4326.y),
            "long": float(tidal_point_4326.x),
            "numberOfDays": number_of_days,
            "startDate": start_date,
            "datum": 'MSL',
            "interval": 10
        }

        return params_dict

    def generate_query_params(
            self,
            tidal_point_gdf: gpd.GeoDataFrame
    ) -> dict:
        """
        Extract tidal data from NIWA API using the designed query

        Parameters
        ----------
        tidal_point_gdf : gdp.GeoDataFrame
            Tidal point generated from river outlet

        Returns
        -------
        tidal_dict : dict
            Dictionary of tidal time and values extracted from NIWA API
        """
        # Design parameter dictionary
        param_dict = self.design_query_params(
            tidal_point_gdf
        )

        # Extract tidal data from NIWA API
        tidal_dict = requests.get(
            "https://api.niwa.co.nz/tides/data",
            params=param_dict
        ).json()

        return tidal_dict

    def resample_tidal_data(
            self,
            tidal_dict: dict
    ) -> pd.DataFrame:
        """
        Resample tidal data into 1 hour interval

        Parameters
        ----------
        tidal_dict : dict
            Dictionary of tidal time and values extracted from NIWA API

        Returns
        -------
        tidal_df : pd.DataFrame
            Data frame of tidal data
        """
        # Set up tidal dataframe
        tidal_df = pd.DataFrame(tidal_dict['values'])

        # Convert time into time format
        tidal_df['time'] = pd.to_datetime(
            tidal_df['time'],
            utc=True
        )

        # Convert UTC to NZ local time
        tidal_df['time'] = tidal_df['time'].dt.tz_convert(
            'Pacific/Auckland'
        )
        tidal_df = tidal_df.set_index('time')

        # Convert datum to NZVD2016
        tidal_df['value'] = tidal_df['value'] - 0.11

        tidal_df.to_csv(
            self.flood_model_path / 'tidal_point_df_query.csv'
        )

        # Resample tidal data
        tidal_df = tidal_df.resample('1h').mean()

        return tidal_df

    def check_tide_existence(self) -> bool:
        """
        Check if the flood aoi boundary includes coastal or is fully inland.
        If includes coastal, the tide exists (True), if not, there is no tide (False)

        Returns
        -------
        bool
            True if coastal and tide exists, False if not and there is no tide
        """
        # Check if flood aoi boundary intersects with coastline, True, else, False
        if self.terrain_bounding_box.intersects(self.land_boundary_union):
            log.info(
                "There is coastline within the area of interest. "
                "Tide data will be generated."
            )
            return True

        log.info(
            "The area of interest is inland. "
            "No tide data will be generated."
        )
        return False

    def generate_tidal_data(self) -> pd.DataFrame | None:
        """
        Generate tidal data from NIWA API

        Returns
        -------
        pd.DataFrame or None
            If there is coastline within the area of interest, then tide_df
            If not, then None
        """
        # There is coastline within the area of interest
        if self.check_tide_existence():
            # Generate tidal point from river outlet
            tidal_point_gdf = self.generate_tidal_point_from_river_outlet()

            # Extract tidal data from NIWA API
            tidal_dict = self.generate_query_params(tidal_point_gdf)

            # Resample tidal data
            tidal_df = self.resample_tidal_data(tidal_dict)

            # Filter the time
            tidal_df = tidal_df[
                (tidal_df.index.tz_localize(None) >= self.start_time) &
                (tidal_df.index.tz_localize(None) <= self.end_time)
            ]

            # Add more seconds
            tidal_df["seconds"] = (
                tidal_df.index - tidal_df.index[0]
            ).total_seconds()

            # Generate tidal dataframe
            tidal_df.to_csv(
                self.flood_model_path / 'tidal_point_df.csv'
            )

            return tidal_df

        # The area of interest is inland
        else:
            return None
