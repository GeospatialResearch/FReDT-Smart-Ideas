# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 22:33:14 2026

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
from shapely.geometry import Point


class TidalDataGenerator:
    """Generate tidal data"""

    def __init__(
            self,
            flood_model_path: Path,
            hydromt_path: Path,
            start_time: datetime,
            end_time: datetime
    ) -> None:
        """
        This class is to generate tidal data

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
        """
        self.flood_model_path = flood_model_path
        self.hydromt_path = hydromt_path
        self.start_time = start_time
        self.end_time = end_time

        # River outlet
        self.river_outlet = gpd.read_file(
            self.flood_model_path / 'river_outlet.shp'
        )
        self.river_outlet_geom = self.river_outlet.geometry.iloc[0]

        self.land = gpd.read_file(
            self.hydromt_path / 'nz_coastline.shp'
        )

        # Merge polygons and edges
        self.land_union = self.land.unary_union  # All polygons are merged into one
        self.land_boundary_union = self.land.boundary.unary_union  # All edges are merged into one

        # Generate nearest coastline point
        self.nearest_point = self.land_boundary_union.interpolate(
            self.land_boundary_union.project(self.river_outlet_geom)
        )
        self.extra_distance = 0  # extra distance from the coastline to the onshore

        # Set up url and headers
        self.url = "https://api.niwa.co.nz/tides/data"
        self.headers = {"x-apikey": "MLnPC2sBzVkbvk6mdEyRxGNcTZDRv8y4"}

    def tidal_point_geom_generator(
            self,
            dir_x: float,
            dir_y: float
    ) -> gpd.GeoDataFrame:
        """
        Generate tidal point geometry

        Parameters
        ----------
        dir_x : float
            X direction
        dir_y : float
            Y direction

        Returns
        -------
        nearshore_tidal_point_gdf : gpd.GeoDataFrame
            Tidal point nearshore GeoDataFrame
        """
        # Calculate current distance to the coastline
        current_distance = math.sqrt(dir_x ** 2 + dir_y ** 2)

        # Extra distance is 0 means that taking river outlet as the tidal point
        if self.extra_distance == 0:
            nearshore_tidal_point = self.river_outlet_geom
        else:
            # Normalise direction unit based on current distance
            # If current distance is 0 - river outlet on the coast line
            # This if is only for when the river outlet is already offshore or on the coastline
            if current_distance == 0:
                unit_dir_x, unit_dir_y = 1, 1

            # If current distance is not 0 - river outlet in the offshore
            else:
                unit_dir_x = dir_x / current_distance
                unit_dir_y = dir_y / current_distance

            # Move nearshore
            nearshore_tidal_point = Point(
                self.nearest_point.x - unit_dir_x * self.extra_distance,
                self.nearest_point.y - unit_dir_y * self.extra_distance
            )

        # Convert to GeoDataframe
        nearshore_tidal_point_gdf = gpd.GeoDataFrame(
            geometry=[nearshore_tidal_point],
            crs=2193
        )

        return nearshore_tidal_point_gdf

    def tidal_point_outland_geom_generator(self) -> gpd.GeoDataFrame:
        """
        Generate tidal point geometry outside land
        where the river outlet is offshore or on the coastline

        Returns
        -------
        nearshore_tidal_point : gpd.GeoDataFrame
            Tidal point nearshore GeoDataFrame
        """
        # Calculate direction from coast to outlet
        dir_x = self.river_outlet_geom.x - self.nearest_point.x
        dir_y = self.river_outlet_geom.y - self.nearest_point.y

        # Generate nearshore tidal point
        nearshore_tidal_point_gdf = self.tidal_point_geom_generator(
            dir_x,
            dir_y
        )

        return nearshore_tidal_point_gdf

    def tidal_point_inland_geom_generator(self) -> gpd.GeoDataFrame:
        """
        Generate tidal point geometry outside land
        where the river outlet is inside land

        Returns
        -------
        nearshore_tidal_point_gdf : gdp.GeoDataFrame
            Tidal point nearshore GeoDataFrame
        """
        # If the river outlet is inside land,
        # the symmetrically opposite-through-coastline point will be used as tidal point.
        # Reflect river outlet across coastline
        reflected_x = 2 * self.nearest_point.x - self.river_outlet_geom.x
        reflected_y = 2 * self.nearest_point.y - self.river_outlet_geom.y

        # Calculate direction from coast to outlet
        dir_x = reflected_x - self.nearest_point.x
        dir_y = reflected_y - self.nearest_point.y

        # Generate nearshore tidal point geodataframe
        nearshore_tidal_point_gdf = self.tidal_point_geom_generator(
            dir_x,
            dir_y
        )

        return nearshore_tidal_point_gdf

    def tidal_point_geom_checker_and_generator(self) -> gpd.GeoDataFrame:
        """
        Check if the river outlet is inside, outside land, or on the coastline
        and then generate tidal point

        Returns
        -------
        nearshore_tidal_point_gdf : gdp.GeoDataFrame
            Tidal point nearshore GeoDataFrame
        """
        # Check if the tidal point is inside or outside land then generate tidal point
        if not self.land_union.contains(self.river_outlet_geom):
            nearshore_tidal_point_gdf = self.tidal_point_outland_geom_generator()
        else:
            nearshore_tidal_point_gdf = self.tidal_point_inland_geom_generator()

        # Write out
        nearshore_tidal_point_gdf.to_file(
            self.flood_model_path / "tidal_point.shp"
        )

        return nearshore_tidal_point_gdf

    def nearshore_tidal_point_crs_conversion(
            self,
            nearshore_tidal_point_gdf: gpd.GeoDataFrame
    ) -> tuple[float, float]:
        """
        Reproject tidal point

        Parameters
        ----------
        nearshore_tidal_point_gdf : gdp.GeoDataFrame
            Tidal point nearshore GeoDataFrame

        Returns
        -------
        tidal_lat : float
            tidal point latitude
        tidal_lon : float
            tidal point longitude
        """
        # Get river outlet x, y
        tidal_point_geom = nearshore_tidal_point_gdf.geometry.iloc[0]

        # Set up crs transformer
        transformer = Transformer.from_crs("EPSG:2193", "EPSG:4326", always_xy=True)

        # Convert x, y coordinates into lon, lat
        tidal_lon, tidal_lat = transformer.transform(
            tidal_point_geom.x,
            tidal_point_geom.y
        )

        return tidal_lat, tidal_lon

    def query_params_designer(
            self,
            nearshore_tidal_point_gdf: gpd.GeoDataFrame
    ) -> dict:
        """
        Design query with parameters to extract tidal data from NIWA API

        Parameters
        ----------
        nearshore_tidal_point_gdf : gdp.GeoDataFrame
            Tidal point nearshore GeoDataFrame

        Returns
        -------
        params_dict : dict
            Dictionary of parameters used for extracting tidal data from NIWA API
        """
        # Get tidal lat, lon
        tidal_lat, tidal_lon = self.nearshore_tidal_point_crs_conversion(
            nearshore_tidal_point_gdf
        )

        # Get number of days
        n_days = (self.end_time - self.start_time).days

        # Design query params
        params_dict = {
            "lat": float(tidal_lat),
            "long": float(tidal_lon),
            "numberOfDays": n_days,
            "startDate": self.start_time.strftime("%Y-%m-%d"),
            "datum": 'MSL',
            "interval": 10
        }

        return params_dict

    def query_params_generator(
            self,
            nearshore_tidal_point_gdf: gpd.GeoDataFrame
    ) -> dict:
        """
        Extract tidal data from NIWA API using the designed query

        Parameters
        ----------
        nearshore_tidal_point_gdf : gdp.GeoDataFrame
            Tidal point nearshore GeoDataFrame

        Returns
        -------
        tidal_dict : dict
            Dictionary of tidal time and values extracted from NIWA API
        """
        # Design parameter dictionary
        param_dict = self.query_params_designer(
            nearshore_tidal_point_gdf
        )

        # Extract tidal data from NIWA API
        tidal_dict = requests.get(
            self.url,
            params=param_dict,
            headers=self.headers
        ).json()

        return tidal_dict

    def resample_tidal_data(
            self,
            tidal_dict: dict
    ):
        """
        Resample tidal data into 1 hour interval

        Parameters
        ----------
        tidal_dict : dict
            Dictionary of tidal time and values extracted from NIWA API

        Returns
        -------
        tidal_df : pd.DataFrame
            DataFrame of tidal data
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
        tidal_df['value'] = tidal_df['value'] - 0.11 + 1

        tidal_df.to_csv(
            self.flood_model_path / 'tidal_point_df_query.csv'
        )

        # Resample tidal data
        tidal_df = tidal_df.resample('1h').mean()

        return tidal_df

    def tidal_data_generator(self):
        """Generate tidal data from NIWA API"""
        # Generate nearshore tidal point
        nearshore_tidal_point_gdf = self.tidal_point_geom_checker_and_generator()

        # Extract tidal data from NIWA API
        tidal_dict = self.query_params_generator(nearshore_tidal_point_gdf)

        # Resample tidal data
        tidal_df = self.resample_tidal_data(tidal_dict)

        # Add more seconds
        tidal_df["seconds"] = (
                tidal_df.index - tidal_df.index[0]
        ).total_seconds()

        # Generate tidal dataframe
        tidal_df.to_csv(
            self.flood_model_path / 'tidal_point_df.csv'
        )

        return tidal_df



# # Check code
# tide_generator = TidalDataGenerator(
#     Path(r"D:\Digital_Twin_data\hydrological_hydrodynamic_riverton_path_001"),
#     Path(r"D:\data_checking_tide"),
#     datetime.fromisoformat("2020-02-03T00:00:00"),
#     datetime.fromisoformat("2020-02-05T00:00:00")
# )
#
# tide_df = tide_generator.tidal_data_generator()