# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 22:33:14 2026

@author: mng42
"""
from osgeo import gdal  # Import gdal before rasterio
import rioxarray as rxr
import xarray as xr
from rasterio.enums import Resampling
from scipy.ndimage import binary_dilation

import geopandas as gpd
import numpy as np
from shapely.geometry import box
from shapely.geometry import mapping, Polygon, LineString, MultiLineString

from pathlib import Path
from shapely.geometry import Point, MultiPoint

import pandas as pd
from datetime import datetime

from src.eddie_floodresilience.preprocessing.terrain_data_manipulator import TerrainFilter
from src.eddie_floodresilience.preprocessing.terrain_attributes_generator import TerrainAttributesGenerator


class TerrainGenerator():
    """This class is to generate terrain data for other classes"""

    def __init__(
            self,
            flood_model_path: Path,
            hydromt_path: Path,
            river_name: str,
            aoi_boundary: list,
            polygons: str = None,
            vectors: str = None,
            crs: int = 2193
    ) -> None:
        """
        Generate terrain data with its bounding box for other classes

        Parameters
        ----------
        flood_model_path : Path
            Directory to folder storing terrain data
        hydromt_path : Path
            A directory to where all necessary files are stored to run wflow model
        river_name: str
            Name of directory to where the river information files are stored
        aoi_boundary : list
            Boundaries' coordinates of area of interest.
            Format is [xmin, ymin, xmax, ymax]
        polygons : str = None
            Name of polygon file that is used to change the landcover information.
            This polygon dataframe has 'landcover' column with new values
        vectors : str = None
            Name of vector file that is used to change the elevation information.
            This vector dataframe has 'value' column to specify increasing or decreasing elevation,
            and 'distance' column to specify how smooth to decrease elevation.
        crs : int = 2193
            Targeted crs. The default is 2193 for NZTM.
        """
        self.flood_model_path = flood_model_path
        self.hydromt_path = hydromt_path
        self.river_name = river_name
        self.aoi_boundary = box(*aoi_boundary)  # using * to unpack xmin, ymin, xmax, ymax
        self.polygons = polygons
        self.vectors = vectors
        self.crs = crs

    def read_terrain_data(self) -> xr.Dataset:
        """
        Read terrain data. This terrain data includes:
            - z as DEM
            - zo as roughness length

        Returns
        -------
        terrain_crs : xr.Dataset
            Terrain data with crs
        """
        # The name of terrain here is for easily building up automation
        # It will be coded in the future
        terrain_path = self.hydromt_path / f"river_data/{self.river_name}/8m_geofabric.nc"

        # Read terrain raster
        terrain = xr.open_dataset(terrain_path)

        # Make sure terrain has crs
        terrain_crs = terrain.rio.write_crs(self.crs)

        return terrain_crs

    def clip_terrain_data(
            self,
            terrain_crs: xr.Dataset
    ) -> xr.Dataset:
        """
        Clip terrain data to the area of interest

        Parameters
        ----------
        terrain_crs : xr.Dataset
            Terrain data with crs

        Returns
        -------
        terrain_crs_clipped : xr.Dataset
            Clipped terrain data with crs
        """
        # Clip DEM
        terrain_crs_clipped = terrain_crs.rio.clip(
            [mapping(self.aoi_boundary)]
        )

        return terrain_crs_clipped

    def extract_terrain_bounding_box(
            self,
            terrain_crs_clipped: xr.Dataset
    ) -> Polygon:
        """
        Extract terrain data's bounding box

        Parameters
        ----------
        terrain_crs_clipped : xr.Dataset
            Clipped terrain data with crs

        Returns
        -------
        terrain_bounding_box : Polygon
            Bounding box of terrain data
        """
        # Extract terrain bounding box
        terrain_bounding_box = box(
            *terrain_crs_clipped.rio.bounds()
        )

        return terrain_bounding_box

    def write_out_terrain_data(
            self,
            terrain_crs_clipped: xr.Dataset
    ) -> None:
        """
        Write out clipped terrain data

        Parameters
        ----------
        terrain_crs_clipped : xr.Dataset
            Clipped terrain data with crs
        """
        # Write out
        terrain_crs_clipped.to_netcdf(
            self.flood_model_path / "8m_geofabric_clipped.nc"
        )

    def terrain_data_generator(self):
        """
        Generate and write out terrain data after formatting

        Returns
        -------
        terrain_bounding_box : Polygon
            Bounding's box of terrain data
        terrain_crs_clipped : xr.Dataset
            Clipped terrain data with crs
        """
        # Read terrain data with crs
        terrain_crs = self.read_terrain_data()

        # Clip terrain data using bounding box
        terrain_crs_clipped = self.clip_terrain_data(terrain_crs)

        # Extract terrain bounding box
        terrain_bounding_box = self.extract_terrain_bounding_box(terrain_crs_clipped)

        if self.polygons is None and self.vectors is None:
            # Write out clipped terrain data
            self.write_out_terrain_data(terrain_crs_clipped)

        return terrain_bounding_box, terrain_crs_clipped


class InjectionPointsFloodModelGenerator():
    """This class is to generate injection points for flood model"""

    def __init__(
            self,
            flood_model_path: Path,
            catchment_model_path: Path,
            terrain_bounding_box: Polygon,
            start_time: datetime,
            end_time: datetime,
            polygons: str = None,
            crs: int = 2193
    ) -> None:
        """
        Generate injection points for LISFLOOD-FP flood model

        Parameters
        ----------
        flood_model_path : Path
            Directory to folder storing flood model data
        catchment_model_path : Path
            Directory to folder storing catchment model data
        terrain_bounding_box : Polygon
            Bounding's box of terrain data
        start_time : datetime
            Starting time details. Format is "yyyy-mm-ddThh:mm:ss"
        end_time : datetime
            Ending time details. Format is Dataframe that contains rivers' flow data at injection points
        polygons : str = None
            Name of polygon file that is used to change the landcover information.
            This polygon dataframe has 'landcover' column with new values
        crs : int = 2193
            Targeted crs. The default is 2193 for NZTM.
        """
        self.flood_model_path = flood_model_path
        self.catchment_model_path = catchment_model_path
        self.terrain_bounding_box = terrain_bounding_box
        self.start_time = start_time
        self.end_time = end_time
        self.polygons = polygons
        self.crs = crs

    def reproject_rivers(self) -> gpd.GeoDataFrame:
        """
        Re-project river to the same crs as DEM (default is 2193)

        Returns
        -------
        rivers_new_projection : gpd.GeoDataFrame
            River geodataframe with new crs (2193 for NZTM)
        """
        if self.polygons is not None:
            wflow_test_full_folder = str(max(
                Path(self.catchment_model_path).glob("wflow_test_full_landcover_*"),
                default=Path(self.catchment_model_path) / "wflow_test_full_landcover_001"
            ))
        else:
            wflow_test_full_folder = "wflow_test_full"

        # Get river path from wflow model folder
        river_path = self.catchment_model_path / wflow_test_full_folder / r"staticgeoms/rivers.geojson"

        # Read river file
        rivers = gpd.read_file(river_path)

        # Reproject river
        rivers_new_projection = rivers.to_crs(self.crs)

        return rivers_new_projection

    def intersect_rivers_and_dem_bounding_box(
            self,
            rivers_new_projection: gpd.GeoDataFrame
    ) -> gpd.GeoSeries:
        """
        Generate intersection between rivers and DEM bounding box

        Parameters
        ----------
        rivers_new_projection : gpd.GeoDataFrame
            River geodataframe with new crs (2193 for NZTM)

        Returns
        -------
        intersections : gpd.GeoSeries
            Intersections series between rivers and DEM bounding box
        """
        # Get boundary of bounding box
        boundary = self.terrain_bounding_box.boundary

        # Filter out the invalid river
        rivers_new_projection = rivers_new_projection[
            rivers_new_projection['idx'] != rivers_new_projection['idx_ds']
            ]

        # Generate intersection between rivers and DEM bounding box
        intersections = rivers_new_projection.geometry.intersection(boundary)

        return intersections

    def extract_injection_points_from_intersections(
            self,
            intersections: gpd.GeoSeries
    ) -> list[Point]:
        """
        Filter intersections and extract injection points

        Parameters
        ----------
        intersections : gpd.GeoSeries
            Intersections series between rivers and DEM bounding box

        Returns
        -------
        points : list[Point]
            List of points where rivers cut the DEM bounding box
        """
        # Create list of points where rivers intersect DEM bounding box
        points = []

        for geom in intersections:

            # Skip rivers that do not intersect the boundary
            if geom.is_empty:
                continue

            # Collect intersected points where rivers cut terrain boundary
            # If intersected points were found as Point type
            if isinstance(geom, Point):
                points.append(geom)

            # If intersected points were found as MultiPoint type
            elif isinstance(geom, MultiPoint):
                points.extend(list(geom.geoms))

            # If intersected points were found on LineString
            elif isinstance(geom, LineString):
                coords = list(geom.coords)
                if len(coords) >= 2:
                    points.append(Point(coords[0]))
                    points.append(Point(coords[-1]))

            # If intersected points were found as MultiLineString
            elif isinstance(geom, MultiLineString):
                for line in geom.geoms:
                    coords = list(line.coords)
                    if len(coords) >= 2:
                        points.append(Point(coords[0]))
                        points.append(Point(coords[-1]))

        return points

    def prepare_injection_points(
            self,
            points: list[Point]
    ) -> gpd.GeoDataFrame:
        """
        Prepare injection points by adding IDs and converting crs

        Parameters
        ----------
        points : list[Point]
            List of points where rivers cut the DEM bounding box

        Returns
        -------
        points_gdf : gpd.GeoDataFrame
            Points geodataframe of intersections between rivers and DEM
            bounding box with IDs and converted crs
        """
        # Build up geodataframe for points
        points_gdf = gpd.GeoDataFrame(
            geometry=points,
            crs=self.crs
        )

        # Add ID
        points_gdf['FID'] = [
            f"Q_{i}" for i in range(
                1, points_gdf.shape[0] + 1
            )
        ]

        return points_gdf

    def write_out_injection_points(
            self,
            points_gdf: gpd.GeoDataFrame
    ) -> None:
        """
        Write out injection points

        Parameters
        ----------
        points_gdf : gpd.GeoDataFrame
            Points geodataframe of intersections between rivers and DEM
            bounding box with IDs and converted crs
        """
        # Write out to shapefile
        points_path = self.flood_model_path / "injection_points.shp"
        points_gdf.to_file(points_path)

    def extract_rivers_flow_from_catchment_model(self) -> xr.DataArray:
        """
        Extract rivers' flow data from catchment model outputs

        Returns
        -------
        rivers_flow : xr.DataArray
            Rivers' flow data extracted from catchment model outputs
        """
        if self.polygons is not None:
            wflow_test_full_folder = str(max(
                Path(self.catchment_model_path).glob("wflow_test_full_landcover_*"),
                default=Path(self.catchment_model_path) / "wflow_test_full_landcover_001"
            ))
        else:
            wflow_test_full_folder = "wflow_test_full"

        # Set path to rivers' data
        rivers_data_path = self.catchment_model_path / wflow_test_full_folder / r"run_default/output.nc"

        # Read rives' data from catchment model output
        rivers_data = xr.open_dataset(rivers_data_path)
        rivers_flow = rivers_data['q_river']

        return rivers_flow

    def extract_rivers_flow_for_injection_points(
            self,
            rivers_flow: xr.DataArray,
            points_gdf: gpd.GeoDataFrame
    ) -> pd.DataFrame:
        """
        Extract rivers' flow from catchment model outputs at injection points

        Parameters
        ----------
        rivers_flow : xr.DataArray
            Rivers' flow data extracted from catchment model outputs
        points_gdf : gpd.GeoDataFrame
            Points geodataframe of intersections between rivers and DEM
            bounding box with IDs

        Returns
        -------
        injection_points_flow_df : pd.DataFrame
            A table that contains rivers' flow data at injection points
        """
        # Set up dictionary for rivers' flow data for all injection points
        injection_points_flow_dict = {}

        # Extract rivers' flow from catchment model outputs
        # for injection points
        for i, row in points_gdf.iterrows():
            # Get longitude (or x), latitude (or y), and ID
            lon_x = row.geometry.x
            lat_y = row.geometry.y
            name = row['FID']

            # Extract rivers' flow data at injection points
            injection_points_flow = rivers_flow.sel(
                x=lon_x,
                y=lat_y,
                method='nearest'
            )

            # Append to the dictionary
            injection_points_flow_dict[name] = injection_points_flow.values

        # Convert to dataframe
        injection_points_flow_df = pd.DataFrame(injection_points_flow_dict)

        return injection_points_flow_df

    def write_out_rivers_flow_within_time(
            self,
            injection_points_flow_df: pd.DataFrame,
            rivers_flow: xr.DataArray
    ) -> None:
        """
        Write out rivers' flow within given time

        Parameters
        ----------
        injection_points_flow_df : pd.DataFrame
            A table that contains rivers' flow data at injection points
        rivers_flow : xr.DataArray
            Rivers' flow data extracted from catchment model outputs
        """
        # Set up correct time values
        times = pd.to_datetime(rivers_flow['time'].values)

        # Add time to dataframe of injection points' river flows
        injection_points_flow_df.insert(
            0, "time",
            times
        )

        # Extract rivers' flow within given time
        injection_points_flow_df = injection_points_flow_df[
            (injection_points_flow_df['time'] >= self.start_time) &
            (injection_points_flow_df['time'] <= self.end_time)
            ]

        # Write out csv file
        injection_points_flow_path = self.flood_model_path / "injection_points_flow.csv"
        injection_points_flow_df.to_csv(
            injection_points_flow_path,
            index=False
        )

    def injection_points_flow_generator(self):
        """
        Generate injection points and their flow data from catchment model output.
        These injection points will be used in flood model (LISFLOOD-FP)
        """
        # Generate rivers with new projection
        rivers_new_projection = self.reproject_rivers()
        # Remove the extra drainage point where idx == idx_dst and linestring geometry does not exist
        rivers_new_projection = rivers_new_projection[rivers_new_projection["idx"] != rivers_new_projection["idx_ds"]]
        # Generate intersections between rivers and DEM bounding box
        intersections = self.intersect_rivers_and_dem_bounding_box(rivers_new_projection)

        # Extract points where rivers cut DEM bounding box
        points = self.extract_injection_points_from_intersections(intersections)

        # Prepare injection points
        points_gdf = self.prepare_injection_points(points)

        # Write out injection points to shapefile
        self.write_out_injection_points(points_gdf)

        # Extract rivers' flow from catchment model outputs
        rivers_flow = self.extract_rivers_flow_from_catchment_model()

        # Extract rivers' flow for injection points
        injection_points_flow_df = self.extract_rivers_flow_for_injection_points(
            rivers_flow, points_gdf)

        # Write out injection points' flow within given time
        self.write_out_rivers_flow_within_time(
            injection_points_flow_df,
            rivers_flow
        )