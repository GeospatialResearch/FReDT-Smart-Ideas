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

"""Classes to create input configuration files for Flood Models."""

import logging
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box, mapping, Point, Polygon, LineString, MultiLineString, MultiPoint

from eddie.digitaltwin.utils import setup_logging, LogLevel


setup_logging(LogLevel.DEBUG)
log = logging.getLogger(__name__)


class TerrainGenerator:
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
        log.info("Clipping terrain data to area of interest")
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
        log.info("Writing out terrain data.")
        # Write out
        terrain_crs_clipped.to_netcdf(
            self.flood_model_path / "8m_geofabric_clipped.nc"
        )

    def terrain_data_generator(self) -> None:
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


class InjectionPointsAndStreamlinesAligner:
    """This class is to align injection points with streamlines"""

    def __init__(
        self,
        original_injection_points: gpd.GeoDataFrame,
        dem: xr.DataArray
    ) -> None:
        """
        Align injection points with streamlines.
        This is just a temporary solution for this problem.
        The idea is find the closest and lowest areas to the current injection points.

        Parameters
        ----------
        original_injection_points : gpd.GeoDataFrame
            Original injection points
        dem : xr.DataArray
            DEM used for flood model
        """
        self.original_injection_points = original_injection_points.to_crs("EPSG:2193")
        self.dem = dem

        xmin, ymin, xmax, ymax = self.dem.rio.bounds()
        self.dem_boundary = box(
            xmin, ymin, xmax, ymax
        )

    def clip_dem_around_geometry(
        self,
        dem: xr.DataArray,
        geom: Polygon
    ) -> xr.DataArray:
        """
        Clip DEM around geometry

        Parameters
        ----------
        dem : xr.DataArray
            DEM used for the flood model
        geom : Polygon
            Geometry of injection points used for clipping

        Returns
        -------
        dem_clip : xr.DataArray
            DEM clipped to the geometry
        """
        log.info("Clipping DEM around geometry.")
        # Clip DEM around geometry
        dem_clip = dem.rio.clip(
            [geom],
            crs=2193,
            drop=True
        )

        return dem_clip

    def get_coords_grids_from_dem(
        self,
        dem_clip: xr.DataArray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create coordinate grids from DEM coordinates

        Parameters
        ----------
        dem_clip : xr.DataArray
            DEM clipped to the geometry

        Returns
        -------
        x_array_coord : np.ndarray
            Meshgrid arrays of x coordinates
        y_array_coord : np.ndarray
            Meshgrid arrays of y coordinates
        """
        log.info("Getting coordinate grids from DEM coordinates.")
        # Extract x, y coordinates from DEM that is clipped around geometry
        x_coord = dem_clip.x.values
        y_coord = dem_clip.y.values

        x_array_coord, y_array_coord = np.meshgrid(x_coord, y_coord)

        return x_array_coord, y_array_coord

    def distance_between_cell_and_point(
        self,
        x_array_coord: np.ndarray,
        y_array_coord: np.ndarray,
        point: Point
    ) -> np.ndarray:
        """
        Compute Euclidean distance from each DEM cell to the original injection points

        Parameters
        ----------
        x_array_coord : np.ndarray
            Meshgrid arrays of x coordinates
        y_array_coord : np.ndarray
            Meshgrid arrays of y coordinates
        point : Point
            Original injection point

        Returns
        -------
        distance : np.ndarray
            Distance grid
        """
        log.info("Calculating Euclidean distance from each DEM cell to original injection points.")
        # Calculate the distance
        distance = np.sqrt(
            (x_array_coord - point.x) ** 2 +
            (y_array_coord - point.y) ** 2
        )

        return distance

    def distance_between_cell_and_boundary(
        self,
        x_array_coord: np.ndarray,
        y_array_coord: np.ndarray
    ) -> np.ndarray:
        """
        Compute distance from each DEM cell
        to DEM boundary.

        Parameters
        ----------
        x_array_coord : np.ndarray
            Meshgrid arrays of x coordinates

        y_array_coord : np.ndarray
            Meshgrid arrays of y coordinates

        Returns
        -------
        boundary_distance : np.ndarray
            Distance grid to DEM boundary
        """
        log.info("Calculating Euclidean distance from each DEM cell to DEM boundary")
        boundary_line = self.dem_boundary.boundary

        boundary_distance = np.zeros_like(
            x_array_coord,
            dtype=float
        )

        for row in range(x_array_coord.shape[0]):
            for col in range(x_array_coord.shape[1]):
                point = Point(
                    x_array_coord[row, col],
                    y_array_coord[row, col]
                )

                boundary_distance[row, col] = (
                    point.distance(boundary_line)
                )

        return boundary_distance

    def compute_weighted_score(
        self,
        values: np.ndarray,
        distances: np.ndarray,
        boundary_distances: np.ndarray,
        elevation_weight: float,
        distance_weight: float,
        boundary_weight: float
    ) -> np.ndarray:
        """
        Compute score weighted by elevation and distance.
        (The target is to find the lowest score)

        Parameters
        ----------
        values : np.ndarray
            DEM cell values
        distances : np.ndarray
            Distance grid from original point
        boundary_distances : np.ndarray
            Distance to the boundary grid
        elevation_weight : float
            Weight values that control the influence of elevation.
            Default is 1
        distance_weight : float
            Weight values that control the influence of distance.
            Larger values prefer cells closer to the original injection point.
            Default is 0.01
        boundary_weight : float
            Weight values that manage the distance between the injection points and boundary edges

        Returns
        -------
        np.ndarray
            Computed score grid.
        """
        # Compute score
        log.info("Computing weighted elevation and distance score.")
        score = elevation_weight * values + distance_weight * distances + boundary_weight * boundary_distances

        # Ignore nans
        score[np.isnan(values)] = np.nan

        return score

    def find_best_cell(
        self,
        score: np.ndarray
    ) -> tuple[int, int]:
        """
        Find grid cell with the lowest score

        Parameters
        ----------
        score : np.ndarray
            Combined score grid

        Returns
        -------
        row_column_position : tuple[int, int]
            Row and column indexes
        """
        log.info("Finding best cell")
        # Find the best cell
        best_index = np.nanargmin(score)

        # Find the row nad column position in the grid
        row_column_position = np.unravel_index(best_index, score.shape)
        row, column = row_column_position  # pylint: disable=unbalanced-tuple-unpacking

        return int(row), int(column)

    def convert_index_to_point(
        self,
        row_idx: int,
        column_idx: int,
        dem_clip: xr.DataArray
    ) -> Point:
        """
        Convert grid cell indexes into point

        Parameters
        ----------
        row_idx : int
            Grid xell row index
        column_idx : int
            Grid cell column index
        dem_clip : xr.DataArray
            DEM clipped to the geometry

        Returns
        -------
        new_injection_point : Point
            Injection point at grid cell centre
        """
        # Create new injection point
        new_injection_point = Point(
            dem_clip.x.values[column_idx],
            dem_clip.y.values[row_idx]
        )

        return new_injection_point

    def snap_one_point(
        self,
        point: Point,
        dem: xr.DataArray,
        buffer_distance: float = 100,
        elevation_weight: float = 1,
        distance_weight: float = 0.01,
        boundary_weight: float = 0.02
    ) -> Point:
        """
        Snap a point to a nearby low-elevation DEM grid cell.
        The selected point will have the minimised (elevation + weighted_distance * distance)

        Parameters
        ----------
        point : Point
            Original injection point
        dem : xr.DataArray
            DEM used for the flood model
        buffer_distance : float = 100
            Search radius around the point (metres).
            Default is 100
        elevation_weight : float
            Weight values that control the influence of elevation.
            Default is 1.
        distance_weight : float = 0.05
            Controls balance between lower elevation and closer distance.
            Larger values prefer closer cells.
            Default is 0.01
        boundary_weight : float = 0.02
            Controls balance lower elevation, closer distance, and distance to the boundary edges.
            Default is 0.02

        Returns
        -------
        snapped_point : Point
            Snapped point geometry.
        """
        # Create search area
        buffer_geom = point.buffer(buffer_distance)

        # Clip DEM
        dem_clip = self.clip_dem_around_geometry(
            dem,
            buffer_geom
        )

        # Extract elevation values
        values = dem_clip.values

        # Coordinate grids
        x_array_coord, y_array_coord = self.get_coords_grids_from_dem(dem_clip)

        # Distance grid
        distances = self.distance_between_cell_and_point(
            x_array_coord,
            y_array_coord,
            point
        )

        boundary_distances = self.distance_between_cell_and_boundary(
            x_array_coord,
            y_array_coord
        )

        # Combined score
        score = self.compute_weighted_score(
            values,
            distances,
            boundary_distances,
            elevation_weight,
            distance_weight,
            boundary_weight
        )

        # Best cell
        row_idx, column_idx = self.find_best_cell(score)  # pylint: disable=unbalanced-tuple-unpacking

        # Snapped injection point
        snapped_injection_point = self.convert_index_to_point(
            row_idx,
            column_idx,
            dem_clip
        )

        return snapped_injection_point

    def snap_multiple_injection_points(
        self,
        buffer_distance: float = 100,
        elevation_weight: float = 1,
        distance_weight: float = 0.01,
        boundary_weight: float = 0.02
    ) -> gpd.GeoDataFrame:
        """
        Snap multiple injection points to nearby low-elevation cells

        Parameters
        ----------
        buffer_distance : float = 100
            Search radius around each point (metres).
            Default is 100
        elevation_weight : float
            Weight values that control the influence of elevation.
            Default is 1.
        distance_weight : float = 0.01
            Controls preference for closer cells.
            Default is 0.01
        boundary_weight : float = 0.02
            Controls preference for distance to the edges.
            Default is 0.02

        Returns
        -------
        gpd.GeoDataFrame
            New injection points
        """
        log.info("Snapping injections points to nearby low-elevation cells.")
        # Create empty geometry
        new_geometries = []

        for _, row in self.original_injection_points.iterrows():
            # Snap all original injection points
            snapped_point = self.snap_one_point(
                point=row.geometry,
                dem=self.dem,
                buffer_distance=buffer_distance,
                elevation_weight=elevation_weight,
                distance_weight=distance_weight,
                boundary_weight=boundary_weight
            )

            # Add to new geometries
            new_geometries.append(snapped_point)

        # Create new injection points
        new_injection_points = self.original_injection_points.copy()

        # Assign new point geometries
        new_injection_points.geometry = new_geometries

        return new_injection_points


class InjectionPointsFloodModelGenerator:
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
        log.info("Reprojecting rivers dataframe.")
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
        log.info("Generating intersections between rivers and DEM bounding box.")
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
        log.info("Extracting injection points from intersections.")
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

    def align_injection_points_and_streamlines(
        self,
        original_injection_points: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Align injections points with streamlines

        Parameters
        ----------
        original_injection_points : gpd.GeoDataFrame
            Original injection points

        Returns
        -------
        gpd.GeoDataFrame
            New injection points
        """
        log.info("Aligning injections points with streamlines.")
        # Get DEM
        terrain_data = xr.open_dataset(self.flood_model_path / "8m_geofabric_clipped.nc")
        dem = terrain_data.z.rio.write_crs("EPSG:2193")

        # Set up aligner class
        injection_points_and_streamlines_aligner = InjectionPointsAndStreamlinesAligner(
            original_injection_points,
            dem
        )

        # Align injection points with streamlines by snapping
        new_injection_points = injection_points_and_streamlines_aligner.snap_multiple_injection_points(
            buffer_distance=300,
            elevation_weight=2,
            distance_weight=0.001,
            boundary_weight=0.5
        ).to_crs("EPSG:2193")

        return new_injection_points

    def write_out_injection_points(
        self,
        new_injection_points: gpd.GeoDataFrame
    ) -> None:
        """
        Write out injection points

        Parameters
        ----------
        new_injection_points : gpd.GeoDataFrame
            Points geodataframe of intersections between rivers and DEM
            bounding box with IDs and converted crs
        """
        log.info("Writing out injection points.")
        # Write out to shapefile
        points_path = self.flood_model_path / "injection_points.shp"
        new_injection_points.to_file(points_path)

    def extract_rivers_flow_from_catchment_model(self) -> xr.DataArray:
        """
        Extract rivers' flow data from catchment model outputs

        Returns
        -------
        rivers_flow : xr.DataArray
            Rivers' flow data extracted from catchment model outputs
        """
        log.info("Extracting rivers' flow from catchment model outputs.")
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
        log.info("Extracting rivers' flow from catchment model outputs.")
        injection_points_flow_dict = {}

        # Extract rivers' flow from catchment model outputs
        # for injection points
        for _i, row in points_gdf.iterrows():
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
        log.info("Writing out rivers' flow within time.")
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

    def injection_points_flow_generator(self) -> None:
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
        original_injection_points = self.prepare_injection_points(points)

        # Align original injection points with streamlines
        new_injection_points = self.align_injection_points_and_streamlines(original_injection_points)

        # Write out injection points to shapefile
        self.write_out_injection_points(new_injection_points)

        # Extract rivers' flow from catchment model outputs
        rivers_flow = self.extract_rivers_flow_from_catchment_model()

        # Extract rivers' flow for injection points
        # Here we still use the original injection points to make sure the flow data is extracted smoothly
        injection_points_flow_df = self.extract_rivers_flow_for_injection_points(
            rivers_flow, original_injection_points)

        # Write out injection points' flow within given time
        self.write_out_rivers_flow_within_time(
            injection_points_flow_df,
            rivers_flow
        )
