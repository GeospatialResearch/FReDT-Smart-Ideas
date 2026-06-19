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

"""Classes to create input configuration files for LISFLOOD-FP."""
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import rioxarray as rxr
import xarray as xr
from rasterio.enums import Resampling
from scipy.ndimage import binary_dilation
from shapely.geometry import box, mapping, Point, Polygon, LineString, MultiLineString, MultiPoint

from src.eddie_floodresilience.preprocessing.terrain_data_manipulator import TerrainFilter
from src.eddie_floodresilience.preprocessing.terrain_attributes_generator import TerrainAttributesGenerator


class TerrainGenerator:
    """This class is to generate terrain data for other classes"""

    def __init__(
        self,
        flood_model_path: Path,
        hydromt_path: Path,
        river_name: str,
        aoi_boundary: list,
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
        crs : int = 2193
            Targeted crs. The default is 2193 for NZTM.
        """
        self.flood_model_path = flood_model_path
        self.hydromt_path = hydromt_path
        self.river_name = river_name
        self.aoi_boundary = box(*aoi_boundary)  # using * to unpack xmin, ymin, xmax, ymax
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

        # Write out clipped terrain data
        self.write_out_terrain_data(terrain_crs_clipped)

        return terrain_bounding_box, terrain_crs_clipped


class InjectionPointsandStreamlinesAligner():
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
        # Find the best cell
        best_index = np.nanargmin(score)

        # Find the row nad column position in the grid
        row_column_position = np.unravel_index(best_index, score.shape)

        return row_column_position

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
        row_idx, column_idx = self.find_best_cell(score)

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
        new_injection_points: gpd.GeoDataFrame
            New injection points
        """
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


class TerrainFloodModelGenerator():
    """This class is to generate terrain data (DEM and friction) for flood model"""

    def __init__(
        self,
        flood_model_path: Path,
        hydromt_path: Path,
        river_name: str,
        terrain_crs_clipped: xr.Dataset,
        adjust_manning: bool,
        crs: int = 2193
    ) -> None:
        """
        Generate terrain data includes DEM and friction for flood model

        Parameters
        ----------
        flood_model_path : Path
            Directory to folder storing flood model data
        hydromt_path : Path
            A directory to where all necessary files are stored to run wflow model
        river_name: str
            Name of directory to where the river information files are stored
        terrain_crs_clipped : xr.Dataset
            Clipped terrain data with projection
        adjust_manning : bool
            True means adjusting Manning's n by resampling 4m Manning's n
            False means no Mannning's n adjustment
        crs : int = 2193
            Targeted crs. The default is 2193 for NZTM.
        """
        self.flood_model_path = flood_model_path
        self.hydromt_path = hydromt_path
        self.river_name = river_name
        self.terrain_crs_clipped = terrain_crs_clipped
        self.adjust_manning = adjust_manning
        self.crs = crs

    def fill_nan_and_write_nodata(
        self,
        terrain_variable: xr.Dataset
    ) -> xr.Dataset:
        """
        Fill NaN and write nodata for a specific terrain data

        Parameters
        ----------
        terrain_variable : xr.Dataset
            Specific terrain data that needs filling with -9999 as nodata

        Returns
        -------
        terrain_variable: xr.Dataset
            Specific terrain data that are filled with -9999 as nodata
        """
        # Fill NaN with -9999
        terrain_variable = terrain_variable.fillna(-9999)

        # Write nodata as -9999
        terrain_variable = terrain_variable.rio.write_nodata(-9999)

        return terrain_variable

    def format_terrain_data_pipeline(
        self,
        variable_name: str
    ) -> xr.Dataset:
        """
        Format terrain data to be used by flood model, the steps are:
            - drop spatial reference
            - make sure data have crs
            - fill nan and write nodata
            - round up values

        Parameters
        ----------
        variable_name : str
            Variable name could be 'z' (DEM) and 'zo' (roughness length)

        Returns
        -------
        terrain_variable : xr.Dataset
            Specific terrain data that are formatted
        """
        # Get specific terrain
        terrain_variable = self.terrain_crs_clipped[f'{variable_name}']

        # Drop spatial ref
        terrain_variable = terrain_variable.drop_vars("spatial_ref")

        # Make sure terrain that has crs
        terrain_variable = terrain_variable.rio.write_crs(self.crs)

        # Fill NaN with -9999 and write it as nodata value
        terrain_variable = self.fill_nan_and_write_nodata(terrain_variable)

        # Round up to easily process
        terrain_variable = terrain_variable.round(9)

        return terrain_variable

    def strahler_for_manning_generator(self) -> None:
        """Generate Strahler order streams raster to filter river in Manning's n"""
        # Split terrain data to collect roughness raster
        TerrainFilter(
            path=self.flood_model_path,
            hydromt_path=self.hydromt_path,
            river_name=self.river_name,
            origin_filename='4m_geofabric'
        ).filter_dem_for_wflow()

        # Set up class to generate Strahler order streams
        strahler_for_manning = TerrainAttributesGenerator(
            self.flood_model_path,
            'dem',
            10,
            1000,
            origin_filename='4m_geofabric'
        )

        # Resample resolution (from 4m to 10m)
        # This can be the problem in the future if the terrain is too large,
        # the memory will be broken
        strahler_for_manning.raster_resampling_using_gdal('nn')

        # Fill depression (default is 0.0001)
        strahler_for_manning.raster_fill_depression()

        # Generate D8 pointers
        strahler_for_manning.d8_pointer_generator()

        # Generate streams for whole catchment
        strahler_for_manning.d8_stream_generator()

        # Generate Strahler order streams for manning's n
        strahler_for_manning.strahler_stream_order_generator()

    def resample_roughness(self) -> None:
        """Resample roughness from 4m to 8m"""
        # Get roughness path
        roughness_path = self.flood_model_path / "4m_geofabric_roughness_split.tif"

        # Auto chunk size from native block size
        with rio.open(roughness_path) as src:
            block_y, block_x = src.block_shapes[0] if src.block_shapes else (4096, 4096)

        # Read roughness raster
        roughness = rxr.open_rasterio(
            roughness_path,
            chunks={'x': block_x, 'y': block_y}
        )

        # Convert -9999 to 0.004
        roughness = roughness.where(roughness != -9999, 0.004)

        # Write nodata
        roughness = roughness.rio.write_nodata(-9999)

        # Write crs
        roughness = roughness.rio.write_crs("EPSG:2193")

        # Resample to 8m
        roughness_8m = roughness.rio.reproject(
            2193,
            resolution=8,
            resampling=Resampling.nearest
        )

        # Save roughness 8m out
        roughness_outpath = self.flood_model_path / "roughness_8m.tif"
        roughness_8m.rio.to_raster(
            roughness_outpath,
            tiled=True,
            windowed=True,
            lock=False,
        )

        return roughness_8m

    def strahler_filter_generator(
        self,
        roughness: xr.DataArray
    ) -> None:
        """Generate filtered Strahler Order stream raster"""
        # Get strahler path
        strahler_path = self.flood_model_path / "4m_geofabric_strahler_d8.tif"

        # Read strahler
        strahler = rxr.open_rasterio(strahler_path)

        # Assign crs
        strahler = strahler.rio.write_crs("EPSG:2193")

        # Reproject and rescale strahler to roughness 8m
        strahler_8m = strahler.rio.reproject_match(
            roughness,
            resampling=Resampling.nearest
        )

        # Find max order
        max_order = int(strahler_8m.max().item())

        # Filter to choose only 3 and 4 orders
        strahler_mask_8m = (strahler_8m >= 3) & (strahler_8m <= max_order)

        return strahler_mask_8m

    def roughness_to_manning(
        self,
        roughness: xr.DataArray,
        h: float = 1
    ) -> xr.DataArray:
        """
        Convert raster of roughness length to Manning's n

        Parameters
        ----------
        roughness : xr.DataArray
            Roughness length raster
        h : float = 1
            Value of depth. Default is 1

        Returns
        -------
        manning : xr.DataArray
            Manning's n raster converted from roughness length raster
        """
        # Avoid zero division
        roughness = roughness.clip(min=1e-6)

        # Avoid invalid log inputs
        ratio = h / roughness
        ratio_h_roughness = ratio.where(ratio > 1, 1)

        # Convert roughness length to Manning's n
        numerator = 0.41 * (h ** (1 / 6)) * (ratio_h_roughness - 1)
        denominator = np.sqrt(9.80665) * (1 + ratio_h_roughness * (np.log(ratio_h_roughness) - 1))
        manning_n = numerator / denominator

        # Avoid unreasonable Manning's n
        manning_n = manning_n.clip(min=1e-4, max=0.8)

        return manning_n

    def manning_adjustment(
        self,
        strahler_mask_raster: bool,
        manning: xr.DataArray
    ) -> xr.DataArray:
        """
        Filter out unreasonable Manning's n and adjust river Manning's n

        Parameters
        ----------
        strahler_mask_raster : bool
            Strahler at 8m and being filtered (only 3 and 4 orders)
        manning : xr.DataArray
            Manning's n raster converted from the roughness length raster

        Returns
        -------
        adjusted_manning : xr.DataArray
            Manning's n raster that is filtered out unreasonable values with adjusted river
        """
        # Filter out unreasonable Manning's n
        manning_filtered = manning.clip(
            min=1e-4,
            max=0.8
        )

        # Buffer the mask by 1-pixel wide to 2-pixel wide
        mask = strahler_mask_raster.values.astype(bool)
        buffered_mask = binary_dilation(mask, iterations=1)

        # Convert back to DataArray
        buffered_mask_da = xr.DataArray(
            buffered_mask,
            coords=strahler_mask_raster.coords,
            dims=strahler_mask_raster.dims
        )

        # Adjust river Manning's n.
        # At the moment, the value is kept at 0.06
        manning_river_adjusted = manning_filtered.where(
            ~buffered_mask_da,
            0.08
        )

        return manning_river_adjusted

    def manning_generator(self) -> None:
        """Generate clipped Manning's n from 4m to 8m with adjusted river"""
        if self.adjust_manning:
            # # Generate Strahler Order streams and roughness length at 4m
            # self.strahler_for_manning_generator()

            # Resample roughness to 8m
            roughness_8m = self.resample_roughness()

            # Filter Strahler Order streams
            strahler_mask_8m = self.strahler_filter_generator(roughness_8m)

            # Convert roughness length to Manning's n
            manning = self.roughness_to_manning(
                roughness_8m,
                1
            )

            # Adjust Manning's n
            manning_for_flood = self.manning_adjustment(
                strahler_mask_8m,
                manning
            )

        else:
            # Roughness path
            roughness_path = self.flood_model_path / "8m_geofabric_roughness_split.tif"

            # Read roughness raster
            roughness = rxr.open_rasterio(roughness_path)

            # Convert -9999 to 0.004
            roughness = roughness.where(roughness != -9999, 0.004)

            # Write nodata
            roughness = roughness.rio.write_nodata(-9999)

            # Convert roughness length to Manning's n
            manning_for_flood = self.roughness_to_manning(
                roughness,
                1
            ) * 2

        # Clip Manning's n
        clipped_manning_for_flood = manning_for_flood.rio.clip_box(
            *self.terrain_crs_clipped.rio.bounds()
        )

        # Write out
        manning_outpath = self.flood_model_path / "manning.asc"
        clipped_manning_for_flood.rio.to_raster(manning_outpath)

    def write_out_terrain_data(
        self,
        variable_name: str,
        terrain_variable: xr.Dataset
    ) -> None:
        """
        Write out terrain data as ASCII file

        Parameters
        ----------
        variable_name : str
            Variable name could be 'z' (DEM) and 'zo' (roughness length)
        terrain_variable : xr.DataArray
            Specific terrain data that needs writing out
        """
        # Set up path
        terrain_variable_path = self.flood_model_path / f"{variable_name}.asc"

        # Write out as ASCII file
        terrain_variable.rio.to_raster(terrain_variable_path)

    def terrain_data_generator(
        self,
        variable_name: str
    ) -> None:
        """
        Generate speific terrain data

        Parameters
        ----------
        variable_name : str
            Variable name could be 'z' (DEM) and 'zo' (roughness length)
        """
        # Format terrain data
        terrain_variable = self.format_terrain_data_pipeline(variable_name)

        # Write out terrain data
        self.write_out_terrain_data(
            variable_name,
            terrain_variable
        )

    def execute_terrain_data_generator(self) -> None:
        """Generate specific terrain data"""
        # Generate DEM
        self.terrain_data_generator('z')

        # Generate manning
        self.manning_generator()


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

    def align_injection_points_and_streamlines(
        self,
        original_injection_points: gpd.GeoDataFrame
    ) -> None:
        """
        Align injections points with streamlines

        Parameters
        ----------
        original_injection_points : gpd.GeoDataFrame
            Original injection points
        """
        # Get DEM
        terrain_data = xr.open_dataset(self.flood_model_path / "8m_geofabric_clipped.nc")
        dem = terrain_data.z.rio.write_crs("EPSG:2193")

        # Set up aligner class
        injection_points_and_streamlines_aligner = InjectionPointsandStreamlinesAligner(
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
