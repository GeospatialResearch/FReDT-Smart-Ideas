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
"""This script is for modify drainages"""
from typing import NamedTuple

import pandas as pd
from rasterio.features import rasterize
import xarray as xr
import numpy as np
from shapely.geometry import Point


class GenerateDrainageElevation:
    """This class is for generating elevations for drainage line"""

    def __init__(
        self,
        drainage_line: pd.Series,
        dem: xr.DataArray,
        decrease_elevation_value: float = 0.03
    ) -> None:
        """
        Generate elevation for drainage line

        Parameters
        ----------
        drainage_line : pd.Series
            Drainage line
        dem : xr.DataArray
            DEM
        decrease_elevation_value : float = 0.03
            Value used to decrease the elevation at per resolution node when going downstream.
            Default is 0.03 m.
        """
        self.drainage_geom = drainage_line.geometry
        self.dem = dem
        self.decrease_elevation_value = decrease_elevation_value

    def generate_points_along_drainage_line(self) -> list:
        """
        Generate points along drainage line at resolution nodes

        Returns
        -------
        drainage_points : list
            A list of points along drainage line
        """
        # Get drainage distance interval along the drainage line based on resolution nodes
        # abs() here is for if the value is negative
        drainage_distance_interval = abs(self.dem.rio.resolution()[0])

        # Generate drainage distances along the drainage line based on the intervals
        drainage_length = self.drainage_geom.length  # Get drainage length
        drainage_distances = np.arange(
            0,
            drainage_length,
            drainage_distance_interval
        )

        # Generate points at the distance intervals/resolution nodes
        drainage_points = [
            self.drainage_geom.interpolate(distance) for distance in drainage_distances
        ]

        return drainage_points

    def generate_ordinal_number_of_pixels(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate ordinal number of pixels

        Returns
        -------
        horizontal_nums_arr : np.ndarray
            Horizontal ordinal numbers of drainage line in DEM
        vertical_nums_arr : np.ndarray
            Vertical oridinal numbers of drainage line in DEM
        """
        # Generate points along drainage line based on resolution nodes
        drainage_points = self.generate_points_along_drainage_line()

        # Loop to collect ordinal number of pixels of drainage line within the DEM
        horizontal_nums, vertical_nums = [], []
        for each_point in drainage_points:
            # Convert actual coordinates into ordinal number of pixels.
            # "~" means to inverse the transform
            dem_transform = self.dem.rio.transform()
            vertical_num, horizontal_num = ~dem_transform * (each_point.x, each_point.y)

            # Append to lists of ordinal numbers
            horizontal_nums.append(int(horizontal_num))
            vertical_nums.append(int(vertical_num))

        # Remove duplicate points in the same pixel
        # by using dict.fromkeys
        horizontal_vertical = list(dict.fromkeys(zip(
            horizontal_nums,
            vertical_nums
        )))
        horizontal_nums, vertical_nums = zip(*horizontal_vertical)

        # Convert to numpy array for easier manipulation
        horizontal_nums_arr = np.array(horizontal_nums)
        vertical_nums_arr = np.array(vertical_nums)

        return horizontal_nums_arr, vertical_nums_arr

    @staticmethod
    def check_downstream_upstream(
        horizontal_nums_arr: np.ndarray,
        vertical_nums_arr: np.ndarray,
        drainage_original_elevation: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Check if the line is connected downstream -> upstream. If not, reverse

        Parameters
        ----------
        horizontal_nums_arr : np.ndarray
            Horizontal ordinal numbers of drainage line in DEM
        vertical_nums_arr : np.ndarray
            Vertical oridinal numbers of drainage line in DEM
        drainage_original_elevation: np.ndarray
            Original drainage elevation

        Returns
        -------
        horizontal_nums_arr : np.ndarray
            Horizontal ordinal numbers of drainage line in DEM that are checked
        vertical_nums_arr : np.ndarray
            Vertical oridinal numbers of drainage line in DEM that are checked
        drainage_original_elevation: np.ndarray
            Original drainage elevation that are checked
        """
        # Compare the first and last drainage original elevations.
        # If the first > the last, it's downstream-upstream.
        # If the first < the last, it's upstream-downstream, then reverse.
        if drainage_original_elevation[0] < drainage_original_elevation[-1]:
            horizontal_nums_arr = horizontal_nums_arr[::-1]
            vertical_nums_arr = vertical_nums_arr[::-1]
            drainage_original_elevation = drainage_original_elevation[::-1]

        return horizontal_nums_arr, vertical_nums_arr, drainage_original_elevation

    def generate_minimum_elevation(
        self,
        drainage_original_elevation: np.ndarray
    ) -> np.ndarray:
        """
        Compare between new and original elevations and choose the minimum one

        Parameters
        ----------
        drainage_original_elevation : xr.DataArray
            Original drainage elevation

        Returns
        -------
        drainage_new_elevation : np.ndarray
            New drainage elevation that was compared
        """
        # Generate new empty elevation based on original elevation
        drainage_new_elevation = np.empty_like(drainage_original_elevation)

        # Keep the first elevation
        drainage_new_elevation[0] = drainage_original_elevation[0]

        # Compare and choose the minimum value between new and original elevations.
        for i in range(1, len(drainage_original_elevation)):
            # Here using np.minimum as if using min() it will be flagged not scalar but np.ndarray
            drainage_new_elevation[i] = np.minimum(
                drainage_original_elevation[i],
                # Subtracting a given value when going downhill
                drainage_new_elevation[i - 1] - self.decrease_elevation_value
            )

        return drainage_new_elevation

    def generate_elevation_for_drainage_line(self) -> tuple[np.ndarray, xr.DataArray]:
        """
        Generate new elevations for drainage line

        Returns
        -------
        drainage_new_elevation : np.ndarray
            New drainage elevation along the drainage line
        new_dem : xr.DataArray
            DEM with new drainage elevation
        """
        # Generate ordinal number of pixels
        horizontal_nums_arr, vertical_nums_arr = self.generate_ordinal_number_of_pixels()

        # Get drainage original elevation from DEM
        drainage_original_elevation = self.dem.values[
            horizontal_nums_arr,
            vertical_nums_arr
        ]

        # Check downstream-upstream
        (horizontal_nums_arr,
         vertical_nums_arr,
         drainage_original_elevation) = self.check_downstream_upstream(
            *self.generate_ordinal_number_of_pixels(),
            drainage_original_elevation
        )

        # Generate elevation for drainage line
        drainage_new_elevation = self.generate_minimum_elevation(drainage_original_elevation)

        # Assign new elevation to DEM
        new_dem = self.dem.copy(deep=True)
        new_dem.values[
            horizontal_nums_arr,
            vertical_nums_arr
        ] = drainage_new_elevation

        return drainage_new_elevation, new_dem


class GenerateDrainageGeometry:
    """This class is for generating geometries for drainage line"""  # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        dem: xr.DataArray,
        new_dem: xr.DataArray,
        drainage_line: pd.Series,

        horizontal_nums_arr: np.ndarray,
        vertical_nums_arr: np.ndarray,

        drainage_new_elevation: np.ndarray,

        drainage_base_width: float,
        drainage_surface_width: float,
        drainage_slope: float
    ) -> None:
        """
        Generate geometries for drainage line.
        This includes drainage width, drainage base, and drainage slope

        Parameters
        ----------
        dem : xr.DataArray
            Original DEM that is used to add drainages
        new_dem : xr.DataArray
            New DEM with the added drainage elevations
        drainage_line : pd.Series
            A row of drainage GeoDataFrame
        horizontal_nums_arr : np.ndarray
            Horizontal ordinal numbers of drainage line in DEM
        vertical_nums_arr : np.ndarray
            Vertical oridinal numbers of drainage line in DEM
        drainage_new_elevation : np.ndarray
            New drainage elevation along the drainage line
        drainage_base_width : float
            Value of base width of drainage
        drainage_surface_width : float
            Value of surface width of drainage
        drainage_slope : float
            Value of slope of drainage
        """
        self.dem = dem
        self.new_dem = new_dem
        self.drainage_geom = drainage_line.geometry

        self.horizontal_nums_arr = horizontal_nums_arr
        self.vertical_nums_arr = vertical_nums_arr

        self.drainage_new_elevation = drainage_new_elevation

        self.drainage_base_width = drainage_base_width
        self.drainage_surface_width = drainage_surface_width
        self.drainage_slope = drainage_slope

    def generate_distance_pixel_centre_downstream(self) -> np.ndarray:
        """
        Generate distance between each drainage pixel distance to downstream

        Returns
        -------
        distance_pixel_center_downstream : np.ndarray
            Sorted distances between pixel centers to downstream along the drainage line
        """
        # Generate distance
        distance_pixel_center_downstream_list = []
        for h, v in zip(self.horizontal_nums_arr[::-1], self.vertical_nums_arr[::-1]):
            # Convert ordinal number of pixels back to actual coordinates.
            # Using 0.5 to approach the center of a pixel
            actual_coord_x, actual_coord_y = self.dem.rio.transform() * (
                v + 0.5,
                h + 0.5
            )

            # Generate distance from the pixel center point to downstream
            distance_pixel_center_downstream_point = self.drainage_geom.project(
                Point(actual_coord_x, actual_coord_y)
            )

            # Append to the list
            distance_pixel_center_downstream_list.append(
                distance_pixel_center_downstream_point
            )

        # Convert to array for easier manipulation later
        distance_pixel_center_downstream = np.array(
            distance_pixel_center_downstream_list
        )

        return distance_pixel_center_downstream

    def sort_values_based_on_downstream_distance(
        self,
        distance_pixel_center_downstream: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sort out values based on downstream distance along the drainage line

        Parameters
        ----------
        distance_pixel_center_downstream : np.ndarray
            Sorted distances between pixel centers to downstream along the drainage line

        Returns
        -------
        distance_pixel_center_downstream_sorted : np.ndarray
            Sorted distances between pixel centers to downstream along the drainage line
        drainage_new_elevations_sorted : np.ndarray
            Sorted drainage new elevations
        """
        # Get downstream indices
        downstream_indices = np.argsort(distance_pixel_center_downstream)

        # Sort out distances based on downstream indices (basically reverse)
        distance_pixel_center_downstream_sorted = distance_pixel_center_downstream[downstream_indices]

        # Sort out drainage elevations based on downstream indices
        drainage_new_elevations_sorted = self.drainage_new_elevation[downstream_indices]

        return distance_pixel_center_downstream_sorted, drainage_new_elevations_sorted

    def generate_drainage_width_mask(self) -> xr.DataArray:
        """
        Generate drainage width mask - only two boolean values - True for drainage width and False for not

        Returns
        -------
        drainage_width_mask : xr.DataArray
            Mask of drainage width with True for drainage width and False for not
        """
        # Generate drainage width
        drainage_width = self.drainage_geom.buffer(
            self.drainage_surface_width,
            cap_style=2
        )

        # Generate drainage width mask by rasterising drainage width
        drainage_width_mask = rasterize(
            [(drainage_width, 1)],  # Fill drainage width with 1 in the raster
            out_shape=self.dem.shape,
            transform=self.dem.rio.transform(),
            fill=0,  # If not drainage, then 0
            dtype='uint8'
        ).astype(bool)

        return drainage_width_mask

    def generate_distance_drainage_width(
        self,
        drainage_width_horizontal_index: int,
        drainage_width_vertical_index: int
    ) -> tuple[float, float]:
        """
        Generate distance drainage width to drainage line and downstream

        Parameters
        ----------
        drainage_width_horizontal_index : int
            Horizontal index of drainage width
        drainage_width_vertical_index : int
            Vertical index of drainage width

        Returns
        -------
        distance_drainage_width_to_drainage_line : float
            Distance between drainage width and drainage line
        distance_drainage_width_to_downstream : float
            Distane between drainage width and downstream
        """
        # Convert DEM indices of drainage width points into actual coordinates
        drainage_width_coords = self.dem.rio.transform() * (
            drainage_width_vertical_index + 0.5,
            drainage_width_horizontal_index + 0.5
        )

        # Generate drainage width points based on actual coordinates
        drainage_width_point = Point(drainage_width_coords)

        # Generate distances between points and the drainage line
        distance_drainage_width_to_drainage_line = self.drainage_geom.distance(
            drainage_width_point
        )

        # Generate downstream distances for these drainage width points
        distance_drainage_width_to_downstream = self.drainage_geom.project(
            drainage_width_point
        )

        return distance_drainage_width_to_drainage_line, distance_drainage_width_to_downstream

    @staticmethod
    def generate_drainage_width_elevation(
        distance_pixel_center_downstream_sorted: np.ndarray,
        distance_drainage_width_to_downstream: float,
        drainage_new_elevations_sorted: np.ndarray
    ) -> float:
        """

        Parameters
        ----------
        distance_pixel_center_downstream_sorted : np.ndarray
            Sorted distances between pixel centers to downstream along the drainage line
        distance_drainage_width_to_downstream : float
            Distane between drainage width and downstream
        drainage_new_elevations_sorted : np.ndarray
            New drainage elevation that was sorted

        Returns
        -------
        drainage_width_elevation : float
            Elevation of drainage width
        """
        # Search through the distances between pixel centers and downstream
        # along the drainage line to see where this drainage width is
        drainage_width_index = np.searchsorted(
            distance_pixel_center_downstream_sorted,
            distance_drainage_width_to_downstream
        )

        # In case it is the last point the np.searchsorted could not catch the index
        drainage_width_index = np.clip(
            drainage_width_index,
            0,
            len(drainage_new_elevations_sorted) - 1
        )

        # Get drainage width elevation based on the drainage line elevations
        # float() here is to make sure it is not understood as np.ndarray
        drainage_width_elevation = float(drainage_new_elevations_sorted[drainage_width_index])

        return drainage_width_elevation

    def generate_drainage_slope(
            self,
            distance_drainage_width_to_drainage_line: float
    ) -> float:
        """
        Generate drainage slope

        Parameters
        ----------
        distance_drainage_width_to_drainage_line : float
            Distance between drainage width and drainage line

        Returns
        -------
        slope_beyond_base_width : float
            Drainage slope
        """
        # Get differences between one side of drainage surface width and base width
        diff_surface_vs_base = distance_drainage_width_to_drainage_line - self.drainage_base_width

        # Generate distance beyond the base
        # If < 0, still within the base width, if > 0, beyond the base width
        distance_beyond_base = max(
            0.0,
            diff_surface_vs_base
        )

        # Generate slope for distance beyond base width
        slope_beyond_base_width = self.drainage_slope * distance_beyond_base

        return slope_beyond_base_width

    def generate_drainage_elevation_with_geometry(
        self,
        drainage_width_elevations: float,
        slope_beyond_base_width: float,
        drainage_width_horizontal_index: int,
        drainage_width_vertical_index: int
    ) -> float:
        """
        Generate new drainage elevation with geometry information

        Parameters
        ----------
        drainage_width_elevations: float
            Elevation of drainage width
        slope_beyond_base_width : float
            Drainage slope
        drainage_width_horizontal_index : int
            Horizontal index of drainage width
        drainage_width_vertical_index : int
            Vertical index of drainage width

        Returns
        -------
        drainage_new_elevations_with_geometry : float
            New drainage elevation with geometry information
        """
        # Generate new elevation with geometry
        drainage_new_elevations_with_geometry = drainage_width_elevations + slope_beyond_base_width

        # Compare with current elevations, if the current is already low, choose it,
        # no need for changing
        drainage_new_elevations_with_geometry_minimum = min(
            drainage_new_elevations_with_geometry,
            float(self.dem.values[
                drainage_width_horizontal_index,
                drainage_width_vertical_index
            ])
        )

        return drainage_new_elevations_with_geometry_minimum

    def generate_dem_with_drainage_geometry(self) -> xr.DataArray:
        """
        Generate DEM with full drainage geometries - width, base, and slope.

        Returns
        -------
        new_dem_geometry : xr.DataArray
            New DEM with drainage geometry
        """
        # Generate drainage width mask
        drainage_width_mask = self.generate_drainage_width_mask()

        # Generate distance between pixel center to downstream
        distance_pixel_center_downstream = self.generate_distance_pixel_centre_downstream()

        # Generate sorted distance pixel-downstream and new drainage elevation
        (distance_pixel_center_downstream_sorted,
         drainage_new_elevations_sorted) = self.sort_values_based_on_downstream_distance(
            distance_pixel_center_downstream
        )

        # Copy new DEM for easier manipulation
        new_dem_geometry = self.new_dem.copy(deep=True)

        # Get indices of drainage width within DEM - True for drainage width, False for not
        drainage_width_horizontal, drainage_width_vertical = np.where(drainage_width_mask)

        # Generate drainage geometry through a loop
        for h, v in zip(drainage_width_horizontal, drainage_width_vertical):
            # Generate distances between points and the drainage line and downstream
            (distance_drainage_width_to_drainage_line,
             distance_drainage_width_to_downstream) = self.generate_distance_drainage_width(h, v)

            # Get drainage width elevation
            drainage_width_elevation = self.generate_drainage_width_elevation(
                distance_pixel_center_downstream_sorted,
                distance_drainage_width_to_downstream,
                drainage_new_elevations_sorted
            )

            # Generate slope for distance beyond base width
            slope_beyond_base_width = self.generate_drainage_slope(distance_drainage_width_to_drainage_line)

            # Assign new drainage values with geoemtry information to DEM
            new_dem_geometry.values[h, v] = self.generate_drainage_elevation_with_geometry(
                drainage_width_elevation,
                slope_beyond_base_width,
                h, v
            )

        return new_dem_geometry


class DrainageInfo(NamedTuple):
    """
    Tuple to contain results of generating DEM with drainage elevation.

    Attributes
    ----------
    horizontal_drainage_indices : np.ndarray
        Horizontal indices of drainage width
    vertical_drainage_indices : np.ndarray
        Vertical indices of drainage width
    drainage_elevation : np.ndarray
        Drainage elevation along the drainage line
    new_dem : xr.DataArray
        New DEM with drainage elevation
    """

    horizontal_drainage_indices: np.ndarray
    vertical_drainage_indices: np.ndarray
    drainage_elevation: np.ndarray
    new_dem: xr.DataArray


class GenerateFullDrainage:
    """This class is to generate drainage with elevation and geometry"""

    def __init__(
        self,
        dem: xr.DataArray,
        vector: pd.Series | None = None
    ) -> None:
        """
        Generate drainage with elevation and geometry

        Parameters
        ----------
        dem : xr.DataArray
            DEM that needs modification
        vector : pd.Series | None = None
            Vectors that represent centerline of drainage
        """
        self.dem = dem
        self.vector = vector

    def generate_dem_with_drainage_elevation(self) -> DrainageInfo:
        """
        Generate DEM with drainage elevation

        Returns
        -------
        horizontal_drainage_indices : np.ndarray
            Horizontal indices of drainage width
        vertical_drainage_indices : np.ndarray
            Vertical indices of drainage width
        drainage_elevation : np.ndarray
            Drainage elevation along the drainage line
        new_dem : xr.DataArray
            New DEM with drainage elevation
        """
        # Set up function to generate DEM with drainage
        generate_dem_with_drainage = GenerateDrainageElevation(
            self.vector,
            self.dem
        )

        # Generate new DEM with drainage and drainage elevation
        drainage_elevation, new_dem = generate_dem_with_drainage.generate_elevation_for_drainage_line()

        # Generate indices of drainage
        (horizontal_drainage_indices,
         vertical_drainage_indices) = generate_dem_with_drainage.generate_ordinal_number_of_pixels()

        return DrainageInfo(
            horizontal_drainage_indices,
            vertical_drainage_indices,
            drainage_elevation,
            new_dem)

    def generate_dem_with_drainage_geometry(
        self,
        horizontal_drainage_indices: np.ndarray,
        vertical_drainage_indices: np.ndarray,
        drainage_elevation: np.ndarray,
        new_dem: xr.DataArray
    ) -> xr.DataArray:
        """
        Generate DEM with drainage and geometry info

        Parameters
        ----------
        horizontal_drainage_indices : np.ndarray
            Horizontal indices of drainage width
        vertical_drainage_indices : np.ndarray
            Vertical indices of drainage width
        drainage_elevation : np.ndarray
            Drainage elevation along the drainage line
        new_dem : xr.DataArray
            New DEM with drainage elevation

        Returns
        -------
        dem_drainage_geometry : xr.DataArray
            DEM with drainage geometry information
        """
        # Set up function to modify drainage geometry - surface width, base width, and slope
        generate_dem_with_drainage_geometry = GenerateDrainageGeometry(
            self.dem,
            new_dem,
            self.vector,
            horizontal_drainage_indices,
            vertical_drainage_indices,
            drainage_elevation,
            self.vector['b_width'],
            self.vector['s_width'],
            self.vector['slope']
        )

        # Generate DEM with drainage geometry
        dem_drainage_geometry = generate_dem_with_drainage_geometry.generate_dem_with_drainage_geometry()

        return dem_drainage_geometry

    def generate_full_drainage(self) -> xr.DataArray:
        """
        Generate drainage with full information

        Returns
        -------
        dem_full_drainage : xr.DataArray
            DEM with full drainage information
        """
        # Generate drainage information
        (horizontal_drainage_indices,
         vertical_drainage_indices,
         drainage_elevation,
         new_dem) = self.generate_dem_with_drainage_elevation()

        # Generate DEM with full drainage information
        dem_full_drainage = self.generate_dem_with_drainage_geometry(
            horizontal_drainage_indices,
            vertical_drainage_indices,
            drainage_elevation,
            new_dem
        )

        return dem_full_drainage
