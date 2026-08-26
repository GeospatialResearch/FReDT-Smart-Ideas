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

import geopandas as gpd
import numpy as np
import rasterio.features
import shapely
import xarray as xr


class DrainageChannel:
    """
    Burn drainage channel polylines into a digital elevation model (DEM).

    Each channel line is converted to a 3D line whose start and end elevations are sampled from the DEM,
    with intermediate vertices linearly interpolated by distance along the line.
    The resulting channel bed is then rasterized and used to lower the DEM wherever a channel passes through a cell at
    a higher elevation than its bed.

    Attributes
    ----------
    dem : xarray.DataArray
        Digital elevation model to burn channels into.
    channel_lines : geopandas.GeoDataFrame
        Drainage channels represented as LineString geometries.
    """

    def __init__(self, dem: xr.DataArray, channel_lines: gpd.GeoDataFrame) -> None:
        """
        Parameters
        ----------
        dem : xarray.DataArray
            Digital elevation model to burn channels into.
        channel_lines : geopandas.GeoDataFrame
            Drainage channels represented as LineString geometries. Only
            the start and end point elevations of each line are sampled
            from the DEM; intermediate vertices are interpolated.
        """
        self.dem = dem
        self.channel_lines = channel_lines

    def burn_channels_into_dem(self) -> xr.DataArray:
        """Lower the DEM so each channel follows its interpolated bed profile.

        Returns
        -------
        xarray.DataArray
            A copy of ``self.dem`` where every cell a channel line passes through has been lowered (if necessary) to
            match the interpolated channel bed elevation.
            Cells not touched by any channel are left unchanged.
        """
        return self.dem - self._compute_channel_depth_raster()

    def _compute_channel_depth_raster(self) -> xr.DataArray:
        """Build a raster of elevation to subtract from the DEM for all channels.

        A depth raster (`dem - channel_bed`, clipped to be non-negative, zero where a channel does not cover a cell) is
        computed for every channel feature.
        The per-feature rasters are combined with an element-wise maximum so that, where channels overlap,
        the deepest incision wins.

        Returns
        -------
        xarray.DataArray
            Raster, on the same grid as `self.dem`, of elevation values to subtract from the DEM.
            Zero where no channel is present.
        """
        channel_lines_3d = self._get_channel_lines_as_3d()

        depth = xr.zeros_like(self.dem, dtype=float)
        for geometry in channel_lines_3d.geometry:
            feature_depth = self._get_channel_feature_dem_depth(geometry)
            depth = xr.apply_ufunc(np.fmax, depth, feature_depth)

        return depth

    def _get_channel_feature_dem_depth(
        self, line_3d: shapely.LineString
    ) -> xr.DataArray:
        """
        Rasterize a single 3D channel line into a channel depth DEM.

        The line is split into its individual segments, each rasterized using the mean Z value of its two endpoints.
        The resulting channel bed raster is then compared against the DEM to work out how much elevation must be removed
        from each covered cell.

        Parameters
        ----------
        line_3d : shapely.LineString
            A single line with a Z value at every vertex, describing the channel bed elevation along its length.

        Returns
        -------
        xarray.DataArray
            Raster, on the same grid as `self.dem`, of elevation to subtract from the DEM for this feature.
            Zero where the feature does not cover a cell.
        """
        coords = np.asarray(line_3d.coords)
        segments = [
            (
                shapely.LineString(coords[i: i + 2, :2]),
                float(coords[i: i + 2, 2].mean()),
            )
            for i in range(len(coords) - 1)
        ]

        channel_elevation = rasterio.features.rasterize(
            segments,
            out_shape=self.dem.shape,
            transform=self.dem.rio.transform(),
            fill=np.nan,
            all_touched=True,
            dtype="float64",
        )
        channel_elevation_da = xr.DataArray(
            channel_elevation, coords=self.dem.coords, dims=self.dem.dims
        )

        depth = (self.dem - channel_elevation_da).clip(min=0)
        return depth.fillna(0)

    @staticmethod
    def _get_xyz_values_at_points(dem: xr.DataArray, gs: gpd.GeoSeries) -> gpd.GeoSeries:
        """Sample DEM elevation at a set of 2D points.

        Parameters
        ----------
        dem : xarray.DataArray
            DEM to sample, with ``x``/``y`` coordinates.
        gs : geopandas.GeoSeries
            Points (2D) at which to sample the DEM.

        Returns
        -------
        geopandas.GeoSeries
            3D points, in the same order and with the same index as
            ``gs``, with ``z`` set to the nearest DEM value at each
            point's x/y location.
        """
        x_coords = xr.DataArray(gs.x.to_numpy(), dims="points")
        y_coords = xr.DataArray(gs.y.to_numpy(), dims="points")
        z_coords = dem.sel(x=x_coords, y=y_coords, method="nearest")

        xyz_gs = gpd.GeoSeries.from_xy(
            x_coords.values,
            y_coords.values,
            z=z_coords.values,
            index=gs.index,
            crs=dem.rio.crs,
        )
        return xyz_gs

    def _get_channel_lines_as_3d(self) -> gpd.GeoSeries:
        """Build 3D versions of the channel lines with interpolated bed elevations.

        Each line's start and end elevations are sampled from the DEM.
        Every intermediate vertex is then assigned a Z value by linearly
        interpolating between the start and end elevation, proportional
        to the vertex's distance along the line.

        Returns
        -------
        geopandas.GeoSeries
            3D LineString geometries, one per input channel, with a Z
            value at every vertex describing the channel bed profile.
        """
        channels = self.channel_lines.copy(deep=True)
        channels = channels.to_crs(self.dem.rio.crs)

        start_points = gpd.GeoSeries(
            shapely.get_point(channels.geometry, 0), index=channels.index, crs=channels.crs
        )
        end_points = gpd.GeoSeries(
            shapely.get_point(channels.geometry, -1), index=channels.index, crs=channels.crs
        )
        start_z = self._get_xyz_values_at_points(self.dem, start_points).z.to_numpy()
        end_z = self._get_xyz_values_at_points(self.dem, end_points).z.to_numpy()

        lines_3d = [
            self._interpolate_line_z(geometry, start, end)
            for geometry, start, end in zip(channels.geometry, start_z, end_z)
        ]

        return gpd.GeoSeries(lines_3d, index=channels.index, crs=channels.crs)

    @staticmethod
    def _interpolate_line_z(
        line: shapely.LineString, start_z: float, end_z: float
    ) -> shapely.LineString:
        """Linearly interpolate Z values along a line's vertices.

        Parameters
        ----------
        line : shapely.LineString
            2D line to convert to 3D.
        start_z : float
            Elevation at the line's first vertex.
        end_z : float
            Elevation at the line's last vertex.

        Returns
        -------
        shapely.LineString
            3D line with the same x/y vertices as ``line``, and Z values
            interpolated linearly by distance along the line between
            ``start_z`` and ``end_z``.
        """
        coords = np.asarray(line.coords)
        segment_lengths = np.hypot(*np.diff(coords, axis=0).T)
        cumulative_distance = np.insert(np.cumsum(segment_lengths), 0, 0.0)
        length = cumulative_distance[-1]

        if length > 0:
            fraction = cumulative_distance / length
        else:
            fraction = np.zeros_like(cumulative_distance)
        z = start_z + fraction * (end_z - start_z)

        return shapely.LineString(np.column_stack([coords, z]))
