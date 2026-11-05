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


"""
Unit tests for DrainageChannelBurner.
"""

import geopandas as gpd
import numpy as np
import pytest
import rioxarray  # noqa: F401 (registers the .rio accessor)
import shapely
import xarray as xr

from drainage_channel import DrainageChannel

CRS = "EPSG:2193"


def make_dem(values: np.ndarray, resolution: float = 1.0, origin=(0.0, 0.0)) -> xr.DataArray:
    """
    Build a small, georeferenced test DEM from a 2D array of elevations.

    Parameters
    ----------
    values : numpy.ndarray
        2D array of elevations, row 0 is the northernmost row.
    resolution : float, optional
        Cell size in both x and y.
    origin : tuple of float, optional
        (x, y) coordinate of the top-left corner of the grid.

    Returns
    -------
    xarray.DataArray
        DEM with `x`/`y` coordinates, a CRS, and an affine transform
        set via rioxarray.
    """
    n_rows, n_cols = values.shape
    x_origin, y_origin = origin
    x = x_origin + resolution * (np.arange(n_cols) + 0.5)
    y = y_origin - resolution * (np.arange(n_rows) + 0.5)

    dem = xr.DataArray(
        values.astype(float),
        coords={"y": y, "x": x},
        dims=("y", "x"),
    )
    dem = dem.rio.write_crs(CRS)
    dem = dem.rio.write_transform()
    return dem


@pytest.fixture
def flat_dem() -> xr.DataArray:
    """5x5 DEM, flat at elevation 10."""
    return make_dem(np.full((5, 5), 10.0))


@pytest.fixture
def ramp_dem() -> xr.DataArray:
    """5x5 DEM, elevation increases by 1 per column (0 .. 4)."""
    values = np.tile(np.arange(5, dtype=float), (5, 1))
    return make_dem(values)


class TestInterpolateLineZ:
    def test_straight_line_midpoint(self):
        line = shapely.LineString([(0, 0), (1, 0), (2, 0)])
        line_3d = DrainageChannel._interpolate_line_z(line, start_z=0.0, end_z=10.0)
        z = np.asarray(line_3d.coords)[:, 2]
        np.testing.assert_allclose(z, [0.0, 5.0, 10.0])

    def test_uneven_spacing_is_distance_weighted(self):
        # Vertex at x=1 is 1/4 of the way along a line of total length 4.
        line = shapely.LineString([(0, 0), (1, 0), (4, 0)])
        line_3d = DrainageChannel._interpolate_line_z(line, start_z=0.0, end_z=8.0)
        z = np.asarray(line_3d.coords)[:, 2]
        np.testing.assert_allclose(z, [0.0, 2.0, 8.0])

    def test_equal_start_end_elevation(self):
        line = shapely.LineString([(0, 0), (3, 4)])
        line_3d = DrainageChannel._interpolate_line_z(line, start_z=5.0, end_z=5.0)
        z = np.asarray(line_3d.coords)[:, 2]
        np.testing.assert_allclose(z, [5.0, 5.0])

    def test_zero_length_line_does_not_raise(self):
        line = shapely.LineString([(1, 1), (1, 1)])
        line_3d = DrainageChannel._interpolate_line_z(line, start_z=2.0, end_z=9.0)
        z = np.asarray(line_3d.coords)[:, 2]
        np.testing.assert_allclose(z, [2.0, 2.0])


class TestGetXyzValuesAtPoints:
    def test_samples_nearest_cell_values(self, ramp_dem):
        points = gpd.GeoSeries(
            [shapely.Point(0.5, -0.5), shapely.Point(4.5, -0.5)], crs=CRS
        )
        xyz = DrainageChannel._get_xyz_values_at_points(ramp_dem, points)
        np.testing.assert_allclose(xyz.z.to_numpy(), [0.0, 4.0])

    def test_preserves_index(self, flat_dem):
        points = gpd.GeoSeries(
            [shapely.Point(0.5, -0.5), shapely.Point(1.5, -1.5)],
            index=[10, 20],
            crs=CRS,
        )
        xyz = DrainageChannel._get_xyz_values_at_points(flat_dem, points)
        assert list(xyz.index) == [10, 20]


class TestGetChannelLinesAs3D:
    def test_end_points_match_dem_elevation(self, ramp_dem):
        # Horizontal line from the low (x=0) to the high (x=4) side.
        line = shapely.LineString([(0.5, -2.5), (4.5, -2.5)])
        channel_lines = gpd.GeoDataFrame(geometry=[line], crs=CRS)
        channel = DrainageChannel(ramp_dem, channel_lines)

        lines_3d = channel._get_channel_lines_as_3d()
        coords = np.asarray(lines_3d.iloc[0].coords)

        assert coords[0, 2] == pytest.approx(0.0)
        assert coords[-1, 2] == pytest.approx(4.0)

    def test_reprojects_to_dem_crs(self, flat_dem):
        line = shapely.LineString([(0.5, -0.5), (3.5, -3.5)])
        channel_lines = gpd.GeoDataFrame(geometry=[line], crs=CRS).to_crs("EPSG:4326")
        channel = DrainageChannel(flat_dem, channel_lines)

        lines_3d = channel._get_channel_lines_as_3d()
        assert lines_3d.crs == flat_dem.rio.crs


class TestGetChannelFeatureDemDepth:
    def test_channel_below_dem_produces_positive_depth(self, flat_dem):
        # DEM is flat at 10, channel bed is at 2: expect 8 removed along it.
        line_3d = shapely.LineString([(0.5, -2.5, 2.0), (4.5, -2.5, 2.0)])
        channel = DrainageChannel(flat_dem, gpd.GeoDataFrame(geometry=[]))

        depth = channel._get_channel_feature_dem_depth(line_3d)

        burned_row = depth.sel(y=-2.5, method="nearest")
        np.testing.assert_allclose(burned_row.values, 8.0)

    def test_channel_above_dem_produces_no_depth(self, flat_dem):
        # Channel bed higher than the DEM should never raise the surface.
        line_3d = shapely.LineString([(0.5, -2.5, 20.0), (4.5, -2.5, 20.0)])
        channel = DrainageChannel(flat_dem, gpd.GeoDataFrame(geometry=[]))

        depth = channel._get_channel_feature_dem_depth(line_3d)

        np.testing.assert_allclose(depth.values, 0.0)

    def test_cells_away_from_channel_are_untouched(self, flat_dem):
        line_3d = shapely.LineString([(0.5, -2.5, 2.0), (4.5, -2.5, 2.0)])
        channel = DrainageChannel(flat_dem, gpd.GeoDataFrame(geometry=[]))

        depth = channel._get_channel_feature_dem_depth(line_3d)

        far_row = depth.sel(y=-0.5, method="nearest")
        np.testing.assert_allclose(far_row.values, 0.0)


class TestBurnChannelsIntoDem:
    def test_single_channel_lowers_dem_to_bed_elevation(self, flat_dem):
        line = shapely.LineString([(0.5, -2.5), (4.5, -2.5)])
        channel_lines = gpd.GeoDataFrame(geometry=[line], crs=CRS)
        channel = DrainageChannel(flat_dem, channel_lines)

        # Force a known, lower bed elevation by sampling a DEM where the
        # start/end points already sit at the desired bed height.
        burned = channel.burn_channels_into_dem()

        burned_row = burned.sel(y=-2.5, method="nearest")
        untouched_row = burned.sel(y=-0.5, method="nearest")

        # Flat DEM + flat channel -> channel bed equals DEM, no change.
        np.testing.assert_allclose(burned_row.values, flat_dem.sel(y=-2.5, method="nearest").values)
        np.testing.assert_allclose(untouched_row.values, 10.0)

    def test_channel_cuts_through_higher_terrain(self, ramp_dem):
        # Vertical line at constant low x, so its sampled start/end
        # elevation (0) is lower than cells further along the ramp.
        line = shapely.LineString([(0.5, -0.5), (0.5, -4.5)])
        channel_lines = gpd.GeoDataFrame(geometry=[line], crs=CRS)
        channel = DrainageChannel(ramp_dem, channel_lines)

        burned = channel.burn_channels_into_dem()

        # Column x=0.5 is unaffected (bed equals existing elevation there).
        np.testing.assert_allclose(burned.sel(x=0.5).values, ramp_dem.sel(x=0.5).values)

    def test_overlapping_channels_take_deepest_incision(self, flat_dem):
        shallow = shapely.LineString([(0.5, -2.5, 8.0), (4.5, -2.5, 8.0)])
        deep = shapely.LineString([(0.5, -2.5, 3.0), (4.5, -2.5, 3.0)])
        channel = DrainageChannel(flat_dem, gpd.GeoDataFrame(geometry=[]))

        depth = xr.apply_ufunc(
            np.fmax,
            channel._get_channel_feature_dem_depth(shallow),
            channel._get_channel_feature_dem_depth(deep),
        )

        burned_row = depth.sel(y=-2.5, method="nearest")
        # Deepest channel (bed=3) should dominate: 10 - 3 = 7.
        np.testing.assert_allclose(burned_row.values, 7.0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
