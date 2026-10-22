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

"""Modify kvfrac data according to the nature-based solution scenario."""

from pathlib import Path

from osgeo import gdal
import numpy as np
import xarray as xr
import geopandas as gpd
import rioxarray  # pylint: disable=unused-import  # adds .rio to read staticmaps crs
import rasterio.features
from affine import Affine


class KvfracModification:
    """This class is to modify kvfrac data according to the nature-based solution scenario."""

    def __init__(
        self,
        staticmaps_scenario: Path,
        polygons: gpd.GeoDataFrame,
        scenario_kvfrac: tuple,
        baseline_kvfrac: tuple = (0.03, 1.0, 1.0, 1.0, 1.0, 1.0),
        changed_cells_only: bool = False,
        staticmaps_original: Path | None = None
    ) -> None:
        """
        Modify kvfrac data according to the nature-based solution scenario.
        kvfrac is the factor to multiply the vertical conductivity of each soil layer.
        Cells outside the polygons keep the original kvfrac (from staticmaps_original),
        cells inside the polygons get the scenario kvfrac, cells outside the catchment stay nodata.
        This works for any nature-based solution (e.g. wetland, forest).

        Parameters
        ----------
        staticmaps_scenario : Path
            Directory to the staticmaps of the new scenario (e.g. wetland, forest)
        polygons : gpd.GeoDataFrame
            Polygons of the nature-based solution
        scenario_kvfrac : tuple
            kvfrac for each soil layer inside the polygons, from top to bottom.
            Ex: (0.037, 1.0, 1.0, 1.0, 1.0, 1.0) for wetland, (0.035, 1.0, 1.0, 1.0, 1.0, 1.0) for forest
        baseline_kvfrac : tuple = (0.03, 1.0, 1.0, 1.0, 1.0, 1.0)
            kvfrac for each soil layer outside the polygons, from top to bottom.
            Only used when staticmaps_original is not given
        changed_cells_only : bool = False
            True means only modifying cells inside the polygons whose land cover was changed.
            False means modifying all cells inside the polygons
        staticmaps_original : Path | None = None
            Directory to the staticmaps of the original scenario.
            Its kvfrac is used outside the polygons (recommended).
            Also needed when changed_cells_only is True
        """
        self.staticmaps_scenario = Path(staticmaps_scenario)
        self.polygons = polygons
        self.scenario_kvfrac = scenario_kvfrac
        self.baseline_kvfrac = baseline_kvfrac
        self.changed_cells_only = changed_cells_only
        self.staticmaps_original = Path(staticmaps_original) if staticmaps_original is not None else None

    def load_staticmaps(
        self,
        staticmaps_path: Path
    ) -> xr.Dataset:
        """
        Read staticmaps into memory

        Parameters
        ----------
        staticmaps_path : Path
            Directory to the staticmaps

        Returns
        -------
        staticmaps : xr.Dataset
            Staticmaps
        """
        with xr.open_dataset(staticmaps_path) as ds:
            staticmaps = ds.load()

        return staticmaps

    def build_polygon_mask(
        self,
        ds_scenario: xr.Dataset
    ) -> np.ndarray:
        """
        Convert polygons into raster format on the staticmaps grid

        Parameters
        ----------
        ds_scenario : xr.Dataset
            Staticmaps of the new scenario

        Returns
        -------
        polygon_mask : np.ndarray
            True for cells inside the polygons
        """
        # Make sure polygons have the same crs as staticmaps
        polygons = self.polygons
        if ds_scenario.rio.crs is not None:
            polygons = polygons.to_crs(ds_scenario.rio.crs)

        # Set up grid information from staticmaps
        x = ds_scenario["x"].values
        y = ds_scenario["y"].values
        res_x = x[1] - x[0]
        res_y = y[1] - y[0]
        transform = Affine.translation(x[0] - res_x / 2, y[0] - res_y / 2) * Affine.scale(res_x, res_y)

        # Rasterize all polygons at once
        shapes = [(geom, 1) for geom in polygons.geometry]
        polygon_mask = rasterio.features.rasterize(
            shapes=shapes,
            out_shape=(len(y), len(x)),
            transform=transform,
            fill=0,
            dtype="uint8"
        ).astype(bool)

        return polygon_mask

    def build_modify_mask(
        self,
        ds_scenario: xr.Dataset
    ) -> np.ndarray:
        """
        Choose cells to modify: inside the catchment and inside the polygons
        (and with changed land cover if changed_cells_only is True)

        Parameters
        ----------
        ds_scenario : xr.Dataset
            Staticmaps of the new scenario

        Returns
        -------
        modify_mask : np.ndarray
            True for cells to modify
        """
        # Cells inside the catchment and inside the polygons
        active_mask = ds_scenario["wflow_subcatch"].values > 0
        polygon_mask = self.build_polygon_mask(ds_scenario)
        modify_mask = active_mask & polygon_mask

        # Only keep cells whose land cover was changed
        if self.changed_cells_only:
            ds_original = self.load_staticmaps(self.staticmaps_original)
            changed = ds_scenario["meta_landuse"].values != ds_original["meta_landuse"].values
            modify_mask &= changed

        return modify_mask

    def modify_kvfrac(
        self,
        ds_scenario: xr.Dataset,
        modify_mask: np.ndarray
    ) -> xr.Dataset:
        """
        Build kvfrac: baseline kvfrac everywhere, scenario kvfrac inside the chosen cells.
        It uses the same soil layers as the Brooks-Corey exponent (c)

        Parameters
        ----------
        ds_scenario : xr.Dataset
            Staticmaps of the new scenario
        modify_mask : np.ndarray
            True for cells to modify

        Returns
        -------
        modified_staticmaps : xr.Dataset
            Staticmaps of the new scenario with modified kvfrac
        """
        # Use c as the template for soil layers (layer, y, x)
        soil_layers = ds_scenario["c"]
        n_layers = soil_layers.shape[0]

        # Make sure there is one kvfrac value per soil layer
        if len(self.baseline_kvfrac) != n_layers or len(self.scenario_kvfrac) != n_layers:
            raise ValueError(
                f"Staticmaps has {n_layers} soil layers but baseline_kvfrac has {len(self.baseline_kvfrac)} "
                f"and scenario_kvfrac has {len(self.scenario_kvfrac)}"
            )

        # Start every cell from the original kvfrac (or the baseline values if no original given)
        if self.staticmaps_original is not None:
            ds_original = self.load_staticmaps(self.staticmaps_original)
            kvfrac = ds_original["kvfrac"].values.astype(np.float32).copy()
        else:
            baseline_values = np.array(self.baseline_kvfrac, dtype=np.float32)
            kvfrac = np.ones(soil_layers.shape, dtype=np.float32) * baseline_values[:, None, None]

        # Replace kvfrac inside the chosen cells, layer by layer
        for layer in range(n_layers):
            kvfrac[layer][modify_mask] = self.scenario_kvfrac[layer]

        # Keep nodata outside the catchment
        outside_catchment = ~(ds_scenario["wflow_subcatch"].values > 0)
        kvfrac[:, outside_catchment] = np.nan

        # Write into staticmaps
        modified_staticmaps = ds_scenario.copy()
        modified_staticmaps["kvfrac"] = (soil_layers.dims, kvfrac)

        return modified_staticmaps

    def write_out_staticmaps(
        self,
        modified_staticmaps: xr.Dataset
    ) -> Path:
        """
        Write out modified staticmaps by overwriting the scenario staticmaps

        Parameters
        ----------
        modified_staticmaps : xr.Dataset
            Staticmaps of the new scenario with modified kvfrac

        Returns
        -------
        output_path : Path
            Directory to the modified staticmaps
        """
        output_path = self.staticmaps_scenario
        modified_staticmaps.to_netcdf(output_path, mode="w")

        return output_path

    def run(self) -> Path:
        """
        Run all steps to modify kvfrac data

        Returns
        -------
        staticmaps_path : Path
            Directory to the modified staticmaps
        """
        # Read staticmaps
        ds_scenario = self.load_staticmaps(self.staticmaps_scenario)

        # Choose cells to modify
        modify_mask = self.build_modify_mask(ds_scenario)

        # Modify kvfrac in staticmaps
        modified_staticmaps = self.modify_kvfrac(ds_scenario, modify_mask)

        # Write out
        staticmaps_path = self.write_out_staticmaps(modified_staticmaps)

        return staticmaps_path