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

"""Modify LAI data according to the nature-based solution scenario."""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rioxarray  # noqa: F401  # pylint: disable=unused-import  # adds .rio to read staticmaps crs
import rasterio.features
from affine import Affine


def load_staticmaps(staticmaps_path: Path) -> xr.Dataset:
    """
    Read staticmaps into memory and close the file, so it can be overwritten later

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


def rasterize_polygons(ds: xr.Dataset, polygons: gpd.GeoDataFrame) -> np.ndarray:
    """
    Convert polygons into raster format on the staticmaps grid

    Parameters
    ----------
    ds : xr.Dataset
        Staticmaps that define the grid
    polygons : gpd.GeoDataFrame
        Polygons to rasterize

    Returns
    -------
    polygon_mask : np.ndarray
        True for cells inside the polygons
    """
    # Make sure polygons have the same crs as staticmaps
    if ds.rio.crs is not None:
        polygons = polygons.to_crs(ds.rio.crs)

    # Set up grid information from staticmaps
    x = ds["x"].values
    y = ds["y"].values
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


class LAIModification:
    """This class is to modify LAI data according to the nature-based solution scenario."""

    def __init__(
        self,
        staticmaps_original: Path,
        staticmaps_scenario: Path,
        codes_to_override: list[int] | None = None,
        polygon_shp: Path | None = None,
        use_polygon_mask: bool = False,
        changed_cells_only: bool = True,
        lai_csv_name: str = "lai_per_lulc_landcover.csv"
    ) -> None:
        """
        Modify LAI data according to the nature-based solution scenario.
        When land cover is changed (e.g. pasture to wetland), the LAI of the new land cover
        is mixed with the old LAI. This class replaces it with the LAI from the original scenario.

        Parameters
        ----------
        staticmaps_original : Path
            Directory to the staticmaps of the original scenario
        staticmaps_scenario : Path
            Directory to the staticmaps of the new scenario (e.g. wetland, forest)
        codes_to_override : list[int] | None = None
            Land cover codes that need LAI modifying (e.g. [45] for wetland).
            If None, the codes are found automatically by comparing the two staticmaps
        polygon_shp : Path | None = None
            Directory to the scenario polygons. Only needed when use_polygon_mask is True
        use_polygon_mask : bool = False
            True means only modifying cells inside the polygons.
            False means modifying cells across the whole catchment
        changed_cells_only : bool = True
            True means only modifying cells whose land cover was changed.
            False means modifying all cells that have the new land cover
        lai_csv_name : str = "lai_per_lulc_landcover.csv"
            Name of the LAI table (same name in both scenario folders)
        """
        self.staticmaps_original = Path(staticmaps_original)
        self.staticmaps_scenario = Path(staticmaps_scenario)
        self.codes_to_override = codes_to_override
        self.polygon_shp = Path(polygon_shp) if polygon_shp is not None else None
        self.use_polygon_mask = use_polygon_mask
        self.changed_cells_only = changed_cells_only
        self.lai_csv_name = lai_csv_name

    def detect_changed_codes(
        self,
        ds_original: xr.Dataset,
        ds_scenario: xr.Dataset
    ) -> list[int]:
        """
        Find the new land cover codes in the scenario

        Parameters
        ----------
        ds_original : xr.Dataset
            Staticmaps of the original scenario
        ds_scenario : xr.Dataset
            Staticmaps of the new scenario

        Returns
        -------
        codes : list[int]
            Land cover codes that need LAI modifying
        """
        # Use the given codes if available
        if self.codes_to_override is not None:
            codes = self.codes_to_override
            return codes

        # Find cells inside the catchment where land cover was changed
        active_mask = ds_scenario["wflow_subcatch"].values > 0
        changed = active_mask & (ds_scenario["meta_landuse"].values != ds_original["meta_landuse"].values)

        # Get the new land cover codes of those cells
        codes = np.unique(ds_scenario["meta_landuse"].values[changed]).astype(int).tolist()

        return codes

    def modify_lai_table(
        self,
        codes: list[int]
    ) -> tuple[pd.DataFrame, dict]:
        """
        Replace the monthly LAI of the new land cover in the scenario LAI table
        with the monthly LAI from the original scenario

        Parameters
        ----------
        codes : list[int]
            Land cover codes that need LAI modifying

        Returns
        -------
        modified_table : pd.DataFrame
            Modified LAI table
        reference_lai : dict
            Monthly LAI from the original scenario for each land cover code

        Raises
        ------
        ValueError
            If a land cover code does not appear exactly once in both LAI tables
        """
        # Read LAI tables
        original_table = pd.read_csv(self.staticmaps_original.parent / self.lai_csv_name)
        scenario_table = pd.read_csv(self.staticmaps_scenario.parent / self.lai_csv_name)

        # Get month columns
        month_cols = [col for col in original_table.columns if col not in ("landuse", "samples")]

        # Copy scenario table to not be affected by the change
        modified_table = scenario_table.copy()
        reference_lai = {}

        for code in codes:
            original_row = original_table["landuse"] == code
            scenario_row = modified_table["landuse"] == code

            # Make sure each land cover code appears once in both tables
            if original_row.sum() != 1 or scenario_row.sum() != 1:
                raise ValueError(f"Land cover code {code} must appear once in both LAI tables")

            # Replace monthly LAI with the original one
            monthly_lai = original_table.loc[original_row, month_cols].values.flatten().astype(float)
            modified_table.loc[scenario_row, month_cols] = monthly_lai
            reference_lai[code] = monthly_lai

        return modified_table, reference_lai

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
            True for cells inside the polygons.
            If use_polygon_mask is False, all cells are True
        """
        # No polygon mask needed
        if not self.use_polygon_mask:
            polygon_mask = np.ones(ds_scenario["wflow_subcatch"].shape, dtype=bool)
            return polygon_mask

        # Rasterize the scenario polygons onto the staticmaps grid
        polygon_mask = rasterize_polygons(ds_scenario, gpd.read_file(self.polygon_shp))

        return polygon_mask

    def modify_lai_raster(
        self,
        ds_original: xr.Dataset,
        ds_scenario: xr.Dataset,
        codes: list[int],
        reference_lai: dict
    ) -> xr.Dataset:
        """
        Replace the LAI in the scenario staticmaps with the monthly LAI from the original scenario

        Parameters
        ----------
        ds_original : xr.Dataset
            Staticmaps of the original scenario
        ds_scenario : xr.Dataset
            Staticmaps of the new scenario
        codes : list[int]
            Land cover codes that need LAI modifying
        reference_lai : dict
            Monthly LAI from the original scenario for each land cover code

        Returns
        -------
        modified_staticmaps : xr.Dataset
            Staticmaps of the new scenario with modified LAI

        Raises
        ------
        ValueError
            If the number of months in LAI does not match the LAI table
        """
        # Set up masks
        active_mask = ds_scenario["wflow_subcatch"].values > 0
        polygon_mask = self.build_polygon_mask(ds_scenario)
        scenario_landuse = ds_scenario["meta_landuse"].values
        original_landuse = ds_original["meta_landuse"].values

        # Copy scenario staticmaps to not be affected by the change
        modified_staticmaps = ds_scenario.copy()
        lai = modified_staticmaps["LAI"].values.copy()  # (month, y, x)

        for code in codes:
            monthly_lai = reference_lai[code]

            # Make sure LAI has the same number of months
            if lai.shape[0] != len(monthly_lai):
                raise ValueError(f"LAI has {lai.shape[0]} months but the LAI table has {len(monthly_lai)}")

            # Choose cells to modify
            modify_mask = active_mask & polygon_mask & (scenario_landuse == code)
            if self.changed_cells_only:
                modify_mask &= original_landuse != code

            # Replace LAI month by month
            for month in range(lai.shape[0]):
                lai[month][modify_mask] = monthly_lai[month]

        modified_staticmaps["LAI"].values = lai

        return modified_staticmaps

    def write_out_lai_table(
        self,
        modified_table: pd.DataFrame
    ) -> Path:
        """
        Write out modified LAI table by overwriting the scenario LAI table

        Parameters
        ----------
        modified_table : pd.DataFrame
            Modified LAI table

        Returns
        -------
        output_path : Path
            Directory to the modified LAI table
        """
        output_path = self.staticmaps_scenario.parent / self.lai_csv_name
        modified_table.to_csv(output_path, index=False)

        return output_path

    def write_out_staticmaps(
        self,
        modified_staticmaps: xr.Dataset
    ) -> Path:
        """
        Write out modified staticmaps by overwriting the scenario staticmaps

        Parameters
        ----------
        modified_staticmaps : xr.Dataset
            Staticmaps of the new scenario with modified LAI

        Returns
        -------
        output_path : Path
            Directory to the modified staticmaps
        """
        output_path = self.staticmaps_scenario
        modified_staticmaps.to_netcdf(output_path, mode="w")

        return output_path

    def run(self) -> tuple[Path, Path]:
        """
        Run all steps to modify LAI data

        Returns
        -------
        lai_table_path : Path
            Directory to the modified LAI table
        staticmaps_path : Path
            Directory to the modified staticmaps
        """
        # Read staticmaps
        ds_original = load_staticmaps(self.staticmaps_original)
        ds_scenario = load_staticmaps(self.staticmaps_scenario)

        # Find land cover codes that need LAI modifying
        codes = self.detect_changed_codes(ds_original, ds_scenario)

        # Modify LAI table
        modified_table, reference_lai = self.modify_lai_table(codes)

        # Modify LAI in staticmaps
        modified_staticmaps = self.modify_lai_raster(ds_original, ds_scenario, codes, reference_lai)

        # Write out
        lai_table_path = self.write_out_lai_table(modified_table)
        staticmaps_path = self.write_out_staticmaps(modified_staticmaps)

        return lai_table_path, staticmaps_path
