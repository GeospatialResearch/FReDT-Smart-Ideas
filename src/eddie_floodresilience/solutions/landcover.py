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

"""Provides definitions of landcover class value mappings."""
import importlib
from enum import StrEnum

import pandas as pd


class LandcoverClassDataset(StrEnum):
    """Specifies which Land cover dataset is being used"""

    GLOBCOVER = "globcover"
    LCDB = "lcdb"


class LandCoverColorMapping:
    """Reads appropriate color mapping for landcover dataset."""

    _RESOURCES_PATH = importlib.resources.files(f"{__package__}.resources")

    _color_mapping: pd.DataFrame = None

    def __init__(self, landcover_class_dataset: LandcoverClassDataset) -> None:
        """
        Create reader for appropriate landcover class mapping file.

        Parameters
        ----------
        landcover_class_dataset : LandcoverClassDataset
            Name of land cover dataset to be mapped.
        """
        self._landcover_class_dataset = landcover_class_dataset

    @property
    def color_mapping(self) -> pd.DataFrame:
        """
        Property for displaying color mapping.

        Returns
        -------
        pd.DataFrame
            Copy of lazy initialized color mapping.
        """
        if self._color_mapping is None:
            mapping_path = self._RESOURCES_PATH / f"{self._landcover_class_dataset}_color_mapping.csv"
            self._color_mapping = pd.read_csv(mapping_path)
        return self._color_mapping.copy()

    @property
    def filtered_color_mapping(self) -> pd.DataFrame:
        """
        Property for displaying color mapping filtered to those marked as displayed.

        Returns
        pd.DataFrame
            A lazy-initialised color mapping dataframe filtered based on "displayed" column.
        """
        return self.color_mapping.loc[self.color_mapping.displayed].copy()
