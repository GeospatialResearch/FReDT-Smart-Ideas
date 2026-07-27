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

GLOBCOVER_CLASSES: dict[str, int] = {
    "Dense Deciduous Forest": 50,
    "Evergreen Forest": 40,
    "Deciduous Forest": 60,
    "Needleleaf Forest": 70,
    "Pasture Mosaic": 120,
    "Shrubland": 130,
    "Pasture": 140,
    "Sparse Vegetation": 150,
    "Wetland": 160,
    "Bare Land": 200,
}

LCDB_CLASSES: dict[str, int] = {
    "High Producing Exotic Grassland": 40,
    "Low Producing Grassland": 41,
    "Herbaceous Freshwater Vegetation": 45,
    "Manuka and/or Kanuka": 52,
    "Broadleaved Indigenous Hardwoods": 54,
    "Forest - Harvested": 64,
    "Deciduous Hardwoods": 68,
    "Indigenous Forest": 69,
    "Exotic Forest (needleleaf forest)": 71,
}
