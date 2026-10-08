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

"""Generate groundwater parameters"""

import numpy as np
import xarray as xr
from pathlib import Path

class GroundwaterParameterGenerator:
    """This class is to generate groundwater parameters."""  # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        path,
        input_name: str,
        output_name: str,
        conductivity_divisor: float =50,
        riverbed_thickness_scaled: float =60,
        b_exponent: float =0.5,
        min_depth: float =1,
        max_depth: float =2.5,
        gwf_f_multiplier: float =110
    ):
        """
        Generate groundwater parameters for hydrological modelling

        Parameters
        ----------
        path : Path
            Directory to the wflow model folder that stores the staticmaps file
        input_name : str = "staticmaps.nc"
            Name of the original staticmaps file built by hydromt_wflow
        output_name : str = "staticmaps_gwf.nc"
            Name of the new staticmaps file with groundwater parameters added
        conductivity_divisor : float = 50.0
            Value used to scale horizontal conductivity from KsatVer.
            conductivity = KsatVer / conductivity_divisor
        riverbed_thickness_scaled : float = 60.0
            Riverbed thickness (m) used to calculate infiltration/exfiltration conductance.
            This is a numerical scaling parameter, NOT a physical thickness
        b_exponent : float = 0.5
            Exponent of the downstream hydraulic geometry (Leopold & Maddock, 1953).
            width = a * upstream_area ^ b_exponent
        min_depth : float = 1.0
            River depth (m) for the smallest headwater river
        max_depth : float = 2.5
            River depth (m) for the largest river in the catchment
        gwf_f_multiplier : float = 110
            Value used to scale the conductivity scale parameter for groundwater flow.
            gwf_f = f * gwf_f_multiplier
        """
        # Input and output
        self.input_path = Path(path) / input_name
        self.output_path = Path(path) / output_name

        self.conductivity_divisor = conductivity_divisor
        self.riverbed_thickness_scaled = riverbed_thickness_scaled
        self.b_exponent = b_exponent                                # Leopold & Maddock exponent
        self.min_depth = min_depth                                  # m, smallest headwater river
        self.max_depth = max_depth                                  # m, largest river
        self.gwf_f_multiplier = gwf_f_multiplier                    # gwf_f = f * this

    def load_staticmaps(self) -> xr.Dataset:
        """
        Read the original staticmaps

        Returns
        -------
        ds : xr.Dataset
            Original staticmaps
        """
        ds = xr.open_dataset(self.input_path)
        return ds

    def add_groundwater_parameters(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Add conductivity, specific_yield, smooth river width,
        infiltration and exfiltration conductance

        Parameters
        ----------
        ds : xr.Dataset
            Staticmaps

        Returns
        -------
        ds : xr.Dataset
            Staticmaps with groundwater parameters added
        """
        dims = ds["land_elevation"].dims
        river = ds["wflow_river"].values == 1  # wflow_river is 1/NaN

        # Conductivity and specific yield
        conductivity = ds["KsatVer"].values / self.conductivity_divisor
        specific_yield = ds["thetaS"].values - ds["thetaR"].values

        # Smooth river width, anchored to the real outlet width
        uparea = ds["meta_upstream_area"].values
        real_width = ds["wflow_riverwidth"].values
        outlet_idx = np.nanargmax(np.where(river, uparea, np.nan))
        row, col = np.unravel_index(outlet_idx, uparea.shape)
        a_coef = real_width[row, col] / (uparea[row, col] ** self.b_exponent)
        width_smooth = np.where(river, a_coef * uparea ** self.b_exponent, np.nan)

        # Infiltration / exfiltration conductance (uses smooth width)
        geom_term = (width_smooth * ds["wflow_riverlength"].values) / self.riverbed_thickness_scaled
        infilt_cond = np.where(river, conductivity * geom_term, np.nan)

        # Write into staticmaps
        ds["conductivity"] = (dims, conductivity)
        ds["specific_yield"] = (dims, specific_yield)
        ds["infilt_cond"] = (dims, infilt_cond)
        ds["exfilt_cond"] = (dims, infilt_cond.copy())
        ds["wflow_riverwidth"] = (dims, width_smooth)  # original width overwritten
        return ds

    def fix_river_depth(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Build RiverDepth from rank of upstream area and create river_bottom

        Parameters
        ----------
        ds : xr.Dataset
            Staticmaps

        Returns
        -------
        ds : xr.Dataset
            Staticmaps with new RiverDepth and river_bottom
        """
        dims = ds["land_elevation"].dims
        river = ds["wflow_river"].values == 1
        uparea = ds["meta_upstream_area"].values.astype(float)
        dem = ds["land_elevation"].values.astype(float)

        # Rank of upstream area at river cells: 0 (smallest) .. 1 (largest)
        up_riv = uparea[river]
        ranks = np.empty(len(up_riv))
        ranks[np.argsort(up_riv)] = np.arange(len(up_riv))
        frac = ranks / (len(ranks) - 1)

        # Remap rank into [min_depth, max_depth]
        depth = ds["RiverDepth"].values.astype(float).copy()
        depth[river] = self.min_depth + (self.max_depth - self.min_depth) * frac

        # River bottom = DEM - depth (river cells only)
        river_bottom = np.where(river, dem - depth, np.nan)

        # Write into staticmaps
        ds["RiverDepth"] = (dims, depth.astype(np.float32), ds["RiverDepth"].attrs)
        ds["river_bottom"] = (dims, river_bottom.astype(np.float32),
                              {"unit": "m", "long_name": "river bottom elevation"})
        return ds

    def scale_groundwater_parameter(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Scale groundwater conductivity scale parameter (gwf_f = f * gwf_f_multiplier)

        Parameters
        ----------
        ds : xr.Dataset
            Staticmaps

        Returns
        -------
        ds : xr.Dataset
            Staticmaps with scaled gwf_f
        """
        ds["gwf_f"] = ds["f"] * self.gwf_f_multiplier
        return ds

    def save_staticmaps(self, ds: xr.Dataset) -> Path:
        """
        Write out new staticmaps

        Parameters
        ----------
        ds : xr.Dataset
            Final staticmaps

        Returns
        -------
        Path
            Directory to the new staticmaps
        """
        ds.to_netcdf(self.output_path, mode="w")
        return self.output_path

    def run(self) -> Path:
        """
        Run all steps

        Returns
        -------
        Path
            Directory to the new staticmaps
        """
        ds = self.load_staticmaps()
        ds = self.add_groundwater_parameters(ds)
        ds = self.fix_river_depth(ds)
        ds = self.scale_groundwater_parameter(ds)
        return self.save_staticmaps(ds)