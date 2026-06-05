# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:00:00 2026

@author: mng42
"""

import yaml
from pathlib import Path
import logging
from eddie.digitaltwin.utils import setup_logging, LogLevel
setup_logging(LogLevel.DEBUG)
log = logging.getLogger(__name__)

class DataCatalogGenerator():
    """This class is to generate data_catalog.yml for preprocessing data for wflow"""
    
    def __init__(
            self,
            hydromt_path: Path,
            wflow_model_path: Path,
            forcing_path: Path,
            river_name: str,
            polygons: str = None,
            landcover: str = 'globcover'
        ) -> None:
        """
        Generate data_catalog.yml for preprocessing data for wflow.
        This data_catalog.yml matches with information (mostly parameter information)
        from wflow_build.yml
        
        Parameters
        ----------
        hydromt_path: Path
            A directory to where all necessary files are stored to run wflow model
        wflow_model_path: Path
            A directory to where the data_catalog.yml is stored and to run wflow model
        forcing_path: Path
            A directory to where the forcing files are stored
        river_name: str
            Name of directory to where the river information files are stored
        polygons : str = None
            Name of polygon file that is used to change the landcover information.
            This polygon dataframe has 'landcover' column with new values
        landcover : str = 'globcover'
            Name of land cover. Default is 'globcover'
        """
        self.hydromt_path = hydromt_path
        self.wflow_model_path = wflow_model_path
        self.forcing_path = forcing_path
        self.river_name = river_name
        self.polygons = polygons
        self.landcover = landcover
        
    def meta_section(self) -> dict:
        """
        Write out meta section
        
        Returns
        -------
        meta : dict
            A dictionary contains information of meta section
        """
        # Generate a dictionary with meta section information
        meta = {
            "meta": {
                "root": str(self.hydromt_path)
            }    
        }
        
        return meta

    def forcing_section(self) -> dict:
        """
        Write out forcing section

        Returns
        -------
        forcing : dict
            A dictionary contains information of forcing section
        """
        # At the moment the name era5 and other information is kept fixed for basic automation,
        # it will be changed in the future.
        # NOTE: the information is not from ERA5
        # Generate a dictionary with forcing information
        forcing = {
            "era5_hourly": {
                "crs": 4326,
                "data_type": "RasterDataset",
                "driver": "netcdf",
                "meta": {
                    "category": "meteo",
                    "history": "Extracted from Copernicus Climate Data Store",
                    "paper_doi": "10.1002/qj.3803",
                    "paper_ref": "Hersbach et al. (2019)",
                    "source_license": "https://cds.climate.copernicus.eu/cdsapp/#!/terms/licence-to-use-copernicus-products",
                    "source_url": "https://doi.org/10.24381/cds.bd0915c6"
                },
                "path": str(self.forcing_path / "era5_hourly_*.nc"),
                "unit_add": {
                    "temp": -273.15
                },
                "unit_mult": {
                    "pet": 3600,
                    "precip": 1,
                    "press_msl": 0.01
                }
            }
        }

        return forcing

    def orography_section(self) -> dict:
        """
        Write out orography section

        Returns
        -------
        orography : dict
            A dictionary contains information of orography section

        """
        # At the moment the name era5 and other information is kept fixed for basic automation,
        # it will be changed in the future.
        # NOTE: the information is not from ERA5
        # Generate a dictionary with orography information
        orography = {
            "era5_orography": {
                "crs": 4326,
                "data_type": "RasterDataset",
                "driver": "netcdf",
                "meta": {
                    "category": "meteo",
                    "history": "Extracted from Copernicus Climate Data Store",
                    "paper_doi": "10.1002/qj.3803",
                    "paper_ref": "Hersbach et al. (2019)",
                    "source_license": "https://cds.climate.copernicus.eu/cdsapp/#!/terms/licence-to-use-copernicus-products",
                    "source_url": "https://doi.org/10.24381/cds.bd0915c6"
                },
                "path": "era5_orography.nc"
            }
        }
        
        return orography
    
    def landcover_section(self) -> dict:
        """
        Write out landcover section

        Returns
        -------
        landcover : dict
            A dictionary contains information of landcover
        """
        # Polygons for land cover solutions
        if self.polygons is None:
            if self.landcover == 'globcover':
                landcover_file = 'original_globcover.tif'
            else:
                landcover_file = 'lcdb_2023_50m_fixed_nodata.tif'
        else:
            if self.landcover == 'globcover':
                landcover_file = max(
                    Path(self.hydromt_path).glob("globcover_*.tif"),
                    default=Path(self.hydromt_path) / "globcover_001.tif"
                ).name
            else:
                landcover_file = max(
                    Path(self.hydromt_path).glob("lcdb_*.tif"),
                    default=Path(self.hydromt_path) / "lcdb_001.tif"
                ).name

        # At the moment the name globcover and other information is kept fixed for basic automation,
        # it might be changed in the future.
        # NOTE: the information is not from globcover - it is LCDB converted to globcover
        # Generate a dictionary with landcover information
        landcover = {
            "landcover": {
                "crs": 4326,
                "data_type": "RasterDataset",
                "driver": "raster",
                "meta": {
                    "category": "landuse & landcover",
                    "paper_doi": "10.1594/PANGAEA.787668",
                    "paper_ref": "Arino et al (2012)",
                    "source_license": "CC-BY-3.0",
                    "source_url": "http://due.esrin.esa.int/page_globcover.php"
                },
                "path": landcover_file
            }
        }
        
        return landcover
    
    def lakes_section(self) -> dict:
        """
        Write out lakes section
        
        Returns
        -------
        lakes : dict
            A dictionary contains information of lakes
        """
        # At the moment the name hydro lakes and other information is kept fixed for basic automation,
        # it will be changed in the future.
        # Generate a dictionary with lakes' information
        lakes = {
            "hydro_lakes": {
                "crs": 4326,
                "data_type": "GeoDataFrame",
                "driver": "vector",
                "meta": {
                    "category": "surface water",
                    "source_author": "Arjen Haag"
                },
                "version": 1.0,
                "path": "hydro_lakes.gpkg",
                "unit_mult": {
                    "Area_avg": 1_000_000.0
                }
            }
        }
        
        return lakes
        
    def terrain_section(self) -> dict:
        """
        Write out terrain section

        Returns
        -------
        terrain : dict
            A dictionary contains information of terrain
        """
        # At the moment the name merit hydro and other information is kept fixed for basic automation,
        # it will be changed in the future.
        # NOTE: the data is not merit hydro - it is generated from geofabrics
        # Generate a dictionary with terrain information
        terrain = {
            "merit_hydrox": {
                "crs": 2193,
                "data_type": "RasterDataset",
                "driver": "raster",
                "meta": {
                    "category": "topography",
                    "paper_ref": "Yamazaki et al. (2019)"
                },
                "version": 1.0,
                "path": f"{self.wflow_model_path / 'merit_hydro/{variable}.tif'}"
            }
        }
        
        return terrain
    
    def basin_section(self) -> dict:
        """
        Write out basin section
        
        Returns
        -------
        basin : dict
            A dictionary contains information of basin
        """
        # At the moment the name merit hydro and other information is kept fixed for basic automation,
        # it will be changed in the future.
        # NOTE: the data is not merit hydro - it is generated from DEM of geofabrics
        # Generate a dictionary with basin information
        basin = {
            "merit_hydro_index": {
                "crs": 2193,
                "data_type": "GeoDataFrame",
                "driver": "vector",
                "meta": {
                    "category": "topography",
                    "paper_doi": "10.5194/hess-2020-582",
                    "paper_ref": "Eilander et al. (in review)",
                    "source_license": "CC-BY-NC 4.0"
                },
                "path": f"{self.wflow_model_path / 'merit_hydro_index.gpkg'}"
            }    
        }
        
        return basin
    
    def rivers_section(self) -> dict:
        """
        Write out rivers section

        Returns
        -------
        rivers : dict
            A dictionary contains information of rivers
        """
        # At the moment the name rivers lin is kept fixed for basic automation.
        # It will be changed in the future
        # Generate a dictionary with rivers information
        rivers = {
            "hydro_rivers_lin": {
                "data_type": "GeoDataFrame",
                "driver": "vector",
                "meta": {
                    "category": "hydrography",
                    "paper_doi": "10.5281/zenodo.3552776",
                    "paper_ref": "Lin et al. (2019)",
                    "source_license": "CC-BY-NC 4.0",
                    "source_url": "https://zenodo.org/record/3552776#.YVbOrppByUk",
                    "processing_notes": "hydrography/rivers_lin2019/README"
                },
                "version": 1,
                "path": f"{self.wflow_model_path / 'rivers_lin2019_v1.gpkg'}"
            }    
        }
        
        return rivers
    
    def soilgrids_section(self) -> dict:
        """
        Write out soilgrids section
        
        Returns
        -------
        soilgrids : dict
            A dictionary contains information of soilgrids
        """
        # Generate a dictionary with soilgrids' information
        soilgrids = {
            "soilgrids_2020": {
                "crs": 4326,
                "data_type": "RasterDataset",
                "driver": "raster",
                "meta": {
                    "category": "soil"
                },
                "version": 2020,
                "path": "soilgrids_2020/{variable}.tif"
            }
        }
        
        return soilgrids
        
    def lai_section(self) -> dict:
        """
        Write out LAI section
        
        Returns
        -------
        lai : dict
            A dictionary contains information of LAI
        """
        # At the moment, MODIS LAI is used,
        # but it will be changed into estimation in the future
        # Generate a dictionary of LAI
        if self.river_name == "whirinaki":
            crs = 2193

        else:
            # Mataura
            crs = 4326

        # Set up lai directory
        lai_path = str(Path(self.hydromt_path / fr"river_data/{self.river_name}/modis_lai.nc"))

        # Set up lai information for wflow
        lai = {
            "modis_lai": {
                "crs": crs,
                "data_type": "RasterDataset",
                "driver": "netcdf",
                "meta": {
                    "category": "landuse & landcover"
                },
                "version": 6,
                "path": lai_path,
                "unit_mult": {
                    "LAI": 0.1
                }
            }
        }
        
        return lai
    
    def data_catalog_section(self) -> dict:
        """
        Organise data_catalog's content

        Returns
        -------
        data_catalog : dict
            A dictionary contains information of all sections
        """
        # Set up data_catalog dictionary
        data_catalog = {}

        # If update land cover
        if self.polygons is not None:
            sections_list = [
                self.meta_section(),
                self.landcover_section(),
                self.lai_section()
            ]

        # If not, means start from beginning
        else:
            # Set up list of all sections
            if str(self.forcing_path).endswith(".nc"):
                sections_list = [
                    self.meta_section(),
                    self.orography_section(),
                    self.landcover_section(),
                    self.lakes_section(),
                    self.terrain_section(),
                    self.basin_section(),
                    self.rivers_section(),
                    self.soilgrids_section(),
                    self.lai_section()
                ]
            else:
                sections_list = [
                    self.meta_section(),
                    self.forcing_section(),
                    self.orography_section(),
                    self.landcover_section(),
                    self.lakes_section(),
                    self.terrain_section(),
                    self.basin_section(),
                    self.rivers_section(),
                    self.soilgrids_section(),
                    self.lai_section()
                ]
        
        # Generate a dictionary of data_catalog
        for each_section in sections_list:
            data_catalog.update(each_section)
            
        return data_catalog
    
    def write_out_data_catalog(
            self,
            data_catalog
        ) -> None:
        """
        Write out data_catalog.yml
        
        Parameters
        ----------
        data_catalog : dict
            A dictionary contains information of all sections
        """
        if self.polygons is not None:
            # Find existing file
            existing_file = sorted(
                self.wflow_model_path.glob("data_catalog_landcover_*.yml")
            )

            # Set ID for file
            number = len(existing_file) + 1

            # Create file name
            output_filename = self.wflow_model_path / f"data_catalog_landcover_{number:03d}.yml"

        else:
            # Set up output filename
            output_filename = self.wflow_model_path / "data_catalog.yml"
        
        # Write out data_catalog.yml
        with open(output_filename, "w") as output_file:
            yaml.dump(
                data_catalog, 
                output_file,
                sort_keys=False
            )
        
    def data_catalog_generator(self):
        """Generate data_catalog.yml file"""
        # Set up content for data_catalog file
        log.debug(f"Generating data_catalog.yml file")
        data_catalog = self.data_catalog_section()
        
        # Write data_catalog file
        self.write_out_data_catalog(data_catalog)