import logging
from pathlib import Path

import geopandas as gpd
import rioxarray as rxr
from sqlalchemy.ext.declarative import declarative_base

from eddie import geoserver as gs
from eddie.digitaltwin import setup_environment
from eddie.digitaltwin.utils import setup_logging
from src.eddie_floodresilience.config import EnvVariable
from src.eddie_floodresilience.hydrological.wflow_data_catalog_generator import find_landcover_file

Base = declarative_base()

setup_logging()
log = logging.getLogger(__name__)


class WflowServeDataGenerator:
    def __init__(self, hydromt_path: Path, wflow_model_path: Path, landcover: str, flood_model_output_id: int):
        self.hydromt_path = hydromt_path
        self.wflow_model_path = wflow_model_path
        self.landcover = landcover
        self.flood_model_output_id = flood_model_output_id

    def serve_data(self):
        if not EnvVariable.IS_GEOSERVER_ACTIVE:
            return
        print("serve_data")
        db_name = EnvVariable.POSTGRES_DB
        workspace_name = f"{db_name}-intermediate-wflow"

        catchment_file = self.wflow_model_path / "wflow_test_full/staticgeoms/basins.geojson"

        catchment_poly = gpd.read_file(catchment_file)
        if catchment_poly.crs.to_epsg() is None:
            raise KeyError(f"CRS is not defined in EPSG# form in vector file {vector_file_path}.")
        catchment_poly = catchment_poly.to_crs(epsg=2193)

        log.info(f"Adding catchment file '{catchment_file}' to GeoServer.")
        self._serve_catchment_boundary(workspace_name, catchment_poly)
        self._serve_landcover_file(workspace_name, catchment_poly)

    def _serve_landcover_file(self, workspace_name: str, catchment_poly: gpd.GeoDataFrame):
        is_baseline = self.landcover == "globcover"
        landcover_file = find_landcover_file(self.wflow_model_path, self.hydromt_path, self.landcover, is_baseline)

        tmp_dir = Path("tmp/gtiff") / self.hydromt_path.name
        tmp_dir.mkdir(parents=True, exist_ok=True)
        clipped_path = tmp_dir / landcover_file.name

        with rxr.open_rasterio(landcover_file) as src:
            catchment = catchment_poly.to_crs(src.rio.crs).geometry
            clipped_landcover = src.rio.clip(catchment, drop=True)
            clipped_landcover.rio.to_raster(clipped_path)

        layer_name = f"landcover_{self.flood_model_output_id}"
        if layer_name not in gs.raster_layers.get_workspace_raster_layers(workspace_name):
            gs.add_gtiff_to_geoserver(clipped_path, workspace_name, layer_name)
        # Delete tmp clipped file
        clipped_path.unlink()

    def _serve_catchment_boundary(self, workspace_name: str, catchment_poly: gpd.GeoDataFrame):
        data_store = gs.create_main_db_store(workspace_name)
        # Extract the geometry of the catchment area
        catchment_poly["flood_model_id"] = self.flood_model_output_id
        catchment_poly = catchment_poly[["flood_model_id", "geometry"]]

        engine = setup_environment.get_database()
        with engine.connect() as conn:
            # Create the 'wflow_catchment_boundary' table if it doesn't exist
            table_name = "wflow_catchment_boundary"
            catchment_poly.to_postgis(table_name, conn, if_exists="append", index=False)
            gs.create_datastore_layer(conn, workspace_name, data_store, table_name)
