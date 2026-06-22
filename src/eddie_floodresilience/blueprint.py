"""Endpoints and flask configuration for the Flood Resilience Digital Twin"""
import os
import pathlib

from flask import Blueprint

from eddie.check_celery_alive import check_celery_alive

from src.eddie_floodresilience import hydrological_and_hydrodynamic_service as hh_service

os.environ.pop("Path", None)
# See issue https://github.com/GeospatialResearch/eddie_floodresilience/issues/1 for reason behind disabled QA
from pywps import Service  # pylint: disable=wrong-import-position,wrong-import-order # noqa: E402

blueprint = Blueprint('eddie_floodresilience', __name__)
processes = [
    hh_service.Whirinaki1999BaselineProcessService(),
    hh_service.Whirinaki1999ScenarioProcessService(),
    hh_service.Mataura2020BaselineProcessService(),
    hh_service.Mataura2020ScenarioProcessService(),
]

for working_dir in ["workdir", "outputs", "logs"]:
    path = pathlib.Path("./tmp/pywps") / working_dir
    path.mkdir(exist_ok=True, parents=True)
process_descriptor = {process.identifier: process.abstract for process in processes}
service = Service(processes, ['src/pywps.cfg'])


@blueprint.route('/wps', methods=['GET', 'POST'])
@check_celery_alive
def wps() -> Service:
    """
    End point for OGC WebProcessingService spec, allowing clients such as TerriaJS to request processing.

    Returns
    -------
    Service
        The PyWPS WebProcessing Service instance
    """
    return service
