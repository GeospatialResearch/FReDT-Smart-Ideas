"""Endpoints and flask configuration for the Flood Resilience Digital Twin"""
import os
import pathlib
from http.client import OK

from flask import Blueprint, Response, make_response

from eddie.check_celery_alive import check_celery_alive
from src.eddie_floodresilience import hydrological_and_hydrodynamic_service as hh_service, tasks

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


@blueprint.route('/hydrographs/scenarios/<int:scenario_id>/features/<string:feature_id>')
@check_celery_alive
def retrieve_hydrograph(scenario_id: int, feature_id: str) -> Response:
    """
    Find hydrograph data for the given scenario and feature as CSV format.

    Parameters
    ----------
    scenario_id: str
        The flood model output ID to find query hydrograph data for.
    feature_id: str
        The FID of the specific injection point to query hydrograph data for.

    Returns
    -------
    Response
        Response with body containing hydrograph data in CSV format.
    """
    get_hydrograph_task = tasks.read_hydrograph_data.delay(scenario_id, feature_id)
    hydrograph_data = get_hydrograph_task.get()

    response = make_response(hydrograph_data, OK)
    response.content_type = "text/csv"
    return response
