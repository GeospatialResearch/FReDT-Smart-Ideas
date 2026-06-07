"""Endpoints and flask configuration for the Flood Resilience Digital Twin"""
from http.client import ACCEPTED
import os
import pathlib

from flask import Blueprint, jsonify, make_response, Response

from eddie.check_celery_alive import check_celery_alive

from src.eddie_floodresilience import hydrological_and_hydrodynamic_service as hh_service
from src.eddie_floodresilience import tasks

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


@blueprint.route('/datasets/update', methods=["POST"])
@check_celery_alive
def refresh_lidar_data_sources() -> Response:
    """
    Update LiDAR data sources to the most recent.
    Web-scrape OpenTopography metadata to update the datasets table containing links to LiDAR data sources.
    Takes a long time to run but needs to be run periodically so that the datasets are up to date.
    Supported methods: POST

    Returns
    -------
    Response
        ACCEPTED is the expected response. Response body contains Celery taskId
    """
    # Start task to refresh lidar datasets
    task = tasks.refresh_lidar_datasets.delay()
    # Return HTTP Response with task id, so it can be monitored with get_status(taskId)
    return make_response(
        jsonify({"taskId": task.id}),
        ACCEPTED
    )
