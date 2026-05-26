from datetime import datetime
from itertools import product
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.eddie_floodresilience.config import EnvVariable

from .hydrological_and_hydrodynamic_pipeline import HydrologicalAndHydrodynamicPipeline

DATETIME_FORMAT = "%Y-%m-%d %H_%M_%S"


def create_hhp(num_threads: int, resolution: int) -> HydrologicalAndHydrodynamicPipeline:
    hydro_combination_path = EnvVariable.HYDRO_COMBINATION_PATH
    outlet_gauge_locations_filename = 'river_outlet'
    forcing_name = 'whirinaki'
    river_name = 'whirinaki'
    precipitation_path = EnvVariable.PRECIPITATION_PATH
    start_time = datetime.fromisoformat("1999-01-20T00:00:00")
    end_time = datetime.fromisoformat("1999-01-22T12:00:00")

    subbasin = [1642072.60, 6076218.85]
    bbox = [1639968.20, 6058374.30, 1670723.51, 6114366.30]

    flood_aoi_boundary = [1641145.361, 6072406.885, 1642792.613, 6076268]
    adjust_manning = True

    polygons = None  # r'polygons/polygons.shp'
    vectors = None  # r'vectors/vectors.csv'
    strord = 4
    threshold = 1000
    width_rate_control = 1 / 20
    discharge_rate_control = 1
    crs = 2193

    # Set up hydraulic and hydrodynamic pipeline
    hydrological_hydrodynamic_pipeline = HydrologicalAndHydrodynamicPipeline(
        hydro_combination_path,
        outlet_gauge_locations_filename,

        forcing_name,
        river_name,
        precipitation_path,
        start_time,
        end_time,

        subbasin,
        bbox,
        num_threads,
        flood_aoi_boundary,
        adjust_manning,

        polygons,
        vectors,
        strord,
        resolution,
        threshold,
        width_rate_control,
        discharge_rate_control,
        crs
    )
    return hydrological_hydrodynamic_pipeline


def run_terrain_trial(pipeline: HydrologicalAndHydrodynamicPipeline) -> float:
    # Time the process
    start_time = time.perf_counter()
    pipeline.terrain_data_pipeline()
    end_time = time.perf_counter()

    return end_time - start_time


def run_wflow_trial(pipeline: HydrologicalAndHydrodynamicPipeline) -> float:
    # Time the process
    start_time = time.perf_counter()
    pipeline.wflow_data_pipeline()
    end_time = time.perf_counter()

    return end_time - start_time


def plot_resolution_trials(title: str, trials: pd.DataFrame):
    # Aggregate: mean time_taken per (n_threads, resolution)
    agg = trials.groupby(['n_threads', 'resolution'])['time_taken'].mean().reset_index()

    # Setup
    threads = sorted(agg['n_threads'].unique())
    resolutions = sorted(agg['resolution'].unique())
    n_threads = len(threads)
    n_res = len(resolutions)

    # Bar positioning
    width = 0.35
    x = np.arange(n_res)
    offsets = np.linspace(-(n_threads - 1) / 2, (n_threads - 1) / 2, n_threads) * width

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for i, thread in enumerate(threads):
        subset = agg[agg['n_threads'] == thread].set_index('resolution')
        heights = [subset.loc[r, 'time_taken'] if r in subset.index else 0 for r in resolutions]
        ax.bar(x + offsets[i], heights, width=width * 0.9, label=f'{thread} threads', zorder=3)

    # Grid
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

    # Labels & ticks
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in resolutions])
    ax.set_xlabel('Resolution (m)')
    ax.set_ylabel('Time Taken (s)')
    ax.set_title(title)
    ax.legend(title='Number of Threads')

    plt.tight_layout()
    plt.show()


def plot_threads_trials(trials: pd.DataFrame, resolution: int):
    df = trials.sort_values('n_threads')
    df.plot.bar(
        title=f"Effect of number of threads on wflow ({resolution}m resolution)",
        x="n_threads",
        xlabel="Number of threads",
        y="time_taken",
        ylabel="Time taken (s)",
        rot=0,
        legend=False
    )
    plt.show()


def run_resolution_trials(logging: bool = True):
    n_threads = [64]
    # n_threads = [64, 16]
    resolutions = [
        # 1000,
        # 500,
        # 250,
        100,
        50,
        # 25
    ]
    trials = list(product(n_threads, resolutions))
    terrain_trial_log = []
    wflow_trial_log = []
    print("Running resolution trials...")
    for n, r in tqdm(trials):
        pipeline = create_hhp(n, r)
        timestamp = datetime.now().strftime(DATETIME_FORMAT)
        try:
            terrain_time = run_terrain_trial(pipeline)
        except Exception as e:
            terrain_time = -1
            print("resolution trial failed at terrain")
            print(e.with_traceback())
        terrain_trial_log.append({"trial_type": "terrain", "n_threads": n, "resolution": r, "time_taken": terrain_time})
        if logging:
            # Write data each loop. Costs a little resources but allows for logging in case process needs to stop
            terrain_df = pd.DataFrame(terrain_trial_log)
            terrain_df.to_csv(Path(f"benchmarks/benchmark_terrain_resolution_{timestamp}.csv"), index=False)

        try:
            wflow_time = run_wflow_trial(pipeline)
        except Exception as e:
            wflow_time = -1
            print("resolution trial failed at wflow")
            print(e.with_traceback())
        wflow_trial_log.append({"trial_type": "wflow", "n_threads": n, "resolution": r, "time_taken": wflow_time})
        if logging:
            wflow_df = pd.DataFrame(wflow_trial_log)
            wflow_df.to_csv(Path(f"benchmarks/benchmark_wflow_resolution_{timestamp}.csv"), index=False)

    terrain_df = pd.DataFrame(terrain_trial_log)
    wflow_df = pd.DataFrame(wflow_trial_log)
    plot_resolution_trials("Effect of resolution on terrain", terrain_df)
    plot_resolution_trials("Effect of resolution on wflow", wflow_df)


def run_threads_trials(logging: bool = True):
    n_threads = [
        128,
        64,
        32,
        16,
        8,
        4,
        2,
        1
    ]
    resolution = 50
    print("Prerunning terrain for threads trial.")
    run_terrain_trial(create_hhp(64, resolution))
    trial_log = []
    print("Running wflow for threads trial....")
    for n in tqdm(n_threads):
        pipeline = create_hhp(n, resolution)
        try:
            wflow_time = run_wflow_trial(pipeline)
        except Exception as e:
            wflow_time = -1
            print("threads trial failed at wflow")
            print(e.with_traceback())
        trial_log.append({"trial_type": "wflow", "n_threads": n, "resolution": resolution, "time_taken": wflow_time})

        if logging:
            # Write data each loop. Costs a little resources but allows for logging in case process needs to stop
            trial_df = pd.DataFrame(trial_log)
            current_time = datetime.now().strftime(DATETIME_FORMAT)
            trial_df.to_csv(Path(f"benchmarks/benchmark_threads_{current_time}.csv"), index=False)

    trial_df = pd.DataFrame(trial_log)
    plot_threads_trials(trial_df, resolution)


if __name__ == '__main__':
    Path("benchmarks").mkdir(parents=True, exist_ok=True)
    run_resolution_trials()
    run_threads_trials()