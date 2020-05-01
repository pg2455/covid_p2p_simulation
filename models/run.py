import os
from datetime import timedelta
import pickle
import json
import zipfile
import time
import numpy as np
import functools
from collections import defaultdict
from joblib import Parallel, delayed
import config
from plots.plot_risk import hist_plot
from models.inference_client import InferenceClient
from frozen.utils import encode_message, update_uid, encode_update_message, decode_message

# load the risk map (this is ok, since we only do this #days)
risk_map = np.load(f"{os.path.dirname(os.path.realpath(__file__))}/log_risk_mapping.npy")
risk_map[0] = np.log(0.01)


def query_inference_server(params, **inf_client_kwargs):
    client = InferenceClient(**inf_client_kwargs)
    results = client.infer(params)
    return results


def integrated_risk_pred(humans, data_path, start, current_day, all_possible_symptoms, start_pkl, port=6688, n_jobs=1):
    risk_pred_start = time.time()
    # check that the plot_dir exists:
    if config.PLOT_RISK:
        os.makedirs(config.RISK_PLOT_PATH, exist_ok=True)

    hd = humans[0].city.hd
    all_params = []

    current_date = (start + timedelta(days=max(0, current_day-1))).date()

    for human in humans:
        if human.last_date['run'] != current_date:
            if human.dead or human.obs_hospitalized:
                human.infectiousnesses.appendleft(0.0)

        log_path = f'{os.path.dirname(data_path)}/daily_outputs/{current_day}/{human.name[6:]}/'

        all_params.append({"start": start, "current_day": current_day,
                           "all_possible_symptoms": all_possible_symptoms, "human": human.get_optimized_state(),
                           "COLLECT_TRAINING_DATA": config.COLLECT_TRAINING_DATA, "log_path": log_path, "risk_model": config.RISK_MODEL})
        human.uid = update_uid(human.uid, human.rng)

    batch_start_offset = 0
    batch_size = 25  # @@@@ TODO: make this a high-level configurable arg?
    batched_params = []
    while batch_start_offset < len(all_params):
        batch_end_offset = min(batch_start_offset + batch_size, len(all_params))
        batched_params.append(all_params[batch_start_offset:batch_end_offset])
        batch_start_offset += batch_size

    query_func = functools.partial(query_inference_server, target_port=port)

    with Parallel(n_jobs=n_jobs, batch_size=config.MP_BATCHSIZE, backend=config.MP_BACKEND, verbose=10, prefer="threads") as parallel:
        batched_results = parallel((delayed(query_func)(params) for params in batched_params))

    results = []
    for b in batched_results:
        results.extend(b)

    for result in results:
        if result is not None:
            name, risk, clusters = result
            if config.RISK_MODEL == "transformer":
                hd[name].risk = risk
                hd[name].update_risk_level()

            hd[name].clusters = clusters

    # TODO: @PRATEEK setup similar metrics to those on the Transformer for the Naive method
    if config.PLOT_RISK and config.COLLECT_LOGS:
        daily_risks = [(human.risk, human.is_infectious, human.name) for human in hd.values()]
        hist_plot(daily_risks, f"{config.RISK_PLOT_PATH}/day_{str(current_day).zfill(3)}.png")

    # print out the clusters
    if config.DUMP_CLUSTERS and config.COLLECT_LOGS:
        clusters = []
        for human in hd.values():
            clusters.append(dict(human.clusters.clusters))
        json.dump(clusters, open(config.CLUSTER_PATH, 'w'))
    print(f"{current_day} took {time.time() - risk_pred_start}")
    return humans, start_pkl

