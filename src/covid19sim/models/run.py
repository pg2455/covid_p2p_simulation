"""
Handles querying the inference server with serialized humans and their messages.
"""

from datetime import timedelta
import os
import pickle
import functools
from joblib import Parallel, delayed
import warnings

from covid19sim.server_utils import InferenceClient, InferenceEngineWrapper, proc_human_batch
from covid19sim.configs.exp_config import ExpConfig


def integrated_risk_pred(humans, start, current_day, time_slot, all_possible_symptoms, data_path=None):
    """
    [summary]
    Setup and make the calls to the server

    Args:
        humans ([type]): [description]
        start ([type]): [description]
        current_day ([type]): [description]
        time_slot ([type]): [description]
        all_possible_symptoms ([type]): [description]
        data_path ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    hd = humans[0].city.hd
    all_params = []

    current_time = (start + timedelta(days=current_day, hours=time_slot))

    for human in humans:
        if time_slot not in human.time_slots:
            continue

        human_state = human.get_message_dict()

        log_path = None
        if data_path:
            log_path = f'{os.path.dirname(data_path)}/daily_outputs/{current_day}/{human.name[6:]}/'
        all_params.append({
            "start": start,
            "current_day": current_day,
            "all_possible_symptoms": all_possible_symptoms,
            "human": human_state,
            "log_path": log_path,
            "config_path": ExpConfig.config_path,
            "time_slot": time_slot,
        })
        human.contact_book.update_messages = []
        human.contact_book.messages = []

    parallel_reqs = ExpConfig.config.get('INFERENCE_REQ_PARALLEL_JOBS', 8)
    if ExpConfig.get('USE_INFERENCE_SERVER'):
        batch_start_offset = 0
        batch_size = ExpConfig.config.get('INFERENCE_REQ_BATCH_SIZE', 300)
        batched_params = []
        while batch_start_offset < len(all_params):
            batch_end_offset = min(batch_start_offset + batch_size, len(all_params))
            batched_params.append(all_params[batch_start_offset:batch_end_offset])
            batch_start_offset += batch_size
        parallel_reqs = max(min(parallel_reqs, len(batched_params)), 1)

        def query_inference_server(params, **inf_client_kwargs):
            # lambda used to create one socket per request (so we can request in parallel)
            client = InferenceClient(**inf_client_kwargs)
            return client.infer(params)

        inference_frontend_address = ExpConfig.config.get('INFERENCE_SERVER_ADDRESS', None)
        query_func = functools.partial(query_inference_server, server_address=inference_frontend_address)

        with Parallel(n_jobs=parallel_reqs, backend="loky", prefer="threads") as parallel:
            batched_results = parallel((delayed(query_func)(params) for params in batched_params))
        results = []
        for b in batched_results:
            results.extend(b)
    else:
        # recreating an engine every time should not be too expensive... right?
        engine = InferenceEngineWrapper(ExpConfig.get('TRANSFORMER_EXP_PATH'))
        results = proc_human_batch(all_params, engine, "loky", parallel_reqs)

    for result in results:
        if result is not None:
            name, risk_history, clusters = result
            if ExpConfig.get('RISK_MODEL') == "transformer":
                # TODO: Fix can be None. What should be done in this case
                if risk_history is not None:
                    # risk_history = np.clip(risk_history, 0., 1.)
                    for i in range(len(risk_history)):
                        hd[name].risk_history_map[current_day - i] = risk_history[i]
                    hd[name].update_risk_level()
                    for i in range(len(risk_history)):
                        hd[name].prev_risk_history_map[current_day - i] = risk_history[i]
                elif current_day != ExpConfig.get('INTERVENTION_DAY'):
                    warnings.warn(f"risk history is none for human:{name}", RuntimeWarning)
                hd[name].last_risk_update = current_time
            hd[name].clusters = clusters
            hd[name].last_cluster_update = current_time

    # print out the clusters
    if ExpConfig.get('DUMP_CLUSTERS'):
        os.makedirs(ExpConfig.get('DUMP_CLUSTERS'), exist_ok=True)
        curr_date_str = current_time.strftime("%Y%m%d-%H%M%S")
        curr_dump_path = os.path.join(ExpConfig.get('DUMP_CLUSTERS'), curr_date_str + ".pkl")
        to_dump = {human_id: human.clusters for human_id, human in hd.items()
                   if human.last_cluster_update == current_time}
        with open(curr_dump_path, "wb") as fd:
            pickle.dump(to_dump, fd)

    return humans
