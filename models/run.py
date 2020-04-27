import os
import pickle
import json
import zipfile
import numpy as np
import datetime
from collections import defaultdict
from joblib import Parallel, delayed
import config
from plots.plot_risk import hist_plot
from models.risk_models import RiskModelTristan
from models.inference_client import InferenceClient
from frozen.utils import encode_message, update_uid, encode_update_message, decode_message


def query_inference_server(params):
    ports = [6688]
    client = InferenceClient(ports)
    results = client.infer(pickle.dumps(params))
    return results

def get_days_worth_of_logs(data_path, start, cur_day, start_pkl):
    to_return = defaultdict(list)
    started = False
    try:
        with zipfile.ZipFile(data_path, 'r') as zf:
            for pkl in zf.namelist():
                if not started:
                    if pkl != start_pkl:
                        continue
                started = True
                start_pkl = pkl
                logs = pickle.load(zf.open(pkl, 'r'))
                from base import Event

                for log in logs:
                    if log['event_type'] == Event.encounter:
                        day_since_epoch = (log['time'] - start).days
                        if day_since_epoch == cur_day:
                            to_return[log['human_id']].append(log)
                        elif day_since_epoch > cur_day:
                            return to_return, start_pkl
    except Exception:
        pass
    return to_return, start_pkl

def integrated_risk_pred(humans, data_path, start, current_day, all_possible_symptoms, start_pkl, n_jobs=1):
    if config.RISK_MODEL:
        RiskModel = RiskModelTristan

    # check that the plot_dir exists:
    if config.PLOT_RISK and not os.path.isdir(config.RISK_PLOT_PATH):
        os.mkdir(config.RISK_PLOT_PATH)

    days_logs, start_pkl = get_days_worth_of_logs(data_path + ".zip", start, current_day, start_pkl)
    all_params = []
    hd = {human.name: human for human in humans}
    for human in humans:
        encounters = days_logs[human.name]
        log_path = f'{os.path.dirname(data_path)}/daily_outputs/{current_day}/{human.name[6:]}/'
        # go about your day accruing encounters and clustering them
        for encounter in encounters:
            encounter_time = encounter['time']
            unobs = encounter['payload']['unobserved']
            encountered_human = hd[unobs['human2']['human_id']]
            message = encode_message(encountered_human.cur_message(current_day, RiskModel))
            encountered_human.sent_messages[str(unobs['human1']['human_id']) + "_" + str(encounter_time)] = message
            human.messages.append(message)

            got_exposed = encounter['payload']['unobserved']['human1']['got_exposed']
            if got_exposed:
                human.exposure_message = message

        # if the encounter happened within the last 14 days, and your symptoms started at most 3 days after your contact
        if RiskModel.quantize_risk(human.start_risk) != RiskModel.quantize_risk(human.risk):
            sent_at = start + datetime.timedelta(days=current_day, minutes=human.rng.randint(low=0, high=1440))
            for k, m in human.sent_messages.items():
                message = decode_message(m)
                if current_day - message.day < 14:
                    # add the update message to the receiver's inbox
                    update_message = encode_update_message(
                        human.cur_message_risk_update(message.day, message.risk, sent_at, RiskModel))
                    hd[k.split("_")[0]].update_messages.append(update_message)
            human.sent_messages = {}
        all_params.append({"start": start, "current_day": current_day, "encounters": encounters,
                           "all_possible_symptoms": all_possible_symptoms, "human": human,
                           "save_training_data": True, "log_path": log_path,
                           "random_clusters": False})
        human.uid = update_uid(human.uid, human.rng)

    with Parallel(n_jobs=n_jobs, batch_size=config.MP_BATCHSIZE, backend=config.MP_BACKEND, verbose=1, prefer="threads") as parallel:
        results = parallel((delayed(query_inference_server)(params) for params in all_params))

    for name, risk, clusters in results:
        hd[name].risk = risk
        hd[name].clusters = clusters

    if config.PLOT_RISK and config.COLLECT_LOGS:
        daily_risks = [(np.e ** human.risk, human.is_infectious, human.name) for human in hd.values()]
        hist_plot(daily_risks, f"{config.RISK_PLOT_PATH}day_{str(current_day).zfill(3)}.png")

    # print out the clusters
    if config.DUMP_CLUSTERS and config.COLLECT_LOGS:
        clusters = []
        for human in hd.values():
            clusters.append(dict(human.clusters.clusters))
        json.dump(clusters, open(config.CLUSTER_PATH, 'w'))

    return humans, start_pkl
