import pickle
import json
from base import Event
import numpy as np
from config import *
from utils import binary_to_float
import matplotlib.pyplot as plt
from utils import _encode_message, _decode_message
from bitarray import bitarray
import operator
from collections import defaultdict
import datetime
from tqdm import tqdm
from plots.validate_risk_parameters import dist_plot

class dummy_env:
    def __init__(self):
        return

if __name__ == "__main__":
    PATH_TO_DATA = "data.pkl"
    PATH_TO_HUMANS = "humans.pkl"
    PATH_TO_PLOT = "plots/infected_dist.png"

    with open(PATH_TO_DATA, "rb") as f:
        logs = pickle.load(f)
    with open(PATH_TO_HUMANS, "rb") as f:
        humans = pickle.load(f)
    enc_logs = [l for l in logs if l["event_type"] == Event.encounter]

    hd = {}
    for human in humans:
        try:
            human.infection_timestamp = datetime.datetime.strptime(human.infection_timestamp, '%Y-%m-%d %H:%M:%S')
            print(human.name)
        except Exception:
            human.infection_timestamp = None

        env = dummy_env()
        env.timestamp = datetime.datetime(2020, 2, 28, 0, 0)
        human.env = env
        human.rng = np.random.RandomState(0)
        human.update_uid()

        hd[human.name] = human

    i = 0
    risks = []
    risk_vs_infected = []

    for log in tqdm(enc_logs):
        i+=1
        now = log['time']
        unobs = log['payload']['unobserved']
        h1 = unobs['human1']['human_id']
        h2 = unobs['human2']['human_id']
        this_human = hd[h1]
        other_human = hd[h2]
        this_human.env.timestamp = now
        other_human.env.timestamp = now
        this_human.pending_messages.append(other_human.cur_message(now))

        if this_human.cur_day != now.day:
            # this_human.update_initial_risk()
            this_human.cur_day = now.day
            this_human.update_uid()
            # if this_human.risk != 0:
            #     import pdb; pdb.set_trace()
            for j in range(len(this_human.pending_messages)):
                m_j = this_human.pending_messages.pop()
                quantized_risk = binary_to_float("".join([str(x) for x in np.array(m_j[1].tolist()).astype(int)]), 0, 4)
                this_human.handle_message(m_j)
            risk_vs_infected.append((this_human.risk, this_human.is_infectious))

    # not sure why this is breaking
    dist_plot(risk_vs_infected, PATH_TO_PLOT)

    # contact_histories = []
    # for human in humans:
    #     contact_histories.append(human.A)
    # json.dump(contact_histories, open('contact_histories.json', 'w'))


