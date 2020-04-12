import pandas as pd
import numpy as np
from collections import defaultdict
from config import HUMAN_DISTRIBUTION
import networkx as nx

def get_nested_dict(nesting):
    if nesting == 1:
        return defaultdict(int)
    elif nesting == 2:
        return defaultdict(lambda : defaultdict(int))
    elif nesting == 3:
        return defaultdict(lambda : defaultdict(lambda : defaultdict(int)))
    elif nesting == 4:
        return defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : defaultdict(int))))

class Tracker(object):
    def __init__(self, env, city):
        self.env = env
        # infection
        self.contacts = {
                'all':get_nested_dict(2),
                'location_all': get_nested_dict(3),
                'human_infection': get_nested_dict(2),
                'env_infection':get_nested_dict(1),
                'location_env_infection': get_nested_dict(2),
                'location_human_infection': get_nested_dict(3)
                }

        self.infection_graph = nx.DiGraph()
        self.transition_probability = get_nested_dict(4)

        self.sar = []
        self.avg_infection_duration = 0
        self.n_recovery = 0
        self.n_infectious_contacts = 0
        self.n_contacts = 0
        self.avg_generation_times = (0,0)
        self.generation_time_book = {}

        # demographics
        self.age_bins = sorted(HUMAN_DISTRIBUTION.keys(), key = lambda x:x[0])
        self.age_distribution = []
        self.households_age = []

        self.summarize_population(city)

    def summarize_population(self, city):
        age = pd.DataFrame([h.age for h in city.humans])
        house_age = pd.DataFrame([np.mean([h.age for h in house.residents]) for house in city.households])
        house_size = pd.DataFrame([len(house.residents) for house in city.households])
        print("age distribution\n", age.describe())
        print("house age distribution\n", house_age.describe())
        print("house size distribution\n", house_size.describe())

    def track_contact(self, human1, human2, location):
        for i, (l,u) in enumerate(self.age_bins):
            if l <= human1.age < u:
                bin1 = i
            if l <= human2.age < u:
                bin2 = i

        self.contacts["all"][bin1][bin2] += 1
        self.contacts["location_all"][location.location_type][bin1][bin2] += 1
        self.n_contacts += 1

    def get_R0(self):
        # https://web.stanford.edu/~jhj1/teachingdocs/Jones-on-R0.pdf
        # average infectious contacts (transmission) * average number of contacts * average duration of infection
        time_since_start =  (self.env.timestamp - self.env.initial_timestamp).total_seconds() / 86400 # DAYS
        if time_since_start == 0:
            return -1
        # tau= self.n_infectious_contacts / self.n_contacts
        # c_bar = self.n_contacts / time_since_start
        tau_times_c_bar = self.n_infectious_contacts / time_since_start
        d = self.avg_infection_duration
        return tau_times_c_bar * d

    def get_generation_time(self):
        return self.avg_generation_times[1]

    def track_infection(self, type, from_human, to_human, location, timestamp):
        for i, (l,u) in enumerate(self.age_bins):
            if from_human and l <= from_human.age < u:
                from_bin = i
            if l <= to_human.age < u:
                to_bin = i

        if type == "human":
            self.contacts["human_infection"][from_bin][to_bin] += 1
            self.contacts["location_human_infection"][location.location_type][from_bin][to_bin] += 1

            self.n_infectious_contacts += 1
            delta = timestamp - from_human.infection_timestamp
            self.infection_graph.add_node(from_human.name, bin=from_bin, time=from_human.infection_timestamp)
            self.infection_graph.add_node(to_human.name, bin=to_bin, time=timestamp)
            self.infection_graph.add_edge(from_human.name, to_human.name,  timedelta=delta)

            if from_human.symptom_start_time is not None:
                self.generation_time_book[to_human.name] = from_human.symptom_start_time
        else:
            self.contacts["env_infection"][to_bin] += 1
            self.contacts["location_env_infection"][location.location_type][to_bin] += 1
            self.infection_graph.add_node(to_human.name, bin=to_bin, time=timestamp)
            self.infection_graph.add_edge(-1, to_human.name,  timedelta="")

    def track_generation_times(self, human_name):
        if human_name not in self.generation_time_book:
            return

        generation_time = (self.env.timestamp - self.generation_time_book.pop(human_name)).total_seconds() / 86400 # DAYS
        n, avg_gen_time = self.avg_generation_times
        self.avg_generation_times = (n+1, (avg_gen_time * n + generation_time)/(n+1))

    def track_recovery(self, duration):
        self.avg_infection_duration = (self.n_recovery * self.avg_infection_duration + duration) / (self.n_recovery + 1)
        self.n_recovery += 1

    def summarize_contacts(self):
        x = pd.DataFrame.from_dict(self.contacts['all'])
        x = x[sorted(x.columns)]
        fig = x.iplot(kind='heatmap', asFigure=True, title="all_contacts")
        fig.show()

    def track_trip(self, from_location, to_location, age, hour):
        bin = None
        for i, (l,u) in enumerate(self.age_bins):
            if l <= age < u:
                bin = i
        if bin is None: import pdb; pdb.set_trace()
        self.transition_probability[hour][bin][from_location][to_location] += 1
