import pandas as pd
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
    def __init__(self, city):
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
        self.r = []

        self.sar = []

        # demographics
        self.age_bins = sorted(HUMAN_DISTRIBUTION.keys(), key = lambda x:x[0])
        self.age_distribution = []
        self.households_age = []

        self.summarize_population(city)

    def summarize_population(self, city):
        age = [h.age for h in city.humans]
        print(pd.DataFrame(age).describe())

    def track_contact(self, human1, human2, location):
        for i, (l,u) in enumerate(self.age_bins):
            if l <= human1.age < u:
                bin1 = i
            if l <= human2.age < u:
                bin2 = i

        self.contacts["all"][bin1][bin2] += 1
        self.contacts["location_all"][location.location_type][bin1][bin2] += 1

    def track_infection(self, type, from_human, to_human, location, timestamp):
        for i, (l,u) in enumerate(self.age_bins):
            if from_human and l <= from_human.age < u:
                from_bin = i
            if l <= to_human.age < u:
                to_bin = i

        if type == "human":
            self.contacts["human_infection"][from_bin][to_bin] += 1
            self.contacts["location_human_infection"][location.location_type][from_bin][to_bin] += 1

            delta = timestamp - from_human.infection_timestamp
            self.infection_graph.add_node(from_human.name, bin=from_bin, time=from_human.infection_timestamp)
            self.infection_graph.add_node(to_human.name, bin=to_bin, time=timestamp)
            self.infection_graph.add_edge(from_human.name, to_human.name,  timedelta=delta)
        else:
            self.contacts["env_infection"][to_bin] += 1
            self.contacts["location_env_infection"][location.location_type][to_bin] += 1
            self.infection_graph.add_node(to_human.name, bin=to_bin, time=timestamp)
            self.infection_graph.add_edge(-1, to_human.name,  timedelta="")

    def summarize_contacts(self):
        x = pd.DataFrame.from_dict(self.contacts['all'])
        x = x[sorted(x.columns)]
        fig = x.iplot(kind='heatmap', asFigure=True, title="all_contacts")
        fig.show()

    def track_trip(self, from_location, to_location, age, hour):
        for i, (l,u) in enumerate(self.age_bins):
            if l <= age < u:
                bin = i

        self.transition_probability[hour][bin][from_location][to_location] += 1
