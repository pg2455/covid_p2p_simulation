import pandas as pd
from collections import defaultdict

def get_nested_dict(nesting):
    if nesting == 1:
        return defaultdict(int)
    elif nesting == 2:
        return defaultdict(lambda : defaultdict(int))
    elif nesting == 3:
        return defaultdict(lambda : defaultdict(lambda : defaultdict(int)))

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

        self.r = []

        self.sar = []

        # demographics
        self.age_distribution = []
        self.households_age = []

        self.age_bin = 15

        self.summarize_population(city)

    def summarize_population(self, city):
        age = [h.age for h in city.humans]
        print(pd.DataFrame(age).describe())

    def track_contact(self, human1, human2, location):
        self.contacts["all"][int(human1.age/self.age_bin)][int(human2.age/self.age_bin)] += 1
        self.contacts["location_all"][location.location_type][int(human1.age/self.age_bin)][int(human2.age/self.age_bin)] += 1

    def track_infection(self, type, from_human, to_human, location):
        if type == "human":
            self.contacts["human_infection"][int(from_human.age/self.age_bin)][int(to_human.age/self.age_bin)] += 1
            self.contacts["location_human_infection"][location.location_type][int(from_human.age/self.age_bin)][int(to_human.age/self.age_bin)] += 1
        else:
            self.contacts["env_infection"][int(to_human.age/self.age_bin)] += 1
            self.contacts["location_env_infection"][location.location_type][int(to_human.age/self.age_bin)] += 1

    def summarize_contacts(self):
        x = pd.DataFrame.from_dict(self.contacts['all'])
        x = x[sorted(x.columns)]
        fig = x.iplot(kind='heatmap', asFigure=True, title="all_contacts")
        fig.show()
