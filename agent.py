import numpy as np

from config import LOCATION_TECH, P_CAREFUL_PERSON, P_HAS_APP


class Agent:

    def __init__(self, env, name, age, household, infection_timestamp=None):
        self.env = env
        self.name = name
        self.age = age
        self.household = household

        # Age_modifier: single
        self.age_modifier = 1
        if self.age > 40 or self.age < 12:
            self.age_modifier = 2

        # &carefullness
        self.carefullness = round(np.random.normal(75, 10)) if np.random.rand() < P_CAREFUL_PERSON else round(np.random.normal(35, 10))
        self.has_app = np.random.rand() < (P_HAS_APP / self.age_modifier) + (self.carefullness / 2)

        # Recording of events
        self.events = []

        # MOBILITY - to be change over time by Mobility process
        self.location = household

        # HEALTH - to be change over time by Infection process
        # To be changed
        self.health = {}

        # SHARED CONCEPT between infection and mobility
        self.infection_timestamp = infection_timestamp
        self.wearing_mask = False

    @property
    def is_sick(self):
        return self.infection_timestamp is not None

    # TODO: Maybe move to Mobility or Event Class
    @property
    def lat(self):
        return self.location.lat if self.location else self.household.lat

    @property
    def lon(self):
        return self.location.lon if self.location else self.household.lon

    @property
    def obs_lat(self):
        if LOCATION_TECH == 'bluetooth':
            return round(self.lat + np.random.normal(0, 2))
        else:
            return round(self.lat + np.random.normal(0, 10))

    @property
    def obs_lon(self):
        if LOCATION_TECH == 'bluetooth':
            return round(self.lon + np.random.normal(0, 2))
        else:
            return round(self.lon + np.random.normal(0, 10))

    def __repr__(self):
        return f"person:{self.name}, sick:{self.is_sick}"
