# -*- coding: utf-8 -*-

import simpy

import itertools
import datetime

from config import TICK_MINUTE


class Env(simpy.Environment):

    def __init__(self, initial_timestamp):
        super().__init__()
        self.initial_timestamp = initial_timestamp

    def time(self):
        return self.now

    @property
    def timestamp(self):
        return self.initial_timestamp + datetime.timedelta(
            minutes=self.now * TICK_MINUTE)

    def minutes(self):
        return self.timestamp.minute

    def hour_of_day(self):
        return self.timestamp.hour

    def day_of_week(self):
        return self.timestamp.weekday()

    def is_weekend(self):
        return self.day_of_week() in [0, 6]

    def time_of_day(self):
        return self.timestamp.isoformat()


class City(object):

    def __init__(self, stores, parks, humans, miscs, workplaces):
        self.stores = stores
        self.parks = parks
        self.miscs = miscs
        self.workplaces = workplaces
        self.humans = humans

    @property
    def events(self):
        return list(itertools.chain(*[h.events for h in self.humans]))


class Location(simpy.Resource):

    def __init__(self, env, capacity=simpy.core.Infinity, name='Safeway', location_type='stores', lat=None, lon=None,
                 cont_prob=None):
        super().__init__(env, capacity)
        self.humans = set()
        self.name = name
        self.lat = lat
        self.lon = lon
        self.location_type = location_type
        self.cont_prob = cont_prob

    def sick_human(self):
        return any([h.is_sick for h in self.humans])

    def __repr__(self):
        return f"{self.location_type}:{self.name} - Total number of people in {self.location_type}:{len(self.humans)} - sick:{self.sick_human()}"

    def contamination_proba(self):
        if not self.sick_human():
            return 0
        return self.cont_prob

    def __hash__(self):
        return hash(self.name)


