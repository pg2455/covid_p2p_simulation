import random
from collections import defaultdict

import numpy as np

from agent import Agent
from config import AVERAGE_SHOP_TIME_MINUTES, SCALE_SHOP_TIME_MINUTES, AVG_SCALE_SHOP_TIME_MINUTES, \
    SCALE_SCALE_SHOP_TIME_MINUTES, AVG_EXERCISE_MINUTES, SCALE_EXERCISE_MINUTES, SCALE_SCALE_EXERCISE_MINUTES, \
    AVG_SCALE_EXERCISE_MINUTES, AVG_WORKING_HOURS, SCALE_WORKING_HOURS, AVG_SCALE_WORKING_HOURS, \
    SCALE_SCALE_WORKING_HOURS, AVG_SCALE_MISC_MINUTES, SCALE_SCALE_MISC_MINUTES, SCALE_MISC_MINUTES, AVG_MISC_MINUTES, \
    TICK_MINUTE, WORK_FROM_HOME
from event import Event
from utils import _draw_random_discreet_gaussian, _normalize_scores
from world import City


class Visits:
    parks = defaultdict(int)
    stores = defaultdict(int)
    miscs = defaultdict(int)

    @property
    def n_parks(self):
        return len(self.parks)

    @property
    def n_stores(self):
        return len(self.stores)

    @property
    def n_miscs(self):
        return len(self.miscs)


class HumanMobility:

    actions = {
        'shopping': 1,
        'at_home': 3,
        'exercise': 4
    }

    def __init__(self, agent: Agent, city: City):
        self.human = agent
        self.env = agent.env
        self.city = city

        self.workplace = random.choice(city.workplaces)
        # ########## MOBILITY ATTRIBUTE ########## #
        self.visits = Visits()

        # Used to simulate random travels in the city
        self.rho = 0.3
        self.gamma = 0.21

        self.avg_shopping_time = _draw_random_discreet_gaussian(AVERAGE_SHOP_TIME_MINUTES, SCALE_SHOP_TIME_MINUTES)
        self.scale_shopping_time = _draw_random_discreet_gaussian(AVG_SCALE_SHOP_TIME_MINUTES,
                                                                   SCALE_SCALE_SHOP_TIME_MINUTES)

        self.avg_exercise_time = _draw_random_discreet_gaussian(AVG_EXERCISE_MINUTES, SCALE_EXERCISE_MINUTES)
        self.scale_exercise_time = _draw_random_discreet_gaussian(AVG_SCALE_EXERCISE_MINUTES,
                                                                   SCALE_SCALE_EXERCISE_MINUTES)

        self.avg_working_hours = _draw_random_discreet_gaussian(AVG_WORKING_HOURS, SCALE_WORKING_HOURS)
        self.scale_working_hours = _draw_random_discreet_gaussian(AVG_SCALE_WORKING_HOURS, SCALE_SCALE_WORKING_HOURS)

        self.avg_misc_time = _draw_random_discreet_gaussian(AVG_MISC_MINUTES, SCALE_MISC_MINUTES)
        self.scale_misc_time = _draw_random_discreet_gaussian(AVG_SCALE_MISC_MINUTES, SCALE_SCALE_MISC_MINUTES)

        # TODO: multiple possible days and times & limit these activities in a week
        self.shopping_days = np.random.choice(range(7))
        self.shopping_hours = np.random.choice(range(7, 20))

        self.exercise_days = np.random.choice(range(7))
        self.exercise_hours = np.random.choice(range(7, 20))

        self.work_start_hour = np.random.choice(range(7, 12))

        self.stores_preferences = [(compute_distance(agent.household, s) + 1e-1) ** -1 for s in city.stores]
        self.parks_preferences = [(compute_distance(agent.household, s) + 1e-1) ** -1 for s in city.parks]

    def run(self):
        """
            Process defining mobility pattern for a given human
        """
        self.human.household.humans.add(self.human)
        while True:

            if self.env.hour_of_day() == self.work_start_hour and not self.env.is_weekend() and not WORK_FROM_HOME:
                yield self.env.process(self.go_to_work())
            elif self.env.hour_of_day() == self.shopping_hours and self.env.day_of_week() == self.shopping_days:
                yield self.env.process(self.shop())
            elif self.env.hour_of_day() == self.exercise_hours and self.env.day_of_week() == self.exercise_days:  ##LIMIT AND VARIABLE
                yield self.env.process(self.exercise())
            elif np.random.random() < 0.05 and self.env.is_weekend():
                yield self.env.process(self.take_a_trip())
            self.human.location = self.human.household
            yield self.env.process(self.stay_at_home())

    def stay_at_home(self):
        self.action = self.actions['at_home']
        yield self.env.process(self.at(self.human.household, 60))

    def go_to_work(self):
        t = _draw_random_discreet_gaussian(self.avg_working_hours, self.scale_working_hours)
        yield self.env.process(self.at(self.workplace, t))

    def take_a_trip(self):
        S = 0
        p_exp = 1.0
        while True:
            if np.random.random() > p_exp:  # return home
                yield self.env.process(self.at(self.human.household, 60))
                break

            loc = self._select_location(location_type='miscs')
            S += 1
            p_exp = self.rho * S ** (-self.gamma * self.adjust_gamma)
            with loc.request() as request:
                yield request
                t = _draw_random_discreet_gaussian(self.avg_misc_time, self.scale_misc_time)
                yield self.env.process(self.at(loc, t))

    def shop(self):
        self.action = self.actions['shopping']
        grocery_store = self._select_location(location_type="stores")  ## MAKE IT EPR

        with grocery_store.request() as request:
            yield request
            t = _draw_random_discreet_gaussian(self.avg_shopping_time, self.scale_shopping_time)
            yield self.env.process(self.at(grocery_store, t))

    def exercise(self):
        self.action = self.actions['exercise']
        park = self._select_location(location_type="park")
        t = _draw_random_discreet_gaussian(self.avg_shopping_time, self.scale_shopping_time)
        yield self.env.process(self.at(park, t))

    def _select_location(self, location_type):
        """
        Preferential exploration treatment of visiting places
        rho, gamma are treated in the paper for normal trips
        Here gamma is multiplied by a factor to supress exploration for parks, stores.
        """
        if location_type == "park":
            S = self.visits.n_parks
            self.adjust_gamma = 1.0
            pool_pref = self.parks_preferences
            locs = self.city.parks
            visited_locs = self.visits.parks

        elif location_type == "stores":
            S = self.visits.n_stores
            self.adjust_gamma = 1.0
            pool_pref = self.stores_preferences
            locs = self.city.stores
            visited_locs = self.visits.stores

        elif location_type == "miscs":
            S = self.visits.n_miscs
            self.adjust_gamma = 1.0
            pool_pref = [(compute_distance(self.human.location, m) + 1e-1) ** -1 for m in self.city.miscs if
                         m != self.human.location]
            pool_locs = [m for m in self.city.miscs if m != self.human.location]
            locs = self.city.miscs
            visited_locs = self.visits.miscs

        else:
            raise ValueError(f'Unknown location_type:{location_type}')

        if S == 0:
            p_exp = 1.0
        else:
            p_exp = self.rho * S ** (-self.gamma * self.adjust_gamma)

        if np.random.random() < p_exp and S != len(locs):
            # explore
            cands = [i for i in locs if i not in visited_locs]
            cands = [(loc, pool_pref[i]) for i, loc in enumerate(cands)]
        else:
            # exploit
            cands = [(i, count) for i, count in visited_locs.items()]

        cands, scores = zip(*cands)
        loc = np.random.choice(cands, p=_normalize_scores(scores))
        visited_locs[loc] += 1
        return loc

    def at(self, location, duration):
        self.human.location = location
        location.humans.add(self.human)

        # Update if someone is wearing a mask
        if not self.human.location == self.human.household:
            self.human.mask = np.random.rand() < self.human.carefullness

        # Current hack to capture some notion of time
        self.human.leaving_time = duration + self.env.now
        self.human.start_time = self.env.now

        # Report all the encounters
        for h in location.humans:
            if h == self or location.location_type == 'household':
                continue
            Event.log_encounter(self.human, h,
                                location=location,
                                duration=min(self.human.leaving_time, h.leaving_time) - max(self.human.start_time, h.start_time),
                                distance=np.random.randint(50, 1000),
                                # cm  #TODO: prop to Area and inv. prop to capacity
                                time=self.env.timestamp,
                                )

        if not self.human.is_sick:
            if random.random() < location.contamination_proba():
                # TODO - move function to human for contamination
                self.human.infection_timestamp = self.env.timestamp
                Event.log_contaminate(self.human, self.env.timestamp)
        yield self.env.timeout(duration / TICK_MINUTE)

        # Remove the mask
        self.human.mask = False
        location.humans.remove(self.human)


def compute_distance(loc1, loc2):
    return np.sqrt((loc1.lat - loc2.lat) ** 2 + (loc1.lon - loc2.lon) ** 2)
