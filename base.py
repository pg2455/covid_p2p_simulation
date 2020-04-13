import simpy
import math
import datetime
import itertools
import numpy as np


import copy
from config import TICK_MINUTE, MAX_DAYS_CONTAMINATION, LOCATION_DISTRIBUTION, HUMAN_DISTRIBUTION, MIN_AVG_HOUSE_AGE, HOUSE_SIZE_PREFERENCE
from utils import compute_distance, _get_random_area
from collections import defaultdict
from orderedset import OrderedSet

from track import Tracker

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

    def __init__(self, env, n_people, rng, x_range, y_range, start_time, init_percent_sick, Human):
        self.env = env
        self.rng = rng
        self.x_range = x_range
        self.y_range = y_range
        self.total_area = (x_range[1]-x_range[0])*(y_range[1]-y_range[0])
        self.n_people = n_people
        self.start_time = start_time
        self.init_percent_sick = init_percent_sick
        self.initialize_locations()

        self.humans = []
        self.households = OrderedSet()
        self.initialize_humans(Human)

        self._compute_preferences()
        self.tracker = Tracker(env, self)

    def create_location(self, specs, type, name, area=None):
        _cls = Location
        if type in ['household', 'senior_residency']:
            _cls = Household

        return   _cls(
                        env=self.env,
                        rng=self.rng,
                        name=f"{type}:{name}",
                        location_type=type,
                        lat=self.rng.randint(*self.x_range),
                        lon=self.rng.randint(*self.y_range),
                        area=area,
                        social_contact_factor=specs['social_contact_factor'],
                        capacity= None if not specs['rnd_capacity'] else self.rng.randint(*specs['rnd_capacity']),
                        surface_prob = specs['surface_prob']
                        )

    def initialize_locations(self):
        for location, specs in LOCATION_DISTRIBUTION.items():
            if location in ['household']:
                continue
            n = math.ceil(self.n_people/specs["n"])
            area = _get_random_area(n, specs['area'] * self.total_area, self.rng)
            locs = [self.create_location(specs, location, i, area[i]) for i in range(n)]
            setattr(self, f"{location}s", locs)

    def initialize_humans(self, Human):
        # allocate humans to houses such that (unsolved)
        # 1. average number of residents in a house is (approx.) 2.6
        # 2. not all residents are below 15 years of age
        # 3. age occupancy distribution follows HUMAN_DSITRIBUTION.residence_preference.house_size

        # current implementation is an approximate heuristic

        # make humans
        count_humans = 0
        house_allocations = {2:[], 3:[], 4:[], 5:[]}
        n_houses = 0
        for age_bin, specs in HUMAN_DISTRIBUTION.items():
            n = math.ceil(specs['p'] * self.n_people)
            ages = self.rng.randint(*age_bin, size=n)

            senior_residency_preference = specs['residence_preference']['senior_residency']

            professions = ['healthcare', 'school', 'others', 'retired']
            p = [specs['profession_profile'][x] for x in professions]
            profession = self.rng.choice(professions, p=p, size=n)

            for i in range(n):
                count_humans += 1
                age = ages[i]

                # residence
                res = None
                if self.rng.random() < senior_residency_preference:
                    res = self.rng.choice(self.senior_residencys)

                # workplace
                if profession[i] == "healthcare":
                    workplace = self.rng.choice(self.healthcares + self.senior_residencys)
                elif profession[i] == 'school':
                    workplace = self.rng.choice(self.schools)
                elif profession[i] == 'others':
                    workplace = self.rng.choice(self.workplaces)
                else:
                    workplace = res

                self.humans.append(Human(
                        env=self.env,
                        rng=self.rng,
                        name=count_humans,
                        age=age,
                        household=res,
                        workplace=workplace,
                        profession=profession[i],
                        rho=0.0,
                        gamma=0.21,
                        infection_timestamp=self.start_time if self.rng.random() < self.init_percent_sick else None
                        )
                    )

        # assign houses
        # stores tuples - (location, current number of residents, maximum number of residents allowed)
        remaining_houses = []
        for human in self.humans:
            if human.household is not None:
                continue
            if len(remaining_houses) == 0:
                cap = self.rng.choice(range(1,6), p=HOUSE_SIZE_PREFERENCE, size=1)
                x = self.create_location(LOCATION_DISTRIBUTION['household'], 'household', len(self.households))

                remaining_houses.append((x, cap))

            # get_best_match
            res = None
            for  c, (house, n_vacancy) in enumerate(remaining_houses):
                new_avg_age = (human.age + sum(x.age for x in house.residents))/(len(house.residents) + 1)
                if new_avg_age > MIN_AVG_HOUSE_AGE:
                    res = house
                    n_vacancy -= 1
                    if n_vacancy == 0:
                        remaining_houses = remaining_houses[:c] + remaining_houses[c+1:]
                    break

            if res is None:
                for i, (l,u) in enumerate(HUMAN_DISTRIBUTION.keys()):
                    if l <= human.age < u:
                        bin = (l,u)
                        break

                house_size_preference = HUMAN_DISTRIBUTION[(l,u)]['residence_preference']['house_size']
                cap = self.rng.choice(range(1,6), p=house_size_preference, size=1)
                res = self.create_location(LOCATION_DISTRIBUTION['household'], 'household', len(self.households))
                if cap - 1 > 0:
                    remaining_houses.append((res, cap-1))

            # FIXME: there is some circular reference here
            res.residents.append(human)
            human.assign_household(res)
            self.households.add(res)

        # assign area to house
        area = _get_random_area(len(self.households), LOCATION_DISTRIBUTION['household']['area'] * self.total_area, self.rng)
        for i,house in enumerate(self.households):
            house.area = area[i]


    @property
    def events(self):
        return list(itertools.chain(*[h.events for h in self.humans]))

    def _compute_preferences(self):
        """ compute preferred distribution of each human for park, stores, etc."""
        for h in self.humans:
            h.stores_preferences = [(compute_distance(h.household, s) + 1e-1) ** -1 for s in self.stores]
            h.parks_preferences = [(compute_distance(h.household, s) + 1e-1) ** -1 for s in self.parks]


class Location(simpy.Resource):

    def __init__(self, env, rng, area, name, location_type, lat, lon,
            social_contact_factor, capacity, surface_prob):

        if capacity is None:
            capacity = simpy.core.Infinity

        super().__init__(env, capacity)
        self.humans = OrderedSet() #OrderedSet instead of set for determinism when iterating
        self.name = name
        self.rng = rng
        self.lat = lat
        self.lon = lon
        self.area = area
        self.location_type = location_type
        self.social_contact_factor = social_contact_factor
        self.env = env
        self.contamination_timestamp = datetime.datetime.min
        self.contaminated_surface_probability = surface_prob
        self.max_day_contamination = 0

    def infectious_human(self):
        return any([h.is_infectious for h in self.humans])

    def __repr__(self):
        return f"{self.name} - occ:{len(self.humans)}/{self.capacity} - I:{self.infectious_human()}"

    def add_human(self, human):
        self.humans.add(human)
        if human.is_infectious:
            self.contamination_timestamp = self.env.timestamp
            rnd_surface = float(self.rng.choice(a=MAX_DAYS_CONTAMINATION, size=1, p=self.contaminated_surface_probability))
            self.max_day_contamination = max(self.max_day_contamination, rnd_surface)

    def remove_human(self, human):
        self.humans.remove(human)

    @property
    def is_contaminated(self):
        return self.env.timestamp - self.contamination_timestamp <= datetime.timedelta(days=self.max_day_contamination)

    @property
    def contamination_probability(self):
        if self.is_contaminated:
            lag = (self.env.timestamp - self.contamination_timestamp)
            lag /= datetime.timedelta(days=1)
            p_infection = 1 - lag / self.max_day_contamination # linear decay; &envrionmental_contamination
            return self.social_contact_factor * p_infection
        return 0.0

    def __hash__(self):
        return hash(self.name)

    def serialize(self):
        """ This function serializes the location object"""
        s = self.__dict__
        if s.get('env'):
            del s['env']
        if s.get('rng'):
            del s['rng']
        if s.get('_env'):
            del s['_env']
        if s.get('contamination_timestamp'):
            del s['contamination_timestamp']
        return s

class Household(Location):
    def __init__(self, **kwargs):
        super(Household, self).__init__(**kwargs)
        self.residents = []



class Event:
    test = 'test'
    encounter = 'encounter'
    symptom_start = 'symptom_start'
    contamination = 'contamination'
    recovered = 'recovered'

    @staticmethod
    def members():
        return [Event.test, Event.encounter, Event.symptom_start, Event.contamination]

    @staticmethod
    def log_encounter(human1, human2, location, duration, distance, time):
        h_obs_keys = ['obs_lat', 'obs_lon', 'age', 'reported_symptoms', 'test_results', 'has_app']
        h_unobs_keys = ['carefullness', 'viral_load', 'infectiousness', 'symptoms', 'is_exposed', 'is_infectious']
        loc_obs_keys = ['location_type', 'lat', 'lon']
        loc_unobs_keys = ['contamination_probability', 'social_contact_factor']

        obs, unobs = [], []
        for human in [human1, human2]:
            o = {key:getattr(human, key) for key in h_obs_keys}
            obs.append(o)

            u = {key:getattr(human, key) for key in h_unobs_keys}
            u['is_infected'] = human.is_exposed or human.is_infectious
            u['human_id'] = human.name
            u['location_is_residence'] = human.household == location
            unobs.append(u)

        loc_obs = {key:getattr(location, key) for key in loc_obs_keys}
        loc_unobs = {key:getattr(location, key) for key in loc_unobs_keys}
        loc_unobs['location_p_infection'] = location.contamination_probability / location.social_contact_factor
        other_obs = {'duration':duration, 'distance':distance}

        both_have_app = human1.has_app and human2.has_app
        for i, human in [(0, human1), (1, human2)]:
            if both_have_app:
                obs_payload = {**loc_obs, **other_obs, 'human1':obs[i], 'human2':obs[1-i]}
                unobs_payload = {**loc_unobs, 'human1':unobs[i], 'human2':unobs[1-i]}
            else:
                obs_payload = {}
                unobs_payload = { **loc_obs, **loc_unobs, **other_obs, 'human1':{**obs[i], **unobs[i]},
                                    'human2': {**obs[1-i], **unobs[1-i]} }
            unobs_payload.update({'risk': human.risk})

            human.events.append({
                'human_id':human.name,
                'event_type':Event.encounter,
                'time':time,
                'payload':{'observed':obs_payload, 'unobserved':unobs_payload}
            })

    @staticmethod
    def log_test(human, result, time):
        human.events.append(
            {
                'human_id': human.name,
                'event_type': Event.test,
                'time': time,
                'payload': {
                    'observed':{
                        'result': result,
                    },
                    'unobserved':{
                    }

                }
            }
        )

    @staticmethod
    def log_symptom_start(human, covid, time):
        human.events.append(
            {
                'human_id': human.name,
                'event_type': Event.symptom_start,
                'time': time,
                'payload': {
                    'observed':{
                    },
                    'unobserved':{
                        'covid': covid
                    }

                }
            }
        )

    @staticmethod
    def log_exposed(human, time):
        human.events.append(
            {
                'human_id': human.name,
                'event_type': Event.contamination,
                'time': time,
                'payload': {
                    'unobserved':{
                      'exposed': True
                    }

                }
            }
        )

    @staticmethod
    def log_recovery(human, time, death):
        human.events.append(
            {
                'human_id': human.name,
                'event_type': Event.recovered,
                'time': time,
                'payload': {
                    'observed':{
                    },
                    'unobserved':{
                        'recovered': not death,
                        'death': death
                    }
                }
            }
        )
