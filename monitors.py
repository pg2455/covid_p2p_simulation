from config import TICK_MINUTE
from base import City
from simulator import Human
from matplotlib import pyplot as plt
from collections import Counter
import collections
import json
import pylab as pl
import pickle
import numpy as np

from utils import _json_serialize


class BaseMonitor(object):

    def __init__(self, f=None):
        self.data = []
        self.f = f or 60
        self.s=collections.defaultdict(list)

    def run(self, env, city: City):
        raise NotImplementedError

    def dump(self, dest: str = None):
        pass


class StateMonitor(BaseMonitor):
    def run(self, env, city: City):
        while True:
            d = {
                'time': city.time_of_day(),
                'people': len(city.humans),
                'sick': sum([int(h.is_sick) for h in city.humans]),
            }
            self.data.append(d)
            print(city.clock.time_of_day())
            yield env.timeout(self.f / TICK_MINUTE)

    def dump(self, dest: str = None):
        print(json.dumps(self.data, indent=1))

class SEIRMonitor(BaseMonitor):

    def run(self, env, city: City):

        while True:
            S, E, I, R = 0, 0, 0, 0
            R0 = []
            for h in city.humans:
                S += h.is_susceptible
                E += h.is_exposed
                I += h.is_infectious
                R += h.is_removed
                R0 += h.r0

            self.data.append({
                    'time': env.timestamp,
                    'susceptible': S,
                    'exposed': E,
                    'infectious':I,
                    'removed':R,
                    'R': np.mean(R0) if R0 else -0.01
                    })
            yield env.timeout(self.f / TICK_MINUTE)
            # self.plot()

class InfectionMonitor(BaseMonitor):
    def run(self, env, city:City):
        while True:
            d = {
                'store':0,
                'park':0,
                'workplace':0,
                'misc':0,
                'household':0
            }
            for h in city.humans:
                if len(self.s[h.name])!=0:
                    if (self.s[h.name][0] and h.state[1]) or (self.s[h.name][1] and h.state[2]):
                        d[h.location.location_type]+=1
                    self.s[h.name] = h.state
                else:
                    self.s[h.name] = h.state
            self.data = d if not len(self.data) else Counter(self.data)+Counter(d)        
            yield env.timeout(self.f / TICK_MINUTE)
    
    def avg(self):
        return({k: round(v / total_inf,4) for total_inf in 
        (sum(self.data.values(), 0.0),) for k, v in self.data.items()})

class PlotMonitor(BaseMonitor):

    def run(self, env, city: City):
        fig = plt.figure(figsize=(15, 12))
        while True:
            d = {
                'time': city.clock.time(),
                'htime': city.clock.time_of_day(),
                'sick': sum([int(h.is_sick) for h in city.humans]),
            }
            for k, v in Human.actions.items():
                d[k] = sum(int(h.action == v) for h in city.humans)

            self.data.append(d)
            yield env.timeout(self.f / TICK_MINUTE)
            self.plot()

    def plot(self):
        display.clear_output(wait=True)
        pl.clf()
        time_series = [d['time'] for d in self.data]
        sick_series = [d['sick'] for d in self.data]
        pl.plot(time_series, sick_series, label='sick')
        for k, v in Human.actions.items():
            action_series = [d[k] for d in self.data]
            pl.plot(time_series, action_series, label=k)

        pl.title(f"City at {self.data[-1]['htime']}")
        pl.legend()
        display.display(pl.gcf())

class LatLonMonitor(BaseMonitor):
    def __init__(self, f=None):
        super().__init__(f)
        self.city_data = {}

    def run(self, env, city: City):
        self.city_data['parks'] = [
            {'lat': l.lat,
             'lon': l.lon, } for l in city.parks
        ]
        self.city_data['stores'] = [
            {'lat': l.lat,
             'lon': l.lon, } for l in city.stores
        ]
        fig = plt.figure(figsize=(18, 16))
        while True:
            self.data.extend(
                {'time': city.clock.time_of_day(),
                 'is_sick': h.is_sick,
                 'lat': h.lat(),
                 'lon': h.lon(),
                 'human_id': h.name,
                 'household_id': h.household.name,
                 'location': h.location.name if h.location else None
                 } for h in city.humans
            )
            yield env.timeout(self.f / TICK_MINUTE)
            self.plot()

    def plot(self):
        display.clear_output(wait=True)
        pl.clf()
        # PLOT STORES AND PARKS
        lat_series = [d['lat'] for d in self.city_data['parks']]
        lon_series = [d['lon'] for d in self.city_data['parks']]
        s = 250
        pl.scatter(lat_series, lon_series, s=s, marker='o', color='green', label='parks')

        # PLOT STORES AND PARKS
        lat_series = [d['lat'] for d in self.city_data['stores']]
        lon_series = [d['lon'] for d in self.city_data['stores']]
        s = 50
        pl.scatter(lat_series, lon_series, s=s, marker='o', color='black', label='stores')

        lat_series = [d['lat'] for d in self.data]
        lon_series = [d['lon'] for d in self.data]
        c = ['red' if d['is_sick'] else 'blue' for d in self.data]
        s = 5
        pl.scatter(lat_series, lon_series, s=s, marker='^', color=c, label='human')
        sicks = sum([d['is_sick'] for d in self.data])
        pl.title(f"City at {self.data[-1]['time']} - sick:{sicks}")
        pl.legend()
        display.display(pl.gcf())


class EventMonitor(BaseMonitor):

    def run(self, env, city: City):
        while True:
            self.data = city.events
            yield env.timeout(self.f / TICK_MINUTE)

    def dump(self, dest: str = None):
        if dest is None:
            print(json.dumps(self.data, indent=1, default=_json_serialize))
            return

        with open(f"{dest}.pkl", 'wb') as f:
            pickle.dump(self.data, f)


class TimeMonitor(BaseMonitor):

    def run(self, env, city: City):
        while True:
            print(env.timestamp)
            yield env.timeout(self.f / TICK_MINUTE)
