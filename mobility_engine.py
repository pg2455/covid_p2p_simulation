from typing import List

import random
import datetime
import itertools

import simpy
import networkx as nx

from utils import _draw_random_discreet_gaussian, compute_distance
from config import TICK_MINUTE
import mobility_config as mcfg


class Env(simpy.Environment):
    def __init__(self, initial_timestamp):
        super().__init__()
        self.initial_timestamp = initial_timestamp

    def time(self):
        return self.now

    @property
    def timestamp(self):
        return self.initial_timestamp + datetime.timedelta(
            minutes=self.now * TICK_MINUTE
        )

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


class Location(simpy.Resource):
    def __init__(
        self,
        env,
        capacity=simpy.core.Infinity,
        name="Safeway",
        location_type="stores",
        lat=None,
        lon=None,
        cont_prob=None,
    ):
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
        return (
            f"{self.location_type}:{self.name} - "
            f"Total number of people in {self.location_type}:{len(self.humans)} "
            f"- sick:{self.sick_human()}"
        )

    def contamination_proba(self):
        if not self.sick_human():
            return 0
        return self.cont_prob

    def __hash__(self):
        return hash(self.name)


class Transit(Location):
    def __init__(
        self, env: Env, source: Location, destination: Location, mode: mcfg.MobilityMode
    ):
        self.source = source
        self.destination = destination
        super(Transit, self).__init__(
            env,
            capacity=mode.capacity,
            name=f"{source.name}--({mode.name})-->{destination.name}",
        )


class TransitChain(Location):
    pass


class City(object):
    def __init__(self, env, locations: List[Location]):
        self.env = env
        self.locations = locations
        # Prepare a graph over locations
        self._build_graph()

    def _build_graph(self):
        graph = nx.MultiGraph()
        # Add stores, parks, households as nodes
        graph.add_nodes_from(self.locations)
        # Edges between nodes are annotated by mobility modes
        for source, destination in itertools.product(graph.nodes, graph.nodes):
            if (source, destination) in graph.edges:
                continue
            raw_distance = compute_distance(source, destination)
            for mobility_mode in mcfg.MobilityMode.get_all_mobility_modes():
                mobility_mode: mcfg.MobilityMode
                if mobility_mode.is_compatible_with_distance(distance=raw_distance):
                    graph.add_edge(
                        source, destination, mobility_mode, raw_distance=raw_distance,
                    )
        self.graph = graph

    def plan_trip(self, source: Location, destination: Location) -> List[Transit]:
        # TODO
        pass


if __name__ == "__main__":
    env = Env(datetime.datetime(2020, 2, 28, 0, 0))
    city_limit = ((0, 1000), (0, 1000))
