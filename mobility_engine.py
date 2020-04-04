from typing import List, Mapping

import random
import datetime
import itertools

import simpy
import networkx as nx


from utils import _draw_random_discreet_gaussian, compute_distance, get_random_word
from config import TICK_MINUTE
import mobility_config as mcfg
import mobility_utils as mutl


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
        # FIXME Contamination probability of a location should decay with time since the last
        #  infected human was around, at a rate that depends on the `location_type`.
        if not self.sick_human():
            return 0
        return self.cont_prob

    def __hash__(self):
        return hash(self.name)

    @classmethod
    def random_location(
        cls,
        env: Env,
        city_size: int = 1000,
        capacity: float = simpy.core.Infinity,
        cont_prob: float = None,
    ):
        location = cls(
            env=env,
            capacity=capacity,
            name=get_random_word(),
            lat=random.randint(0, city_size) * mcfg.SPACE_UNIT,
            lon=random.randint(0, city_size) * mcfg.SPACE_UNIT,
            cont_prob=(cont_prob or random.uniform(0, 1)),
            location_type="misc",
        )
        return location


class Transit(Location):
    def __init__(
        self,
        env: Env,
        source: Location,
        destination: Location,
        mobility_mode: mcfg.MobilityMode,
    ):
        self.source = source
        self.destination = destination
        self.mobility_mode = mobility_mode
        super(Transit, self).__init__(
            env,
            capacity=mobility_mode.capacity,
            name=f"{source.name}--({mobility_mode.name})-->{destination.name}",
            location_type="transit",
            # FIXME This should entail counting the number of humans
            cont_prob=mobility_mode.transmission_proba,
        )


class TransitChain(Location):
    def __init__(self, env: Env, stops: List[Transit]):
        pass


class City(object):
    def __init__(self, env: Env, locations: List[Location]):
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
            # To the edges, we're gonna add:
            #   1. Raw distance,
            #   2. A transit object (which is a location)
            raw_distance = mutl.compute_geo_distance(source, destination)
            for mobility_mode in mcfg.MobilityMode.get_all_mobility_modes():
                mobility_mode: mcfg.MobilityMode
                if mobility_mode.is_compatible_with_distance(distance=raw_distance):
                    graph.add_edge(
                        source,
                        destination,
                        mobility_mode,
                        transit=Transit(self.env, source, destination, mobility_mode),
                        raw_distance=raw_distance,
                    )
        self.graph = graph

    def plan_trip(
        self,
        source: Location,
        destination: Location,
        mobility_mode_preference: Mapping[mcfg.MobilityMode, int],
    ) -> List[Transit]:
        # TODO Compute Dijkstra path weighted by preference
        pass


if __name__ == "__main__":
    env = Env(datetime.datetime(2020, 2, 28, 0, 0))
    city = City(env, [Location.random_location(env) for _ in range(100)])
