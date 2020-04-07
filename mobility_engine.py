from typing import List, Mapping
from collections import namedtuple

import random
import datetime
import itertools
import weakref

import simpy
import networkx as nx
from addict import Dict


from utils import _draw_random_discreet_gaussian, compute_distance, get_random_word
from config import TICK_MINUTE
from base import Env
import mobility_config as mcfg
import mobility_utils as mutl
from mobility_topology import NaiveTopology

# This is to prevent circular imports
# noinspection PyUnreachableCode
if False:
    from simulator import Human
else:
    Human = type


class Location(simpy.Resource):
    def __init__(
        self,
        env,
        capacity=simpy.core.Infinity,
        name="Safeway",
        location_type="stores",
        lat=None,
        lon=None,
        area=None,
        cont_prob=None,
    ):
        super().__init__(env, capacity)
        self.humans = set()
        self.name = name
        self.lat = lat
        self.lon = lon
        self.area = area
        self.location_type = location_type
        self.cont_prob = cont_prob

    def infectious_human(self):
        return any([h.is_infectious for h in self.humans])

    def __repr__(self):
        return (
            f"{self.name} - occ:{len(self.humans)}/{self.capacity}"
            f" - I:{self.infectious_human()}"
        )

    @property
    def contamination_probability(self):
        if not self.infectious_human():
            return 0
        return self.cont_prob

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.name == other.name

    @classmethod
    def random_location(
        cls,
        env: Env,
        city_limits: Dict = mcfg.DEFAULT_CITY,
        capacity: float = simpy.core.Infinity,
        cont_prob: float = None,
        location_type: str = "misc",
    ):
        random_geolocation = mutl.sample_in_city(city_spec=city_limits)[0]
        location = cls(
            env=env,
            capacity=capacity,
            name=f"{location_type}_{get_random_word()}",
            lat=random_geolocation.lat,
            lon=random_geolocation.lon,
            cont_prob=(cont_prob or random.uniform(0, 1)),
            location_type=location_type,
        )
        return location


class PublicTransitStation(Location):
    def __init__(
        self,
        env: Env,
        mobility_mode: mcfg.MobilityMode,
        capacity: int = simpy.core.Infinity,
        name: str = get_random_word(6),
        lat: float = None,
        lon: float = None,
        cont_prob: float = None,
    ):
        assert (
            mobility_mode.fixed_routes
        ), f"Mobility mode {mobility_mode} does not require fixed routes."
        name = f"{mobility_mode.name}_stop_{name}"
        super(PublicTransitStation, self).__init__(
            env=env,
            capacity=capacity,
            name=name,
            location_type="public_transit_station",
            lat=lat,
            lon=lon,
            cont_prob=cont_prob,
        )
        self.mobility_mode = mobility_mode

    @classmethod
    def random_station(
        cls,
        env: Env,
        mobility_mode: mcfg.MobilityMode,
        city_limits: Dict = mcfg.DEFAULT_CITY,
        capacity: float = simpy.core.Infinity,
        cont_prob: float = None,
    ):
        random_geolocation = mutl.sample_in_city(city_spec=city_limits)[0]
        transit_station = cls(
            env=env,
            mobility_mode=mobility_mode,
            capacity=capacity,
            name=get_random_word(),
            lat=random_geolocation.lat,
            lon=random_geolocation.lon,
            cont_prob=(cont_prob or random.uniform(0, 1)),
        )
        return transit_station


class Transit(Location):
    def __init__(
        self,
        env: Env,
        source: Location,
        destination: Location,
        mobility_mode: mcfg.MobilityMode,
        num_routes: int = 1,
    ):
        # Privates
        self._travel_time = None
        # Publics
        self.source = source
        self.destination = destination
        self.mobility_mode = mobility_mode
        self.num_routes = num_routes
        super(Transit, self).__init__(
            env,
            capacity=mobility_mode.capacity,
            name=f"{source.name}--({mobility_mode.name})-->{destination.name}",
            location_type="transit",
            # FIXME This should entail counting the number of humans
            cont_prob=mobility_mode.transmission_proba,
        )

    @property
    def travel_time(self):
        if self._travel_time is None:
            self._travel_time = self.mobility_mode.travel_time(
                distance=mutl.compute_geo_distance(self.source, self.destination)
            )
        return self._travel_time


class City(object):
    _all_cities = set()

    def __init__(
        self, env: Env, locations: List[Location], graph: nx.MultiGraph = None
    ):
        # Publics
        self.env = env
        self.locations = locations
        if graph is None:
            # Prepare a random graph over locations
            self.graph = NaiveTopology().build_graph(self.env, self.locations)
        else:
            assert isinstance(
                graph, nx.MultiGraph
            ), "Graph must be an instance of nx.MultiGraph"
            self.graph = graph
        self._all_cities.add(weakref.ref(self))

    def plan_trip(
        self,
        source: Location,
        destination: Location,
        mobility_mode_preference: Mapping[mcfg.MobilityMode, int],
    ) -> List[Transit]:

        if destination == source:
            return []
        assert source in self.locations, f"Trip source {source} is not in city {city}!"
        assert (
            destination in self.locations
        ), f"Trip destination {destination} is not in city {city}!"

        # Keep track of the transit mode with the best score
        favorite_modes = Dict()
        total_distance = mutl.compute_geo_distance(source, destination)

        # The weight function provides a measure of "distance" for Djikstra
        def weight_fn(u, v, d):
            # First case is when the mobility mode is not supported
            valid_mobility_modes = set(d.keys()).intersection(
                set(mobility_mode_preference.keys())
            )
            if not valid_mobility_modes:
                # This means that mobility_mode_preference does not specify
                # a preference for this mode, so we assume that the edge cannot
                # be traversed. Returning None tells networkx just that.
                return None

            # We assume that the preference is a multiplier.
            raw_distance = d[next(iter(d.keys()))]["raw_distance"]

            # This is an important component: it couples favorability,
            # travel time (i.e. mode speed and distance) and preference.
            def mode_weight_fn(mode: mcfg.MobilityMode):
                weight = (
                    (mode.favorability_given_distance(total_distance) + 1)
                    * mode.travel_time(raw_distance).to("minute").magnitude
                    / mobility_mode_preference[mode]
                )
                return weight

            mode_weights = {mode: mode_weight_fn(mode) for mode in valid_mobility_modes}
            min_weight = min(list(mode_weights.values()))
            favorite_mode = [
                mode for mode, weight in mode_weights.items() if weight == min_weight
            ][0]
            favorite_modes[u][v] = favorite_mode
            return min_weight

        try:
            # Now get that Djikstra path!
            djikstra_path = nx.dijkstra_path(
                self.graph, source, destination, weight=weight_fn
            )
        except nx.exception.NetworkXNoPath:
            # No path; destination might have to be resampled
            return []
        # Convert path to transits and return
        transits = []
        for transit_source, transit_destination in zip(
            djikstra_path, djikstra_path[1:]
        ):
            favorite_transit_mode = favorite_modes[transit_source][transit_destination]
            transit = self.graph[transit_source][transit_destination][
                favorite_transit_mode
            ]["transit"]
            transits.append(transit)

        return transits

    def get_location_type(self, location_type):
        return [
            location
            for location in self.locations
            if location.location_type == location_type
        ]

    @property
    def stores(self):
        return self.get_location_type("store")

    @property
    def parks(self):
        return self.get_location_type("park")

    @property
    def miscs(self):
        return self.get_location_type("misc")

    @property
    def humans(self):
        humans = set()
        for location in self.locations:
            humans.update(location.humans)
        return humans

    @property
    def events(self):
        return list(itertools.chain(*[h.events for h in self.humans]))

    def _compute_preferences(self):
        for human in self.humans:
            # noinspection PyProtectedMember
            human._compute_preferences(self)

    def __contains__(self, item):
        from simulator import Human

        if isinstance(item, Human):
            return item.location in self.locations
        elif isinstance(item, Location):
            return item in self.locations
        else:
            raise ValueError(
                f"Cannot check if {type(item)} is contained in city instance."
            )

    @classmethod
    def find_human(cls, human: Human):
        for city in cls.get_all_cities():
            if human in city:
                return city
        return None

    @classmethod
    def make(cls, city_limits=mcfg.DEFAULT_CITY):
        # TODO: Make a semirealistic city with stores, workplaces,
        #  households, etc.
        pass

    @classmethod
    def get_all_cities(cls):
        dead = set()
        for ref in cls._all_cities:
            obj = ref()
            if obj is not None:
                yield obj
            else:
                dead.add(ref)
        cls._all_cities -= dead


class Trip(object):
    """Implements an interface between `Human` and the `City`."""

    def __init__(
        self,
        env: Env,
        human: Human,
        trip_plan: List[Transit] = None,
        source: Location = None,
        destination: Location = None,
        city: City = None,
    ):
        self.env = env
        self.human = human
        if trip_plan is not None:
            self.trip_plan = trip_plan
            self.source = self.trip_plan[0].source
            self.destination = self.trip_plan[-1].destination
        else:
            assert None not in [
                source,
                destination,
            ], "Both source and destination must be provided in order to plan a trip."
            # Find human's city
            city: City = City.find_human(human) if city is None else city
            assert city is not None, "Human not found in a city."
            assert human in city, f"Human {human} not found in city {city}."
            # TODO Add `mobility_mode_preference` as an attribute in humans.
            self.trip_plan = city.plan_trip(
                source, destination, human.mobility_mode_preference
            )
            self.source = source
            self.destination = destination

    def take(self):
        for transit in self.trip_plan:
            with transit.request() as request:
                yield request
                yield self.env.process(self.human.at(transit, transit.travel_time))
        return self


ExcursionType = namedtuple("ExcursionType", ["location", "duration", "request"])


if __name__ == "__main__":
    env = Env(datetime.datetime(2020, 2, 28, 0, 0))
    city = City(env, [Location.random_location(env) for _ in range(100)])
    # noinspection PyTypeChecker
    plan = city.plan_trip(
        source=city.locations[0],
        destination=city.locations[1],
        mobility_mode_preference={mcfg.WALKING: 2.0, mcfg.BUS: 1.0},
    )
