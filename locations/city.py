from typing import TYPE_CHECKING, Mapping, Union, List
from collections import namedtuple
import itertools
from addict import Dict

import numpy as np
import networkx as nx

from locations.city_graphs import ScaleFreeTopology
from locations import location_helpers as lty
from locations.location import Location
from utilities import spatial as spu


if TYPE_CHECKING:
    from base import Env
    from humans.human import ProtoHuman


TransitSpec = namedtuple("TransitSpec", ["transit", "travel_time"])


class City(object):
    def __init__(
        self,
        location_graph: nx.MultiGraph,
        teleport: bool = False,
        verbose: bool = False,
    ):
        self.location_graph = location_graph
        # Turning on the teleporter would make humans not use a transit mode.
        # Useful to turn off the overhead due to transit planning
        self.teleport = teleport
        self.verbose = verbose

    def print(self, message):
        if self.verbose:
            print(message() if callable(message) else message)

    def toggle_teleporter(self, value=None):
        if value is None:
            self.teleport = not self.teleport
        else:
            self.teleport = bool(value)
        return self

    def go(
        self,
        human: "ProtoHuman",
        duration: Union[float, int],
        from_location: Location,
        to_location: Location,
    ):
        if self.teleport:
            self.print(f"{human.name} teleporting to {to_location.name}.")
            yield human.env.process(human.at(to_location, duration=duration))
            return
        # If we're not teleporting, plan a trip.
        mobility_mode_preference = getattr(human, "mobility_mode_preference", None)
        self.print(
            lambda: f"Planning trip from {from_location.name} to "
            f"{to_location.name} (distance = {from_location.distance_to(to_location)})"
            f" for {human.name}."
        )
        trip_plan = self.plan_trip(
            start=from_location,
            stop=to_location,
            mobility_mode_preference=mobility_mode_preference,
        )
        if len(trip_plan) == 0:
            # This shouldn't happen.
            raise RuntimeError("Path not found in graph.")
        for transit, travel_time in trip_plan:
            self.print(
                f"{human.name} is in transit {transit.name} "
                f"({transit.location_type.name})for travel time {travel_time}."
            )
            yield human.env.process(human.at(transit, duration=travel_time))
        yield human.env.process(human.at(to_location, duration=duration))

    def plan_trip(
        self,
        start: Location,
        stop: Location,
        mobility_mode_preference: Mapping[lty.MobilityMode, float] = None,
    ) -> List[TransitSpec]:
        return plan_trip(self.location_graph, start, stop, mobility_mode_preference)

    def sample_location_of_type(
        self,
        location_type: lty.LocationType = None,
        name: str = None,
        config_key: str = None,
        rng: np.random.RandomState = None,
    ):
        # Filter
        if location_type is not None:
            locations = [
                location
                for location in self.location_graph.nodes
                if location.spec.location_type == location_type
            ]
        elif name is not None:
            locations = [
                location
                for location in self.location_graph.nodes
                if location.spec.location_type.name == name
            ]
        elif config_key is not None:
            locations = [
                location
                for location in self.location_graph.nodes
                if location.spec.location_type.config_key == config_key
            ]
        else:
            return None
        rng = np.random if rng is None else rng
        return rng.choice(locations)


def plan_trip(
    location_graph: nx.MultiGraph,
    start: Location,
    stop: Location,
    mobility_mode_preference: Mapping[lty.MobilityMode, float] = None,
) -> List[TransitSpec]:
    assert start in location_graph.nodes
    assert stop in location_graph.nodes
    # This must come from the human
    if mobility_mode_preference is None:
        mobility_mode_preference = lty.DEFAULT_MOBILITY_MODE_PREFERENCE

    travel_distance = start.distance_to(stop)
    # Modulate the mobility_mode_preferences by the travel distance
    mobility_mode_preference = {
        mode: (
            preference * (1 if mode.is_compatible_with_distance(travel_distance) else 0)
        )
        for mode, preference in mobility_mode_preference.items()
    }
    favorite_modes = Dict()

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

        # This is an important component: it couples travel time
        # (i.e. mode speed and distance) and preference.
        mode_weights = {
            mode: d[mode]["travel_time"] / mobility_mode_preference[mode]
            for mode in valid_mobility_modes
        }
        min_weight = min(list(mode_weights.values()))
        favorite_mode = [
            mode for mode, weight in mode_weights.items() if weight == min_weight
        ][0]
        favorite_modes[u][v] = favorite_mode
        return min_weight

    try:
        # Now get that Djikstra path!
        djikstra_path = nx.dijkstra_path(location_graph, start, stop, weight=weight_fn)
    except nx.exception.NetworkXNoPath:
        # No path; destination might have to be resampled
        return []
    # Convert path to transits and return
    transits = []
    for transit_source, transit_destination in zip(djikstra_path, djikstra_path[1:]):
        favorite_transit_mode = favorite_modes[transit_source][transit_destination]
        transit = location_graph[transit_source][transit_destination][
            favorite_transit_mode
        ]["transit_location"]
        travel_time = location_graph[transit_source][transit_destination][
            favorite_transit_mode
        ]["travel_time"]
        transits.append(TransitSpec(transit=transit, travel_time=travel_time))
    return transits


# This method is expensive af, but we'll need it only once so I've
# not bothered with much optimization.
def sample_city(
    env: "Env",
    location_type_distribution: Mapping[lty.LocationType, int] = None,
    geobox: spu.GeoBox = lty.DEFAULT_GEOBOX,
    transit_node_distribution: Mapping[lty.MobilityMode, int] = None,
    rng: np.random.RandomState = np.random,
    **topology_kwargs,
):
    if location_type_distribution is None:
        location_type_distribution = lty.DEFAULT_LOCATION_TYPE_DISTRIBUTION
    num_locations = sum(list(location_type_distribution.values()))

    if transit_node_distribution is None:
        transit_node_distribution = {
            lty.BUS: num_locations // 10,
            lty.SUBWAY: num_locations // 50,
        }
    # Sample coordinates inside the geobox.
    coordinates = geobox.sample(num_samples=num_locations, rng=rng)
    # Generate a graph over the coordinates with a scale-free-ish topology.
    topology = ScaleFreeTopology(rng=rng, **topology_kwargs)
    coordinate_graph = topology.build_graph(coordinates)
    # Sample the transit nodes and connect them. Note that this makes a lot of sense
    # for subways, but less for buses (except if there are dedicated bus roads).
    transit_graphs = {
        mode: topology.extract_hot_tree(topology.extract_hot_nodes(num_hot_nodes=count))
        for mode, count in transit_node_distribution.items()
    }
    # Convert location_type to a range of indices over the list of coordinates
    location_type_to_coordinate_idx = {}
    cursor = 0
    for _location_type in location_type_distribution:
        location_type_to_coordinate_idx[_location_type] = range(
            cursor, cursor + location_type_distribution[_location_type]
        )
        cursor += location_type_distribution[_location_type]
    # Now, convert the coordinates to locations
    locations = []
    for idx, coordinate in enumerate(coordinates):
        for location_type in location_type_to_coordinate_idx:
            if idx in location_type_to_coordinate_idx[location_type]:
                location_spec = lty.LocationSpec(
                    location_type, coordinate, closest_graph_node=coordinate
                )
                locations.append(Location(env, location_spec=location_spec))
                break
        else:
            raise RuntimeError
    # Now, to construct the graph over locations
    location_graph = nx.MultiGraph()
    location_graph.add_nodes_from(locations)
    for coord_u, coord_v in coordinate_graph.edges:
        # Find the indices of u and v, and add an edge over corresponding
        # locations if required
        idx_u = coordinates.index(coord_u)
        idx_v = coordinates.index(coord_v)
        location_u = locations[idx_u]
        location_v = locations[idx_v]
        # Now we want to add a node from u --> v, but skip this if u == v or if an
        # edge v --> u already exists.
        if (
            location_u == location_v
            or (location_v, location_u, lty.WALK) in location_graph.edges
        ):
            continue
        distance = coordinate_graph[coord_u][coord_v]["distance"]
        travel_times = {
            lty.WALK: lty.WALK.compute_travel_time(distance),
            lty.CAR: lty.CAR.compute_travel_time(distance),
        }
        transit_locations = {
            lty.WALK: Location(
                env, location_spec=lty.LocationSpec(lty.WALK, location_size=distance),
            ),
            lty.CAR: Location(
                env, location_spec=lty.LocationSpec(lty.CAR, location_size=distance),
            ),
        }
        # Assume all edges are walkable and driveable.
        location_graph.add_edge(
            location_u,
            location_v,
            lty.WALK,
            distance=distance,
            travel_time=travel_times[lty.WALK],
            transit_location=transit_locations[lty.WALK],
        )
        location_graph.add_edge(
            location_u,
            location_v,
            lty.CAR,
            distance=distance,
            travel_time=travel_times[lty.CAR],
            transit_location=transit_locations[lty.CAR],
        )
        # Connect edges if they are traversable by public transit node
        for mode in transit_graphs:
            mode: lty.MobilityMode
            if (coord_u, coord_v) in transit_graphs[mode].edges:
                # Get the distance
                travel_time = mode.compute_travel_time(distance)
                transit_location = Location(
                    env, location_spec=lty.LocationSpec(mode, location_size=distance)
                )
                location_graph.add_edge(
                    location_u,
                    location_v,
                    mode,
                    distance=distance,
                    travel_time=travel_time,
                    transit_location=transit_location,
                )
    # Now that the locations are in, initialize the city and return
    return City(location_graph)


def test_dijkstra():
    import datetime
    import time
    from base import Env

    env = Env(datetime.datetime(2020, 2, 28, 0, 0))

    print("Sampling city...")
    city = sample_city(
        env, distance_weight=6.0, degree_weight=0.05, num_sampling_steps="auto"
    )

    print(f"Number of edges: {len(city.location_graph.edges)}")

    print("Profiling Dijkstra...")
    tic = time.time()
    for _ in range(100):
        start = np.random.choice(city.location_graph.nodes)
        stop = np.random.choice(city.location_graph.nodes)
        trip = city.plan_trip(start, stop)
    toc = time.time()
    print(f"Per run: {(toc - tic) / 100}")


def test_with_human():
    import datetime
    import time
    from base import Env
    from humans.human import ProtoHuman

    env = Env(datetime.datetime(2020, 2, 28, 0, 0))

    print("Sampling city...")
    city = sample_city(
        env, distance_weight=6.0, degree_weight=0.05, num_sampling_steps="auto"
    )
    city.verbose = True
    print(f"Number of edges: {len(city.location_graph.edges)}")

    human = ProtoHuman(env, "Bob")
    workplace = city.sample_location_of_type(lty.OFFICE)
    home = city.sample_location_of_type(lty.HOUSEHOLD)

    env.process(
        city.toggle_teleporter(False).go(
            human=human, duration=100, from_location=home, to_location=workplace
        )
    )
    env.run(until=200)


if __name__ == "__main__":
    test_dijkstra()
    # test_with_human()
    pass
