from typing import TYPE_CHECKING, Mapping
import itertools

import numpy as np
import networkx as nx

from locations.city_graphs import ScaleFreeTopology
from locations import location_helpers as lty
from locations.location import Location
from utilities import spatial as spu


if TYPE_CHECKING:
    from base import Env


class City(object):
    def __init__(self, location_graph):
        self.location_graph = location_graph

    def plan_trip(self):
        # TODO
        pass


def sample_city(
    env: "Env",
    location_type_distribution: Mapping[lty.LocationType : int] = None,
    geobox: spu.GeoBox = spu.TUEBINGEN_GEOBOX,
    transit_node_distribution: Mapping[lty.MobilityMode : int] = None,
    rng: np.random.RandomState = np.random,
    **topology_kwargs
):
    if location_type_distribution is None:
        location_type_distribution = {
            # Shelter
            lty.HOUSEHOLD: 300,
            # Workplaces
            lty.OFFICE: 150,
            lty.SCHOOL: 3,
            lty.UNIVERSITY: 2,
            # Necessities
            lty.SUPERMARKET: 5,
            lty.GROCER: 20,
            lty.GYM: 5,
            # Leisure
            lty.PARK: 5,
            lty.DINER: 5,
            lty.BAR: 5,
        }
    num_locations = sum(list(location_type_distribution.values()))

    if transit_node_distribution is None:
        transit_node_distribution = {
            lty.BUS: num_locations // 10,
            lty.SUBWAY: num_locations // 50,
        }
    # Sample coordinates inside the geobox.
    coordinates = geobox.sample(num_samples=location_type_distribution, rng=rng)
    # Generate a graph over the coordinates with a scale-free-ish topology.
    topology = ScaleFreeTopology(rng=rng, **topology_kwargs)
    coordinate_graph = topology.build_graph(coordinates)
    # Sample the transit nodes and connect them. Note that this makes a lot of sense
    # for subways, but less for buses (except if there are dedicated bus roads).
    transit_graphs = {
        mode: topology.extract_hot_nodes(num_hot_nodes=count)
        for mode, count in transit_node_distribution.items()
    }
    # Now, consolidate all graphs to one multi-graph, in 6 steps.
    location_graph = nx.MultiGraph()
    # First, gather the coordinates that we're going to assign to each location_type
    location_type_to_coordinates = {}
    cursor = 0
    for _location_type in location_type_distribution:
        location_type_to_coordinates[_location_type] = coordinates[
            cursor : location_type_distribution[_location_type] + cursor
        ]
        cursor += location_type_distribution[_location_type]
    # Next, make the location specs.
    location_type_to_specs = {
        location_type: [
            lty.LocationSpec(
                location_type=location_type,
                coordinates=coordinate,
                closest_graph_node=coordinate,
            )
            for coordinate in coordinates
        ]
        for location_type, coordinates in location_type_to_coordinates.items()
    }
    # Finally, make the actual locations
    location_type_to_locations = {
        location_type: [
            Location(env, location_spec=location_spec)
            for location_spec in location_specs
        ]
        for location_type, location_specs in location_type_to_specs.items()
    }
    location_graph.add_nodes_from(
        (loc for locs in location_type_to_locations.values() for loc in locs)
    )
    # Build edges
    # Connect locations
    for u, v in itertools.product(location_graph.nodes, location_graph.nodes):
        u: Location
        v: Location
        if u == v or u in location_graph[v]:
            # u = v or the edge v --> u already exists, so nothing to do here.
            continue
        distance = coordinate_graph[u.spec.coordinates][v.spec.coordinates]["distance"]
        # Assumption: all edges can be traversed by cars and pedestrians.
        location_graph.add_edge(u, v, lty.CAR, distance=distance)
        location_graph.add_edge(u, v, lty.WALK, distance=distance)
        # TODO
        pass
