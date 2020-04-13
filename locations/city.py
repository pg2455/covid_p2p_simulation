import numpy as np

from locations.city_graphs import ScaleFreeTopology
from locations import location_helpers as lty
from utilities import spatial as spu


class City(object):
    def __init__(self, location_graph):
        self.location_graph = location_graph

    def plan_trip(self):
        # TODO
        pass

    @classmethod
    def sample(
        cls,
        num_locations: int = 500,
        geobox: spu.GeoBox = spu.TUEBINGEN_GEOBOX,
        num_transit_nodes: int = 50,
        rng: np.random.RandomState = np.random,
        **topology_kwargs
    ):
        # Sample coordinates inside the geobox.
        coordinates = geobox.sample(num_samples=num_locations, rng=rng)
        # Generate a graph over the coordinates with a scale-free-ish topology.
        topology = ScaleFreeTopology(rng=rng, **topology_kwargs)
        graph = topology.build_graph(coordinates)
        # Sample the transit nodes and connect them
        transit_coordinates = topology.extract_hot_nodes(
            num_hot_nodes=num_transit_nodes
        )
        transit_graph = topology.extract_hot_tree(transit_coordinates)
        # TODO Continue

        pass
