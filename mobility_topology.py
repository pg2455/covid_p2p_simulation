import networkx as nx
from addict import Dict
import itertools

import mobility_config as mcfg
import mobility_utils as mutl


class Topology(object):
    def build_graph(self, env, location):
        raise NotImplementedError


class NaiveTopology(Topology):
    def build_graph(self, env, locations):
        from mobility_engine import Transit, PublicTransitStation

        graph = nx.MultiGraph()
        # Add stores, parks, households as nodes
        graph.add_nodes_from(locations)
        # To connect nodes and build the graph, we'll do the following.
        #   1. We'll try to connect all nodes with each other with the
        #      non-fixed-route mobility modes, if the distance is ok.
        #   2. We'll connect the public transport nodes (with the
        #      corresponding edges).
        # Record of the distances (which we'll need later)
        distances = Dict()
        # Edges between nodes are annotated by mobility modes
        for source, destination in itertools.product(graph.nodes, graph.nodes):
            if (source, destination) in graph.edges:
                continue
            # No self edges
            if source == destination:
                continue
            # If the geo distance between two nodes can be supported by an
            # available mobility mode without a fixed route, we add it as
            # a labelled edge in the multi-graph. Moreover, we label the edges
            # with the following:
            #   1. Raw distance,
            #   2. A transit object (which is a location)
            distances[source][destination] = raw_distance = mutl.compute_geo_distance(
                source, destination
            )
            for mobility_mode in mcfg.MobilityMode.get_all_mobility_modes():
                mobility_mode: mcfg.MobilityMode
                if (
                    mobility_mode.is_compatible_with_distance(distance=raw_distance)
                    and not mobility_mode.fixed_routes
                ):
                    graph.add_edge(
                        source,
                        destination,
                        mobility_mode,
                        transit=Transit(env, source, destination, mobility_mode),
                        raw_distance=raw_distance,
                    )
        # Connect the public transits
        public_transit_stations = [
            node for node in graph.nodes if isinstance(node, PublicTransitStation)
        ]
        for source in public_transit_stations:
            # Find the closest destination that compatible and not connected
            closest_destination = None
            distance_to_closest_destination = 10e10 * mcfg.KM
            for destination in public_transit_stations:
                if source == destination:
                    continue
                if source.mobility_mode != destination.mobility_mode:
                    # Source and destination are of two different modes.
                    continue
                mobility_mode = source.mobility_mode
                if (source, destination) in graph.edges and mobility_mode in graph[
                    source
                ][destination]:
                    # Source and destination are already connected by their respective
                    # mobility mode, nothing to do here.
                    continue
                if not mobility_mode.is_compatible_with_distance(
                    distances[source][destination]
                ):
                    # Destination too far away
                    continue
                # Compute distance between destinations
                if distances[source][destination] < distance_to_closest_destination:
                    distance_to_closest_destination = distances[source][destination]
                    closest_destination = destination
            if closest_destination is None:
                # No good destination found :(
                continue
            # Found the best destination, now we need to connect them.
            graph.add_edge(
                source,
                closest_destination,
                source.mobility_mode,
                transit=Transit(env, source, closest_destination, source.mobility_mode),
                raw_distance=distances[source][closest_destination],
            )
        return graph


class ScaleFreeTopology(Topology):
    def __init__(self, num_closest):
        pass

    def build_graph(self, env, location):
        pass

    pass
