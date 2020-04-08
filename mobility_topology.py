import numpy as np
import networkx as nx
from addict import Dict
import itertools

import mobility_config as mcfg
import mobility_utils as mutl


class Topology(object):
    def build_graph(self, env, locations):
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
    DEFAULT_DISTANCE_UNIT = mcfg.KM

    def __init__(
        self,
        degree_weight=1.0,
        distance_weight=1.0,
        num_sampling_steps="auto",
        mask_connected_edges=True,
        num_samples_per_step=10,
    ):
        self.precomputed_distances = None
        self.adjacency_matrix = None
        assert degree_weight >= 0, "Degree weight must be positive."
        assert distance_weight >= 0, "Distance weight must be positive."
        self.degree_weight = degree_weight
        self.distance_weight = distance_weight
        self.num_sampling_steps = num_sampling_steps
        self.mask_connected_edges = mask_connected_edges
        self.num_samples_per_step = num_samples_per_step

    def prepare(self, locations):
        distances = np.zeros(shape=(len(locations),) * 2)
        for (i, loc1), (j, loc2) in itertools.product(
            enumerate(locations), enumerate(locations)
        ):
            if i == j:
                # This is done to make sure that self-edges
                # are avoided in the later sampling steps.
                distances[i, j] = np.inf
                continue
            # TODO Vectorize this!!
            distances[i, j] = (
                mutl.compute_geo_distance(loc1, loc2)
                .to(self.DEFAULT_DISTANCE_UNIT)
                .magnitude
            )
        self.precomputed_distances = distances
        self.adjacency_matrix = np.zeros(shape=distances.shape)
        return distances

    def compute_degree_energies(self, weight=None):
        assert (
            self.adjacency_matrix is not None
        ), "Prepare the class first by calling `self.prepare`."
        # Get the total degree of u's and v's of all edges
        edge_degrees = (
            self.adjacency_matrix.sum(0, keepdims=True)
            + self.adjacency_matrix.sum(1, keepdims=True)
        ) / 2
        # Edges between nodes of large degree gets lower energy.
        degree_energy = (
            -(weight if weight is not None else self.degree_weight) * edge_degrees
        )
        return degree_energy

    def compute_distance_energies(self, weight=None):
        assert (
            self.precomputed_distances is not None
        ), "Prepare the class first by calling `self.prepare.`"
        # Large distances get lower energy
        distance_energies = (
            weight if weight is not None else self.distance_weight
        ) * self.precomputed_distances
        return distance_energies

    def compute_edge_sampling_probabilities(
        self, degree_weight=None, distance_weight=None
    ):
        assert (
            self.precomputed_distances is not None
        ), "Precompute distances first by calling self.prepare with locations."
        degree_energy = self.compute_degree_energies(degree_weight)
        distance_energies = self.compute_distance_energies(distance_weight)
        energy = degree_energy + distance_energies
        # Mask out edges that are already connected
        if self.mask_connected_edges:
            energy[self.adjacency_matrix > 0] = np.inf
        # Compute a Boltzmann distry over all edges
        unnormalized = np.exp(-energy)
        normalized = unnormalized / unnormalized.sum()
        return normalized

    def sampling_step(self, num_samples=None, degree_weight=None, distance_weight=None):
        num_samples = (
            num_samples if num_samples is not None else self.num_samples_per_step
        )
        edge_sampling_proba = self.compute_edge_sampling_probabilities(
            degree_weight=degree_weight, distance_weight=distance_weight
        )
        raveled_adjacency_matrix = self.adjacency_matrix.ravel()
        # Sample a few indices to activate
        sampled_indices = np.random.choice(
            np.arange(edge_sampling_proba.size),
            p=edge_sampling_proba.ravel(),
            size=(num_samples,),
            replace=True,
        )
        # Add sampled edges to the adjacency matrix. Note that this
        # operation is in-place w.r.t. self.adjacency_matrix
        raveled_adjacency_matrix[list(sampled_indices)] = 1
        # Now we make the new matrix symmetric (because the graph is undirected)
        self.adjacency_matrix = np.clip(
            self.adjacency_matrix + self.adjacency_matrix.T, a_max=1.0, a_min=0.0
        )

    def graph_from_adjacency_matrix(self, locations, adjacency_matrix=None):
        adjacency_matrix = (
            adjacency_matrix if adjacency_matrix is not None else self.adjacency_matrix
        )
        edges = [
            (locations[i], locations[j], mcfg.ANY)
            for i, j in zip(*adjacency_matrix.nonzero())
        ]
        graph = nx.MultiGraph()
        graph.add_nodes_from(locations)
        graph.add_edges_from(edges)
        return graph

    def all_nodes_connected(self):
        return self.adjacency_matrix.sum(0).min() > 0

    def build_graph(self, env, locations):
        self.prepare(locations)
        # Connect the locations
        step = 0
        while True:
            step += 1
            if self.num_sampling_steps == "auto":
                # Check if all nodes connected
                if self.all_nodes_connected():
                    break
            else:
                assert isinstance(self.num_sampling_steps, int)
                # Break if number of allowed steps is exceeded
                if step > self.num_sampling_steps:
                    break
            # ... or if all nodes are
            self.sampling_step()
        graph = self.graph_from_adjacency_matrix(locations)
        return graph


if __name__ == "__main__":
    from base import Env
    import datetime
    import matplotlib.pyplot as plt

    env = Env(datetime.datetime(2020, 2, 28, 0, 0))
    locations = mutl.sample_in_city(mcfg.CITIES.REGION_MONTREAL, 100)
    graph = ScaleFreeTopology(
        distance_weight=1.1, degree_weight=0.3, num_sampling_steps="auto"
    ).build_graph(env, locations)
    pos = {loc: np.array([loc.lat, loc.lon]) for loc in locations}

    plt.figure()
    nx.draw(graph, pos=pos)
    plt.savefig("/Users/nrahaman/Python/covid_p2p_simulation/scratch/graph.png")
    plt.close()

