from typing import TYPE_CHECKING, List, Union
import itertools

import numpy as np
import networkx as nx
from networkx.algorithms.community.asyn_fluid import asyn_fluidc
from addict import Dict

from units import KM
from utilities import spatial as spu
import locations.location_helpers as lty

if TYPE_CHECKING:
    from utilities.spatial import GeoCoordinates
    from locations.location import Location


class Topology(object):
    def build_graph(self, locations):
        raise NotImplementedError


class ScaleFreeTopology(Topology):
    """
    This implements an energy based graph sampling mechanism. There are two knobs to turn:
        1. distance_weight: larger value results in spatially closer nodes being
                            more likely to be connected.
        2. degree_weight: larger value results in nodes with large degree getting preferentially
                          connected.
    """

    DEFAULT_DISTANCE_UNIT = KM

    def __init__(
        self,
        degree_weight: float = 1.0,
        distance_weight: float = 1.0,
        num_sampling_steps: Union[int, str] = "auto",
        mask_connected_edges: bool = True,
        num_samples_per_step: int = 10,
        rng: np.random.RandomState = np.random,
    ):
        self.precomputed_distances = None
        self.adjacency_matrix = None
        self.coordinates = None
        self.graph = None
        self.dijkstra_paths = None
        assert degree_weight >= 0, "Degree weight must be positive."
        assert distance_weight >= 0, "Distance weight must be positive."
        self.degree_weight = degree_weight
        self.distance_weight = distance_weight
        self.num_sampling_steps = num_sampling_steps
        self.mask_connected_edges = mask_connected_edges
        self.num_samples_per_step = num_samples_per_step
        self.rng = rng

    def prepare(self, coordinates: List["GeoCoordinates"]):
        self.coordinates = coordinates
        distances = np.zeros(shape=(len(coordinates),) * 2)
        for (i, crd1), (j, crd2) in itertools.product(
            enumerate(coordinates), enumerate(coordinates)
        ):
            if i == j:
                # This is done to make sure that self-edges
                # are avoided in the later sampling steps.
                distances[i, j] = np.inf
                continue
            # TODO Vectorize this!!
            distances[i, j] = (
                crd1.distance_to(crd2).to(self.DEFAULT_DISTANCE_UNIT).magnitude
            )
        self.precomputed_distances = distances
        self.adjacency_matrix = np.zeros(shape=distances.shape)
        return distances

    def compute_degree_energies(self, weight: float = None):
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

    def compute_distance_energies(self, weight: float = None):
        assert (
            self.precomputed_distances is not None
        ), "Prepare the class first by calling `self.prepare.`"
        # Large distances get lower energy
        distance_energies = (
            weight if weight is not None else self.distance_weight
        ) * self.precomputed_distances
        return distance_energies

    def compute_edge_sampling_probabilities(
        self, degree_weight: float = None, distance_weight: float = None
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

    def sampling_step(
        self,
        num_samples: int = None,
        degree_weight: float = None,
        distance_weight: float = None,
    ):
        num_samples = (
            num_samples if num_samples is not None else self.num_samples_per_step
        )
        edge_sampling_proba = self.compute_edge_sampling_probabilities(
            degree_weight=degree_weight, distance_weight=distance_weight
        )
        raveled_adjacency_matrix = self.adjacency_matrix.ravel()
        # Sample a few indices to activate
        sampled_indices = self.rng.choice(
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

    def graph_from_adjacency_matrix(
        self,
        coordinates: List["GeoCoordinates"] = None,
        adjacency_matrix: np.ndarray = None,
    ):
        coordinates = self.coordinates if coordinates is None else coordinates
        adjacency_matrix = (
            adjacency_matrix if adjacency_matrix is not None else self.adjacency_matrix
        )
        edge_attr_factory = lambda i, j: {
            "distance": self.precomputed_distances[i, j] * self.DEFAULT_DISTANCE_UNIT
        }
        edges = [
            (coordinates[i], coordinates[j], edge_attr_factory(i, j))
            for i, j in zip(*adjacency_matrix.nonzero())
        ]
        graph = nx.Graph()
        graph.add_nodes_from(coordinates)
        graph.add_edges_from(edges)
        return graph

    def all_nodes_connected(self):
        return self.adjacency_matrix.sum(0).min() > 0

    def build_graph(self, coordinates: List["GeoCoordinates"]):
        self.prepare(coordinates)
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
        self.graph = self.graph_from_adjacency_matrix()
        return self.graph

    def compute_pairwise_djikstra_paths(
        self, node_pairs: Union[list, tuple] = None, nodes: list = None
    ):
        weight_fn = lambda u, v, d: self.precomputed_distances[
            self.coordinates.index(u), self.coordinates.index(v)
        ]
        nodes = self.graph.nodes if nodes is None else nodes
        node_pairs = (
            itertools.product(nodes, nodes) if node_pairs is None else node_pairs
        )
        self.dijkstra_paths = Dict()
        for crd1, crd2 in node_pairs:
            if self.dijkstra_paths[crd1][crd2]:
                continue
            elif self.dijkstra_paths[crd2][crd1]:
                # If path from crd2 --> crd1 exists, the reverse path
                # is trivially just the reverse.
                self.dijkstra_paths[crd1][crd2] = reversed(
                    self.dijkstra_paths[crd2][crd1]
                )
            # Find the shortest path
            self.dijkstra_paths[crd1][crd2] = nx.dijkstra_path(
                self.graph, crd1, crd2, weight=weight_fn
            )
        return self.dijkstra_paths

    def get_dijkstra_path(self):
        pass

    def extract_hot_nodes(
        self,
        graph: nx.Graph = None,
        num_hot_nodes: int = 10,
        return_clusters: bool = False,
        seed: int = None,
    ):
        graph = self.graph if graph is None else graph
        clusters = [
            list(cluster) for cluster in asyn_fluidc(graph, k=num_hot_nodes, seed=seed)
        ]
        # Extract hot nodes from clusters as the nodes with the largest in_degrees
        hot_nodes = []
        for cluster in clusters:
            hot_nodes.append(
                cluster[int(np.argmax([graph.degree[n] for n in cluster]))]
            )
        if return_clusters:
            return hot_nodes, clusters
        else:
            return hot_nodes

    def extract_hot_tree(self, hot_nodes: List["GeoCoordinates"]):
        # To extract a tree, we fully connect all nodes and run Kruskal's
        graph = nx.Graph()
        graph.add_nodes_from(hot_nodes)
        for crd1, crd2 in itertools.product(hot_nodes, hot_nodes):
            if crd1 == crd2:
                continue
            i, j = self.coordinates.index(crd1), self.coordinates.index(crd2)
            graph.add_edge(
                crd1,
                crd2,
                distance=self.precomputed_distances[i, j],
            )
        tree = nx.minimum_spanning_tree(graph, weight="distance")
        return tree


def test_topology():
    from base import Env
    import datetime
    import matplotlib.pyplot as plt
    from locations.location import Location
    import locations.location_helpers as lty

    print("Preparing...")
    env = Env(datetime.datetime(2020, 2, 28, 0, 0))
    coordinates = spu.TUEBINGEN_GEOBOX.sample(500)

    locations = [
        Location(env, name=str(i), location_spec=lty.LocationSpec(lty.VOID, coord))
        for i, coord in enumerate(coordinates)
    ]

    print("Sampling...")
    topology = ScaleFreeTopology(
        distance_weight=6.0, degree_weight=0.05, num_sampling_steps="auto"
    )
    graph = topology.build_graph(coordinates)

    print("Extracting communities...")
    summary_nodes, communities = topology.extract_hot_nodes(
        graph, num_hot_nodes=50, return_clusters=True
    )
    summary_graph = topology.extract_hot_tree(summary_nodes)

    pos = {loc: np.array([loc.lat, loc.lon]) for loc in coordinates}

    colors = [
        "red",
        "blue",
        "green",
        "yellow",
        "purple",
        "violet",
        "cyan",
        "magenta",
    ] * 50

    print("Plotting...")
    plt.figure()
    for idx, community in enumerate(communities):
        nx.draw(
            graph,
            pos=pos,
            node_size=20,
            node_color=colors[idx],
            nodelist=list(community),
        )
    plt.savefig(
        "/Users/nrahaman/Python/covid_p2p_simulation_pg2455/plots/graph_all.png"
    )
    plt.close()

    plt.figure()
    nx.draw(
        graph,
        pos=pos,
        nodelist=list(set(graph.nodes) - set(summary_nodes)),
        node_size=20,
        edge_size=2,
    )
    nx.draw(summary_graph, pos=pos, node_color="r", node_size=40, edge_color="r")
    plt.savefig(
        "/Users/nrahaman/Python/covid_p2p_simulation_pg2455/plots/graph_transit.png"
    )
    plt.close()


if __name__ == "__main__":
    test_topology()
