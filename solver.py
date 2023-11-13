import numpy as np
import networkx as nx
import heapq
import itertools
from collections import namedtuple
from scipy.optimize import linear_sum_assignment
import scipy.sparse
from typing import List

DirectedChain = namedtuple('DirectedChain', ['start', 'end', 'nodes'])


class TSPSolver:
    max_iter = 10
    num_chains_switch = 64

    def solve(self, similarity_matrix):
        """
        Solves the problem of organizing files in a repository to enhance the training of autoregressive Language Models
        This method initializes the graph, identifies connected components, converts them to directed chains,
        and then connects these chains to form a Hamiltonian path.

        Args:
            similarity_matrix (numpy.ndarray): A square matrix representing the similarities (weights) between nodes.

        Returns:
            List: The nodes in the final chain, representing the Hamiltonian path.
        """
        assert similarity_matrix.shape[0] == similarity_matrix.shape[1], "similarity_matrix must be a square matrix"

        self.similarity_matrix = similarity_matrix
        self._cost_matrix = -similarity_matrix + np.diag(np.ones(len(similarity_matrix)) * float('inf'))

        adj_matrix = self.seed_clustering(similarity_matrix)

        chains = self.extract_chains(adj_matrix)

        for _ in range(self.max_iter):
            if len(chains) <= self.num_chains_switch:
                break
            chains = self.chains_merging(chains)


        final_chain = self.chains_merging_heap(chains)

        return final_chain.nodes

    def extract_chains(self, adj_matrix) -> List[DirectedChain]:
        graph = nx.from_scipy_sparse_array(adj_matrix, create_using=nx.DiGraph())
        components = [graph.subgraph(nodes).copy() for nodes in nx.connected_components(graph.to_undirected())]
        chains = [self.component_to_directed_chain(component) for component in components]
        return chains

    def seed_clustering(self, similarity_matrix):
        """
        Initializes the graph from a similarity matrix by creating a cost matrix and
        applying the Jonker-Volgenant algorithm to find the optimal assignment.
        The resulting graph is sparse with simple chains and cycles.

        Args:
            similarity_matrix (numpy.ndarray): A square matrix representing the similarities between nodes.

        Raises:
            AssertionError: If the similarity matrix is not square.
        """

        row_ind, col_ind = linear_sum_assignment(self._cost_matrix)
        graph = scipy.sparse.csc_matrix((similarity_matrix[row_ind, col_ind], (row_ind, col_ind)))
        return graph

    def chains_merging(self, chains: List[DirectedChain]) -> List[DirectedChain]:
        starts = np.array([chain.start for chain in chains])
        ends = np.array([chain.end for chain in chains])

        for start, end in zip(starts, ends):
            # forbid self-connection
            self._cost_matrix[start, end] = float('inf')
            self._cost_matrix[end, start] = float('inf')

        end_ind, start_ind = linear_sum_assignment(self._cost_matrix[ends, :][:, starts])
        row_ind, col_ind = list(ends[end_ind]), list(starts[start_ind])  # remap

        for chain in chains:
            for _from, _to in zip(chain.nodes[:-1], chain.nodes[1:]):
                row_ind.append(_from)
                col_ind.append(_to)

        graph = scipy.sparse.csc_matrix((self.similarity_matrix[row_ind, col_ind], (row_ind, col_ind)))
        return self.extract_chains(graph)

    @staticmethod
    def _is_cycle(component):
        """
        Determines whether a component of the graph is a cycle.
        A cycle is defined as a component where every node has two edges (degree = 2).

        Args:
            component (networkx.Graph): A subgraph or component of the main graph.

        Returns:
            bool: True if the component is a cycle, False otherwise.
        """
        degree_dict = dict(component.degree())
        return all(degree == 2 for _, degree in degree_dict.items())

    @staticmethod
    def _break_cycle(component):
        """
        Breaks a cycle within a component by removing an edge with the least weight.
        This is a crucial step to transform cycles into chains, simplifying the graph structure.

        Args:
            component (networkx.Graph): A subgraph or component of the main graph that is a cycle.

        Returns:
            networkx.Graph: The modified graph component with one less edge.
        """
        if TSPSolver._is_cycle(component):
            edges = list(component.edges)
            min_idx = np.argmin([component[edge[0]][edge[1]]['weight'] for edge in edges])
            min_edge = edges[min_idx]
            component.remove_edge(*min_edge)
        return component

    def component_to_directed_chain(self, component):
        """
        Converts a graph component into a directed chain. This involves identifying start and end nodes,
        breaking cycles if necessary, and determining the traversal order based on the weights.

        Args:
            component (networkx.Graph): A subgraph or component of the main graph.

        Returns:
            DirectedChain: A namedtuple representing the directed chain with start, end, and nodes attributes.
        """
        component = self._break_cycle(component)

        degree_dict = dict(component.degree())
        start, end = [node for node, degree in degree_dict.items() if degree == 1]

        forward_sum, backward_sum = 0, 0
        traversal = [start]

        for edge in nx.dfs_edges(component.to_undirected(), start):
            traversal.append(edge[1])
            forward_sum += self.similarity_matrix[edge[0]][edge[1]]
            backward_sum += self.similarity_matrix[edge[1]][edge[0]]

        start, end, traversal = (start, end, traversal) if forward_sum >= backward_sum else (
            end, start, reversed(traversal))
        return DirectedChain(start, end, tuple(traversal))

    def _connection_cost(self, chain1, chain2):
        """
        Calculates the connection cost between two chains, defined as the negative weight of the edge
        connecting the end of the first chain to the start of the second. This negative weight is used
        because the algorithm seeks to maximize the total weight in the path.

        Args:
            chain1 (DirectedChain): The first chain.
            chain2 (DirectedChain): The second chain.

    Returns:
            int: The cost of connecting the end of the first chain to the start of the second chain.
        """
        return self._cost_matrix[chain1.end][chain2.start]

    def initialize_priority_queue(self):
        """
        Initializes a priority queue with all possible connections between chains.
        The queue stores tuples of cost and chain pairs, prioritizing connections with the highest weights.

        Returns:
            list: A priority queue with tuples containing the cost and chain pairs.
        """
        pq = []
        for chain1, chain2 in itertools.permutations(self._chains, 2):
            cost = self._connection_cost(chain1, chain2)
            heapq.heappush(pq, (cost, chain1, chain2))
        return pq

    def push_new_pair(self, pq, chain1, chain2):
        """
        Pushes a new pair of chains onto the priority queue, evaluating both directions of connection
        and their respective costs.

        Args:
            pq (list): The priority queue.
            chain1 (DirectedChain): The first chain.
            chain2 (DirectedChain): The second chain.

        Returns:
            list: The updated priority queue.
        """
        cost1 = self._connection_cost(chain1, chain2)
        cost2 = self._connection_cost(chain2, chain1)
        heapq.heappush(pq, (cost1, chain1, chain2))
        heapq.heappush(pq, (cost2, chain2, chain1))
        return pq

    def update_priority_queue(self, pq, new_chain):
        """
        Updates the priority queue with new pairings involving the newly merged chain.
        This is essential to reflect the new potential connections in the graph.

        Args:
            pq (list): The priority queue.
            new_chain (DirectedChain): The newly merged chain.

        Returns:
            list: The updated priority queue.
        """
        for chain in self._chains:
            if chain != new_chain:
                pq = self.push_new_pair(pq, new_chain, chain)
        return pq

    def merge_chains(self, chain1, chain2):
        """
        Merges two chains into one, creating a new chain that extends from the start of the first chain
        to the end of the second. This is a key step in building the final Hamiltonian path.

        Args:
            chain1 (DirectedChain): The first chain.
            chain2 (DirectedChain): The second chain.

        Returns:
            DirectedChain: A new chain resulting from the merging of chain1 and chain2.
        """
        return DirectedChain(chain1.start, chain2.end, nodes=chain1.nodes + chain2.nodes)

    def chains_merging_heap(self, chains):
        """
        Connects all chains in the graph to form a single, continuous Hamiltonian path.
        This method iteratively merges chains, using a priority queue to determine the most advantageous mergers.

        Returns:
            DirectedChain: The final merged chain representing the Hamiltonian path.
        """
        self._chains = set(chains)

        pq = self.initialize_priority_queue()
        while len(self._chains) > 1:
            _, chain1, chain2 = heapq.heappop(pq)
            while (chain1 not in self._chains or chain2 not in self._chains) and len(pq) > 0:
                _, chain1, chain2 = heapq.heappop(pq)

            merged_chain = self.merge_chains(chain1, chain2)
            self._chains.remove(chain1)
            self._chains.remove(chain2)
            self._chains.add(merged_chain)

            pq = self.update_priority_queue(pq, merged_chain)
        return self._chains.pop()
