import numpy as np
import networkx as nx
import heapq
import itertools
from collections import namedtuple
from scipy.optimize import linear_sum_assignment
import scipy.sparse

DirectedChain = namedtuple('DirectedChain', ['start', 'end', 'nodes'])

class TSPSolver:
    def solve(self, similarity_matrix):
        """
        Solves the optimization problem to find the maximum weight Hamiltonian path.

        Args:
            similarity_matrix (numpy.ndarray): A square matrix representing the similarities between nodes.

        Returns:
            List: The nodes in the final chain, representing the Hamiltonian path.
        """
        self.init_graph(similarity_matrix)

        components = [self.graph.subgraph(nodes).copy() for nodes in
                      nx.connected_components(self.graph.to_undirected())]
        self.chains = set([self.component_to_directed_chain(component) for component in components])

        final_chain = self.connect_all_chains()

        return final_chain.nodes

    def init_graph(self, similarity_matrix):
        """
        Initializes the graph based on the provided similarity matrix.

        Args:
            similarity_matrix (numpy.ndarray): A square matrix representing the similarities between nodes.

        Raises:
            AssertionError: If the similarity matrix is not square.
        """
        assert similarity_matrix.shape[0] == similarity_matrix.shape[1], "similarity_matrix must be a square matrix"

        cost = -similarity_matrix + np.diag(np.ones(len(similarity_matrix)) * float('inf'))

        row_ind, col_ind = linear_sum_assignment(cost)
        graph = scipy.sparse.csc_matrix((similarity_matrix[row_ind, col_ind], (row_ind, col_ind)))

        self.graph = nx.from_scipy_sparse_array(graph, create_using=nx.DiGraph())
        self.weights = similarity_matrix

    @staticmethod
    def _is_cycle(component):
        """
        Checks if a given graph component is a cycle.

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
        Breaks a cycle in a graph component by removing an edge.

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
        Converts a graph component to a directed chain.

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
            forward_sum += self.weights[edge[0]][edge[1]]
            backward_sum += self.weights[edge[1]][edge[0]]

        start, end, traversal = (start, end, traversal) if forward_sum >= backward_sum else (
            end, start, reversed(traversal))
        return DirectedChain(start, end, tuple(traversal))

    def _connection_cost(self, chain1, chain2):
        """
        Calculates the connection cost between two chains.

        Args:
            chain1 (DirectedChain): The first chain.
            chain2 (DirectedChain): The second chain.

        Returns:
            int: The cost of connecting the end of the first chain to the start of the second chain.
        """
        return -self.weights[chain1.end][chain2.start]

    def initialize_priority_queue(self):
        """
        Initializes a priority queue with all possible chain connections.

        Returns:
            list: A priority queue with tuples containing the cost and chain pairs.
        """
        pq = []
        for chain1, chain2 in itertools.permutations(self.chains, 2):
            cost = self._connection_cost(chain1, chain2)
            heapq.heappush(pq, (cost, chain1, chain2))
        return pq

    def push_new_pair(self, pq, chain1, chain2):
        """
        Pushes a new pair of chains onto the priority queue.

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

        Args:
            pq (list): The priority queue.
            new_chain (DirectedChain): The newly merged chain.

        Returns:
            list: The updated priority queue.
        """
        for chain in self.chains:
            if chain != new_chain:
                pq = self.push_new_pair(pq, new_chain, chain)
        return pq

    def merge_chains(self, chain1, chain2):
        """
        Merges two chains into one.

        Args:
            chain1 (DirectedChain): The first chain.
            chain2 (DirectedChain): The second chain.

        Returns:
            DirectedChain: A new chain resulting from the merging of chain1 and chain2.
        """
        return DirectedChain(chain1.start, chain2.end, nodes=chain1.nodes + chain2.nodes)

    def connect_all_chains(self):
        """
        Connects all chains to form a single Hamiltonian path.

        Returns:
            DirectedChain: The final merged chain representing the Hamiltonian path.
        """
        pq = self.initialize_priority_queue()
        while len(self.chains) > 1:
            _, chain1, chain2 = heapq.heappop(pq)
            while (chain1 not in self.chains or chain2 not in self.chains) and len(pq) > 0:
                _, chain1, chain2 = heapq.heappop(pq)

            merged_chain = self.merge_chains(chain1, chain2)
            self.chains.remove(chain1)
            self.chains.remove(chain2)
            self.chains.add(merged_chain)

            pq = self.update_priority_queue(pq, merged_chain)
        return self.chains.pop()
