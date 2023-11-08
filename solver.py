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

        self.init_graph(similarity_matrix)

        components = [self.graph.subgraph(nodes).copy() for nodes in
                      nx.connected_components(self.graph.to_undirected())]
        self.chains = set([self.component_to_directed_chain(component) for component in components])

        final_chain = self.connect_all_chains()

        return final_chain.nodes

    def init_graph(self, similarity_matrix):

        assert similarity_matrix.shape[0] == similarity_matrix.shape[1], "similarity_matrix must be a square matrix"

        cost = -similarity_matrix + np.diag(np.ones(len(similarity_matrix)) * float('inf'))

        row_ind, col_ind = linear_sum_assignment(cost)
        graph = scipy.sparse.csc_matrix((similarity_matrix[row_ind, col_ind], (row_ind, col_ind)))

        self.graph = nx.from_scipy_sparse_array(graph, create_using=nx.DiGraph())
        self.weights = similarity_matrix

    @staticmethod
    def _is_cycle(component):
        degree_dict = dict(component.degree())
        return all(degree == 2 for _, degree in degree_dict.items())

    @staticmethod
    def _break_cycle(component):
        if TSPSolver._is_cycle(component):
            edges = list(component.edges)
            min_idx = np.argmin([component[edge[0]][edge[1]]['weight'] for edge in edges])
            min_edge = edges[min_idx]
            component.remove_edge(*min_edge)
        return component

    def component_to_directed_chain(self, component):
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
        return -self.weights[chain1.end][chain2.start]

    def initialize_priority_queue(self):
        pq = []
        for chain1, chain2 in itertools.permutations(self.chains, 2):
            cost = self._connection_cost(chain1, chain2)
            heapq.heappush(pq, (cost, chain1, chain2))
        return pq

    def push_new_pair(self, pq, chain1, chain2):
        cost1 = self._connection_cost(chain1, chain2)
        cost2 = self._connection_cost(chain2, chain1)
        heapq.heappush(pq, (cost1, chain1, chain2))
        heapq.heappush(pq, (cost2, chain2, chain1))
        return pq

    def update_priority_queue(self, pq, new_chain):
        for chain in self.chains:
            if chain != new_chain:
                pq = self.push_new_pair(pq, new_chain, chain)
        return pq

    def merge_chains(self, chain1, chain2):
        return DirectedChain(chain1.start, chain2.end,
                             nodes=chain1.nodes + chain2.nodes)

    def connect_all_chains(self):
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
