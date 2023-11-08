from typing import List, Text, Dict
import numpy as np

from content_analyze import ContentAnalyzer
from static_analyze import StaticAnalyzer
from similarity import compute_bm25_similarity
from solver import TSPSolver


class Repo:
    content_analyzer = ContentAnalyzer()
    static_analyzer = StaticAnalyzer()
    solver = TSPSolver()

    def __init__(self, repo_files: List[Dict[str, Text]], connection_boost=1.):
        """

        :param repo_files: List of dictionaries with keys:ss
            - 'filepath' : path to the file inside of the repository
            - 'lang' : programming language of the file
            - 'content' : source code of the file
        """
        self.repo_files = tuple(self.presorting(repo_files))
        self.connection_boost = connection_boost
        self._keywords = None
        self._similarity_matrix = None
        self._connections_graph = None

    def order_files(self):
        if len(self.repo_files) < 2:
            return self.repo_files
        try:
            similarity_matrix = self.parse()
            order = self.solver.solve(similarity_matrix)
            ordered_files = [self.repo_files[idx] for idx in order]
        except Exception as err:
            # TODO : logger
            # print(type(err), err)
            ordered_files = self.repo_files
        return ordered_files

    @staticmethod
    def presorting(files):
        return list(sorted(files, key=lambda f: f['path']))

    def parse(self) -> np.ndarray:
        self._keywords = [self.content_analyzer.analyze(file['content'], file['language']) for file in self.repo_files]
        self._connections_graph = self.static_analyzer.analyze_connections(self.repo_files)

        return self.prepare_similarity_matrix(self._keywords, self._connections_graph)

    def prepare_similarity_matrix(self, documents, connections_graph=None):
        self._similarity_matrix = similarity_matrix = compute_bm25_similarity(documents)
        if connections_graph is not None:
            similarity_matrix = self.boost_connections(similarity_matrix, connections_graph)
        # Set diagonal to zero
        similarity_matrix = similarity_matrix - np.diag(np.diag(similarity_matrix))
        return similarity_matrix

    def boost_connections(self, similarity_matrix, connections_graph) -> np.ndarray:
        max_value = similarity_matrix.max()
        similarity_matrix = similarity_matrix + self.connection_boost * max_value * connections_graph.toarray()
        return similarity_matrix
