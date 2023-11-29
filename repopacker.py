from typing import List, Text, Dict
import numpy as np
from scipy.sparse import csr_matrix
import logging

from content_analyze import ContentAnalyzer
from static_analyze import StaticAnalyzer
from similarity import compute_bm25_similarity
from solver import TSPSolver
from metrics import connection_metrics

logging.basicConfig(level=logging.INFO)


class RepoPacker:
    """
    A class designed to optimize the ordering of repository files to enhance 
    the training of autoregressive Language Models.

    This class utilizes content and static analysis to create a similarity matrix 
    between files and then applies a Traveling Salesperson Problem (TSP) solver 
    to determine the optimal ordering of the files.

    Attributes:
        content_analyzer (ContentAnalyzer): An instance of ContentAnalyzer for analyzing file content.
        static_analyzer (StaticAnalyzer): An instance of StaticAnalyzer for analyzing static connections between files.
        solver (TSPSolver): An instance of TSPSolver for solving the optimal ordering problem.
    """

    content_analyzer = ContentAnalyzer()
    static_analyzer = StaticAnalyzer()
    solver = TSPSolver()

    def __init__(self, repo_files: List[Dict[str, Text]], repo_name='', connection_boost=1.):
        """
        Initializes the RepoPacker with a list of repository files and an optional connection boost factor.

        Args:
            repo_files (List[Dict[str, Text]]): A list of dictionaries, each containing:
                - 'filepath': Path to the file within the repository.
                - 'language': Programming language of the file.
                - 'content': Source code of the file.
            connection_boost (float, optional): A factor to boost the connections in the similarity matrix.
        """
        self.repo_name = repo_name
        self.repo_files = tuple(self.presorting(repo_files))
        self.connection_boost = connection_boost
        self._keywords = []
        self._connections = []
        self._similarity_matrix = None
        self._connections_graph = None
        self._matching_scores = []

    def order_files(self,
                    max_repo_size=10_000,
                    output_matching_score=True,
                    output_connections=False,
                    output_connections_score=False
                    ):
        """
        Orders the repository files based on their semantic and structural relationships to optimize training data.

        Args:
            max_repo_size (int): The maximum size of the repository to process.
            If exceeds returns a lexicographically ordered files. Defaults to 10000.

        Returns:
            List[Dict[str, Text]]: An ordered list of files, each represented as a dictionary.

        Note:
            If an exception occurs during processing, returns lexicographically ordered files.
        """
        matching_scores = [None] * len(self.repo_files)
        coherence_score = None
        ordered_files = self.repo_files

        if 1 < len(self.repo_files) <= max_repo_size:
            try:
                similarity_matrix = self.parse()
                if similarity_matrix is not None:
                    order = self.solver.solve(similarity_matrix)
                    ordered_files = [self.repo_files[idx] for idx in order]

                    if self._similarity_matrix is not None:
                        matching_scores = [self._similarity_matrix[i, j]
                                           for file, i, j in zip(ordered_files, order[:-1], order[1:])]
                        # Last file gets score for backward connection
                        matching_scores.append(self._similarity_matrix[order[-1], order[-2]])
                        coherence_score = np.median(matching_scores)
            except Exception as err:
                logging.error(f"Error in ordering files while parsing {self.repo_name} : {err}")

        output = dict(files=ordered_files)
        if output_matching_score:
            output['matching_scores'] = matching_scores
        if output_connections:
            output['connections'] = self._connections
        if output_connections_score:
            output['coherence_score'] = coherence_score
            output.update(connection_metrics(ordered_files, self._connections))
        return output

    @staticmethod
    def presorting(files):
        """
        Pre-sorts the files based on their file paths.

        Args:
            files (List[Dict[str, Text]]): A list of file dictionaries.

        Returns:
            List[Dict[str, Text]]: The pre-sorted list of file dictionaries.
        """
        for file in files:
            file['path'] = file['path'] if file['path'].startswith('/') else ('/' + file['path'])
        return list(sorted(files, key=lambda f: f['path']))

    def parse(self) -> np.ndarray:
        """
        Parses the repository files to extract keywords and analyzes static connections.

        Returns:
            np.ndarray: A similarity matrix representing the relationships between files.
        """
        similarity_matrix = self.compute_similarity(self.repo_files)
        connections_graph = self.prepare_connections_graph(self.repo_files)
        return self.prepare_similarity_matrix(similarity_matrix, connections_graph)

    def prepare_connections_graph(self, files):
        try:
            self._connections = self.static_analyzer.analyze_connections(files)
            augmented_connections = self.static_analyzer.augment_connections(self._connections)
            self._connections_graph = self.static_analyzer.make_connection_graph(augmented_connections, files)
            return self._connections_graph
        except Exception as err:
            logging.warning(f"Connection analysis failed while parsing {self.repo_name} with error: {err}")
            return None

    def compute_similarity(self, files):
        try:
            if len(files) == 2:
                # Skip content analysis for repository of size == 2
                return np.array([[1., 1.], [1., 1.]])
            else:
                self._keywords = [self.content_analyzer.analyze(file) for file in files]
                self._similarity_matrix = compute_bm25_similarity(self._keywords)
                return self._similarity_matrix
        except Exception as err:
            logging.warning(f"Similarity analysis failed while parsing {self.repo_name} with error: {err}")
            return None

    def prepare_similarity_matrix(self, similarity_matrix=None, connections_graph=None):
        """
        Prepares the BM25 similarity matrix and applies connection boosts if a connections graph is provided.

        Args:
            documents (List[Text]): A list of documents (file contents) for which to compute the similarity matrix.
            connections_graph (np.ndarray, optional): A graph representing static connections between documents.

        Returns:
            np.ndarray: The adjusted similarity matrix.
        """
        if similarity_matrix is None and connections_graph is None:
            return None

        if similarity_matrix is not None:
            if connections_graph is not None:
                similarity_matrix = self.boost_connections(similarity_matrix, connections_graph)
        else:
            similarity_matrix = connections_graph.toarray()

        # Set diagonal to zero
        similarity_matrix = similarity_matrix - np.diag(np.diag(similarity_matrix))

        return similarity_matrix

    def boost_connections(self, similarity_matrix, connections_graph) -> np.ndarray:
        """
        Boosts the connections in the similarity matrix based on the connections graph and the connection boost factor.

        Args:
            similarity_matrix (np.ndarray): The original similarity matrix.
            connections_graph (np.ndarray): A graph representing static connections between documents.

        Returns:
            np.ndarray: The boosted similarity matrix.
        """
        stable_max_value = max(0.01, similarity_matrix.max() \
            if len(similarity_matrix) < 10 else np.percentile(similarity_matrix, 95))

        similarity_matrix = similarity_matrix + self.connection_boost * stable_max_value * connections_graph.toarray()
        return similarity_matrix
