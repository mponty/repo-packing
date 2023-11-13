from typing import List, Text, Dict
import numpy as np

from content_analyze import ContentAnalyzer
from static_analyze import StaticAnalyzer
from similarity import compute_bm25_similarity
from solver import TSPSolver


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

    def __init__(self, repo_files: List[Dict[str, Text]], connection_boost=1.):
        """
        Initializes the RepoPacker with a list of repository files and an optional connection boost factor.

        Args:
            repo_files (List[Dict[str, Text]]): A list of dictionaries, each containing:
                - 'filepath': Path to the file within the repository.
                - 'language': Programming language of the file.
                - 'content': Source code of the file.
            connection_boost (float, optional): A factor to boost the connections in the similarity matrix.
        """

        self.repo_files = tuple(self.presorting(repo_files))
        self.connection_boost = connection_boost
        self._keywords = None
        self._similarity_matrix = None
        self._connections_graph = None

        for file in self.repo_files:
            file['matching_score'] = 0.

    def order_files(self, max_repo_size=10_000):
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
        if len(self.repo_files) < 2 or len(self.repo_files) > max_repo_size:
            return self.repo_files

        try:
            similarity_matrix = self.parse()
            order = self.solver.solve(similarity_matrix)
            ordered_files = [self.repo_files[idx] for idx in order]
            for file, i, j in zip(ordered_files, order[:-1], order[1:]):
                file['matching_score'] = self._similarity_matrix[i, j]

            # Last file gets score for backward connection
            ordered_files[-1]['matching_score'] = self._similarity_matrix[order[-1], order[-2]]
        except Exception as err:
            # TODO : logger
            # print(type(err), err)
            ordered_files = self.repo_files
        return ordered_files

    @staticmethod
    def presorting(files):
        """
        Pre-sorts the files based on their file paths.

        Args:
            files (List[Dict[str, Text]]): A list of file dictionaries.

        Returns:
            List[Dict[str, Text]]: The pre-sorted list of file dictionaries.
        """
        return list(sorted(files, key=lambda f: f['path']))

    def parse(self) -> np.ndarray:
        """
        Parses the repository files to extract keywords and analyzes static connections.

        Returns:
            np.ndarray: A similarity matrix representing the relationships between files.
        """
        self._keywords = [self.content_analyzer.analyze(file['content'], file['language']) for file in self.repo_files]
        self._connections_graph = self.static_analyzer.analyze_connections(self.repo_files)

        return self.prepare_similarity_matrix(self._keywords, self._connections_graph)

    def prepare_similarity_matrix(self, documents, connections_graph=None):
        """
        Prepares the BM25 similarity matrix and applies connection boosts if a connections graph is provided.

        Args:
            documents (List[Text]): A list of documents (file contents) for which to compute the similarity matrix.
            connections_graph (np.ndarray, optional): A graph representing static connections between documents.

        Returns:
            np.ndarray: The adjusted similarity matrix.
        """
        self._similarity_matrix = similarity_matrix = compute_bm25_similarity(documents)
        if connections_graph is not None:
            similarity_matrix = self.boost_connections(similarity_matrix, connections_graph)

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
        max_value = similarity_matrix.max()
        similarity_matrix = similarity_matrix + self.connection_boost * max_value * connections_graph.toarray()
        return similarity_matrix
