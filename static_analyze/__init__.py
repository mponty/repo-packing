from typing import List, Dict
from scipy.sparse import csr_matrix
from itertools import groupby
from .python_imports import PythonImportsAnalyzer

__all__ = ['StaticAnalyzer']


class StaticAnalyzer:
    analyzer_classes = {
        'Python': PythonImportsAnalyzer,
    }

    def analyze_connections(self, files: List[Dict]) -> csr_matrix:
        connections_graph: csr_matrix = None

        for language, selected_files in self.group_by_language(files):
            if language in self.analyzer_classes:
                analyzer = self.analyzer_classes[language](selected_files)
                connected_files = analyzer.analyze()
                graph = self.make_graph(connected_files, files)
                graph = self.augment_backward_connections(graph, language)
                if connections_graph is None:
                    connections_graph = graph
                else:
                    connections_graph = connections_graph + graph

        return connections_graph

    def group_by_language(self, files: List[Dict]):
        sorted_files = sorted(files, key=lambda f: f['lang'])

        for language, group in groupby(sorted_files, key=lambda f: f['lang']):
            yield language, list(group)

    def make_graph(self, connected_files, files) -> csr_matrix:
        filenames = [f['path'] for f in files]

        row_ind, col_ind, weights = [], [], []
        for edge in connected_files:
            row_ind.append(filenames.index(edge['from_file']))
            col_ind.append(filenames.index(edge['to_file']))
            weights.append(edge['weight'])

        graph = csr_matrix((weights, (row_ind, col_ind)), shape=(len(filenames), len(filenames)))
        return graph

    def augment_backward_connections(self, graph: csr_matrix, language=None) -> csr_matrix:
        if language == 'Python':
            graph = graph + 0.25 * graph.T
        return graph
