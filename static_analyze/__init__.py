from typing import List, Dict
from itertools import groupby, combinations
from scipy.sparse import csr_matrix
from .python_imports import PythonImportsAnalyzer

__all__ = ['StaticAnalyzer']


class StaticAnalyzer:
    analyzer_classes = {
        'Python': PythonImportsAnalyzer,
    }

    def analyze_connections(self, files: List[Dict]) -> List[Dict]:
        connections = []

        for language, selected_files in self.group_by_language(files):
            if language in self.analyzer_classes:
                analyzer = self.analyzer_classes[language](selected_files)
                lang_connections = list(analyzer.analyze())
                for edge in lang_connections:
                    edge['language'] = language
                connections += lang_connections
        return connections

    def group_by_language(self, files: List[Dict]):
        return self.group_by(files, key=lambda f: f['language'])

    @staticmethod
    def group_by(items: List[Dict], key=None):
        sorted_items = sorted(items, key=key)

        for key_value, group in groupby(sorted_items, key=key):
            yield key_value, list(group)

    def make_connection_graph(self, connected_files, files: List[Dict]) -> csr_matrix:
        filenames = [f['path'] for f in files]

        row_ind, col_ind, weights = [], [], []
        for edge in connected_files:
            row_ind.append(filenames.index(edge['from_file']))
            col_ind.append(filenames.index(edge['to_file']))
            weights.append(edge['weight'])

        graph = csr_matrix((weights, (row_ind, col_ind)), shape=(len(filenames), len(filenames)))
        return graph

    def augment_connections(self, connected_files: List[Dict]):
        augmented_connections = self._make_backward_connections(connected_files)
        augmented_connections += self._make_transitive_connections(connected_files)
        augmented_connections = list(self._aggregate_connections(connected_files + augmented_connections))
        return augmented_connections

    def _aggregate_connections(self, connected_files: List[Dict]):
        for (from_file, to_file), group in self.group_by(connected_files, key=lambda f: (f['from_file'], f['to_file'])):
            weight = sum([edge['weight'] for edge in group])
            yield dict(
                from_file=from_file,
                to_file=to_file,
                weight=weight,
                language=group[0]['language']
            )

    def _make_transitive_connections(self, connected_files: List[Dict], weight=0.5):
        connected_files = [edge for edge in connected_files if edge['from_file'] != edge['to_file']]
        transitive = []

        def _make_trans_edge(edge_1, edge_2):
            return dict(
                from_file=edge_1['to_file'],
                to_file=edge_2['to_file'],
                weight=weight,
                language=edge_1['language']
            )

        for from_file, group in self.group_by(connected_files, key=lambda f: f['from_file']):
            for edge_1, edge_2 in combinations(group, 2):
                transitive.append(_make_trans_edge(edge_1, edge_2))
                transitive.append(_make_trans_edge(edge_2, edge_1))

        return transitive

    def _make_backward_connections(self, connected_files: List[Dict], decay=0.25):
        backwards = []
        for edge in connected_files:
            if edge['language'] == 'Python':
                new_edge = dict(
                    from_file=edge['to_file'],
                    to_file=edge['from_file'],
                    weight=decay * edge['weight'],
                    language=edge['language']
                )
                backwards.append(new_edge)
        return backwards
