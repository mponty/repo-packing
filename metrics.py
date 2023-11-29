from typing import List, Dict


def connection_metrics(ordered_files: List[Dict], connections: List[Dict]):
    ordered_paths = [file['path'] for file in ordered_files]
    n_connections = len(connections)
    n_files = len(ordered_paths)

    if n_files == 1:
        return dict(
            n_files=n_files,
            n_connections=n_connections,
            jaccard=None,
            recall=None,
            connection_score=None
        )

    connections = [(edge['from_file'].lstrip('/'), edge['to_file'].lstrip('/')) for edge in connections]
    order_connections = [(from_file.lstrip('/'), to_file.lstrip('/'))
                         for from_file, to_file in zip(ordered_paths[:-1], ordered_paths[1:])]

    intersection = set(connections) & set(order_connections)
    union = set(connections) | set(order_connections)

    return dict(
        n_files=n_files,
        n_connections=n_connections,
        jaccard=len(intersection) / len(union),
        recall=len(intersection) / (n_files - 1),
        connection_score=None if n_connections == 0 else \
            max(len(intersection) / (n_files - 1), len(intersection) / n_connections)
    )
