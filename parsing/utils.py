from collections import deque

from nltk import RegexpTokenizer

_tokenizer = RegexpTokenizer('[_a-zA-Z\u0080-\uFFFF]+[0-9]*')


def basic_tokenize(content):
    token_spans = [(content[slice(*span)], span) for span in _tokenizer.span_tokenize(content)]
    return token_spans


def get_node_by_type(nodes, types, recursive=False):
    if isinstance(types, str):
        types = (types,)

    queue = deque(nodes)
    while queue:
        current_node = queue.popleft()
        if current_node.type in types:
            return current_node
        if recursive:
            queue.extend(current_node.children)
