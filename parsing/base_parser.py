from tree_sitter import Node, Parser
from analysis.core.document import Document
from parsing.utils import basic_tokenize


class DefaultParser:
    def parse(self, document: Document) -> Document:
        document.identifiers = basic_tokenize(document.content)
        return document


class CodeParser:
    tree_sitter_parser: Parser = None
    string_types = ['comment', 'line_comment', 'block_comment', 'string', 'string_literal', 'raw_string_literal']

    def __init__(self):
        self._method_cache = {}

    def parse(self, document: Document) -> Document:
        tree = self.tree_sitter_parser.parse(document.content.encode())
        self.walk_tree(tree.root_node, document)
        return document

    def walk_tree(self, node: Node, document: Document):
        stack = [node]

        while stack:
            current_node = stack.pop()

            # Check if the visit method is cached
            if current_node.type not in self._method_cache:
                method_name = 'visit_' + current_node.type
                self._method_cache[current_node.type] = getattr(self, method_name, self.generic_visit)

            # Get the visitor method from the cache
            visitor = self._method_cache[current_node.type]
            visitor(current_node, document)

            self.extract_if_identifier(current_node, document)
            self.extract_if_string(current_node, document)

            # Add children to the stack
            stack.extend(reversed(current_node.children))

    def extract_if_identifier(self, node, document):
        if 'identifier' in node.type and not node.children:
            name = node.text.decode('utf-8')
            span = node.byte_range
            document.identifiers.append((name, span))

    def extract_if_string(self, node, document):
        if node.type in self.string_types and not node.children:
            text = node.text.decode('utf-8')
            offset = node.byte_range[0]
            for word, span in basic_tokenize(text):
                span = offset + span[0], offset + span[1]
                document.strings.append((word, span))

    def generic_visit(self, node, document):
        # This method will be called if no explicit visitor function is available for a node type.
        pass


class TextParser:
    def parse(self, document: Document) -> Document:
        document.strings = basic_tokenize(document.content)
        return document
