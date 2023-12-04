import ast
from tree_sitter import Node
from tree_sitter_languages import get_parser

from parsing.base_parser import CodeParser
from parsing.parser_registry import register_parser
from parsing.utils import get_node_by_type


@register_parser('Python')
class PythonParser(CodeParser):
    tree_sitter_parser = get_parser('python')

    class ImportExtractor(ast.NodeVisitor):
        def __init__(self, document):
            self.document = document

        @staticmethod
        def resolve_module_path(module: str, level: int) -> str:
            module = module.replace('.', '/')
            level = max(level - 1, 0)
            module_path = ''.join(['../', ] * level) + module
            return module_path

        def visit_Import(self, node):
            for alias in node.names:
                module = alias.name
                self.document.connections.append(
                    dict(
                        from_path=self.resolve_module_path(module, 0),
                        name=alias.name
                    )
                )

        def visit_ImportFrom(self, node):
            module = node.module if node.module else ''
            level = node.level
            for alias in node.names:
                self.document.connections.append(
                    dict(
                        from_path=self.resolve_module_path(module, level),
                        name=alias.name
                    )
                )

    def visit_import_statement(self, node, document):
        extractor = self.ImportExtractor(document)
        extractor.visit(ast.parse(node.text.decode('utf-8')))

    def visit_import_from_statement(self, node, document):
        extractor = self.ImportExtractor(document)
        extractor.visit(ast.parse(node.text.decode('utf-8')))


@register_parser('JavaScript')
class JavaScriptParser(CodeParser):
    tree_sitter_parser = get_parser('javascript')

    def visit_import_statement(self, node: Node, document):
        import_clause_node = get_node_by_type(node.children, 'import_clause')
        names = list(self.process_import_clause(import_clause_node))

        from_path_node = get_node_by_type(node.children, 'string')
        from_path = from_path_node.text.decode('utf-8').strip('";') if from_path_node else None

        if names:
            for name in names:
                document.connections.append({"from_path": from_path, "name": name})
        else:
            # Handle empty import case
            document.connections.append({"from_path": from_path, "name": None})

    def process_import_clause(self, node: Node):
        if node:
            for child in node.children:
                if child.type in ['imported_default_binding', 'named_imports', 'identifier']:
                    yield from self.extract_imported_names(child)
                elif child.type == 'namespace_import':
                    # Handle namespace import
                    namespace_node = get_node_by_type(child.children, 'identifier')
                    if namespace_node:
                        yield namespace_node.text.decode('utf-8')

    def extract_imported_names(self, node: Node):
        if node.type == 'named_imports':
            for named_child in node.children:
                if named_child.type == 'import_specifier':
                    yield named_child.text.decode('utf-8')
        elif node.type in ['imported_default_binding', 'identifier']:
            yield node.text.decode('utf-8')

    def visit_call_expression(self, node: Node, document):
        # Process require and dynamic import statementsd
        function_node = get_node_by_type(node.children, ['identifier', 'import'])
        if function_node and function_node.text.decode('utf-8') in ['require', 'import']:
            arguments = get_node_by_type(node.children, 'arguments', recursive=True)

            from_path_node = get_node_by_type(arguments.children,
                                              ['string', 'template_string']) if arguments else None
            from_path = from_path_node.text.decode('utf-8').strip('`\'";') if from_path_node else None

            # Determine the parent node
            parent = function_node.parent
            if parent and parent.type not in ['variable_declarator', 'assignment_expression', 'member_expression']:
                parent = parent.parent

            # Extract the name if available
            name_node = get_node_by_type(parent.children, ['identifier', 'property_identifier']) \
                if parent else None
            name = name_node.text.decode('utf-8') if name_node else None

            if from_path:
                document.connections.append({"from_path": from_path, "name": name})


@register_parser('TypeScript')
class TypeScriptParser(JavaScriptParser):
    tree_sitter_parser = get_parser('typescript')
