import ast
import re
from collections import namedtuple, defaultdict
from pathlib import Path
from typing import List, Tuple, Dict

ImportedName = namedtuple('ImportedName', ['name', 'module', 'from_file', 'to_file'])


class PythonImportsAnalyzer:

    def __init__(self, python_files, propagation_lvl=3):
        self.files = {file['path']: file['content'] for file in python_files}
        self._keywords = dict()
        self.propagation_lvl = propagation_lvl

    def analyze(self) -> List[Dict]:
        imported_names = self.parse_imported_names()

        # transitive imports propagation
        for _ in range(self.propagation_lvl):
            imported_names = self.propagate_names(imported_names)

        connections = set((name.from_file, name.to_file) for name in imported_names)

        return [
            dict(
                from_file=from_file,
                to_file=to_file,
                weight=1.)
            for from_file, to_file in connections]

    def parse_imports(self, source_code):
        """
        Parses Python source code and extracts all names in import statements.

        :param source_code: String containing the Python source code
        :return: Dictionary with module paths as keys and list of imported names as values
        """

        class ImportExtractor(ast.NodeVisitor):
            def __init__(self):
                self.imports = defaultdict(list)

            def visit_Import(self, node):
                for alias in node.names:
                    module = alias.name
                    self.imports[(module, 0)].append(alias.name)

            def visit_ImportFrom(self, node):
                module = node.module if node.module else ''
                level = node.level
                for alias in node.names:
                    self.imports[(module, level)].append(alias.name)

        tree = ast.parse(source_code)
        extractor = ImportExtractor()
        extractor.visit(tree)

        # Return the dictionary containing the imports
        return extractor.imports

    def resolve_module_path(self, module: str, level: int, current_path: str):
        current_dir = Path(current_path).parent
        module = module.replace('.', '/')
        level = max(level - 1, 0)

        level_dir = current_dir.joinpath(''.join(['../', ] * level)).resolve()
        module_path = level_dir.joinpath(module).resolve()

        if str(module_path) + '.py' in self.files:
            return str(module_path) + '.py'
        elif str(module_path.joinpath('__init__.py')) in self.files:
            return str(module_path.joinpath('__init__.py'))

        # Try absolute path from root
        module_path = Path('/').joinpath(module).resolve()

        if str(module_path) + '.py' in self.files:
            return str(module_path) + '.py'
        elif str(module_path.joinpath('__init__.py')) in self.files:
            return str(module_path.joinpath('__init__.py'))

        return None

    def parse_imported_names(self) -> List[ImportedName]:
        imported_names = []
        for current_path, source_code in self.files.items():

            imports = self.parse_imports(source_code)

            for (module, level), names in imports.items():
                module_path = self.resolve_module_path(module, level, current_path)
                if module_path:
                    for name in names:
                        imported = ImportedName(name=name, module=module,
                                                from_file=module_path, to_file=current_path)
                        imported_names.append(imported)

        return imported_names

    def _filter_names(self, names: List[ImportedName]):
        term_pattern = re.compile('[_a-zA-Z0-9]+')

        filtered = []
        for name in names:
            if name.from_file not in self._keywords:
                content = self.files[name.from_file]
                self._keywords[name.from_file] = set(re.findall(term_pattern, content))

            if name.name in self._keywords[name.from_file]:
                filtered.append(name)

        return filtered

    def propagate_names(self, imported_names: List[ImportedName]):
        """Propagate names through transitive imports."""
        transit_dict = dict()
        transited = []
        for imported in imported_names:
            transit_dict[(imported.name, imported.to_file)] = imported.from_file

        for imported in imported_names:
            from_file = transit_dict.get((imported.name, imported.from_file))
            if from_file is None:
                # wildcart import
                from_file = transit_dict.get(('*', imported.from_file))

            if from_file:
                transited.append(
                    ImportedName(name=imported.name, module=imported.module, to_file=imported.to_file,
                                 from_file=from_file)
                )

        transited = self._filter_names(transited)
        return imported_names + transited
