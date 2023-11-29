from typing import Any, Dict
from .analyzers import *

__all__ = ['ContentAnalyzer']


class ContentAnalyzer:
    default_analyzer = DefaultAnalyzer()
    analyzers = {
        'Python': PythonAnalyzer(),
        'Jupyter Notebook': JupyterAnalyzer(),
        'Java': JavaAnalyzer(),
        'JavaScript': JavaScriptAnalyzer(),
        'C': CAnalyzer(),
        'C++': CppAnalyzer(),
        'C#': CSharpAnalyzer(),
        'PHP': PHPAnalyzer(),
        'XML': XMLAnalyzer(),
        'JSON': JSONAnalyzer(),
        'YAML': YAMLAnalyzer(),
        'HTML': HTMLAnalyzer(),
        'Markdown': TextAnalyzer(),
        'Text': TextAnalyzer(),
    }

    def analyze(self, file: Dict[str, Any]) -> Text:
        content, path, language = file['content'], file['path'], file['language']
        analyzer = self.analyzers.get(language, self.default_analyzer)
        return analyzer.analyze(path + '\n' + content)
