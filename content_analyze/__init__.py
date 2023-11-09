from typing import List, Text
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

    def analyze(self, content: Text, language='default') -> Text:
        analyzer = self.analyzers.get(language, self.default_analyzer)
        return analyzer.analyze(content)
