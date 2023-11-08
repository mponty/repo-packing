from typing import List, Text
from .analyzers import DefaultAnalyzer, PythonAnalyzer

__all__ = ['ContentAnalyzer']


class ContentAnalyzer:
    default_analyzer = DefaultAnalyzer()
    analyzers = {
        'Python': PythonAnalyzer(),
    }

    def analyze(self, content: Text, language='default') -> Text:
        analyzer = self.analyzers.get(language, self.default_analyzer)
        return analyzer.analyze(content)
