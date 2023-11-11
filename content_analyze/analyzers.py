import re
from typing import Text, List, Tuple, Iterator
from pathlib import Path
import itertools
import functools


class BaseAnalyzer:
    identifier_pattern = re.compile('[_a-zA-Z\u0080-\uFFFF]+[0-9]*')
    camel_case_pattern = re.compile(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))')

    def __init__(self, stopwords: List[Text] = None, min_length: int = 2):
        self.stopwords = set(stopwords) if stopwords else {}
        self.min_length = min_length

    def analyze(self, content: Text) -> Text:
        """`
        :param content: code or text to analyze
        :return: space separated keyword terms
        """
        keywords = self.parse_keywords(content)
        keywords = self.plural_to_singular_stem(keywords)
        keywords = self.word_filter(keywords)
        camel_split = self.camel_case_split(keywords)
        camel_trim = self.camel_case_trim(keywords)
        snake_split = self.snake_case_split(keywords)
        snake_trim = self.snake_case_trim(keywords)

        keywords = keywords + self.length_filter(camel_split + camel_trim + snake_split + snake_trim)
        keywords = keywords + [word.lower() for word in keywords]

        return ' '.join(keywords)

    def parse_keywords(self, content):
        keywords = self.identifier_pattern.findall(content)
        return keywords

    def word_filter(self, keywords):
        return [word for word in keywords if (len(word) > self.min_length) and (word not in self.stopwords)]

    def length_filter(self, keywords):
        return [word for word in keywords if len(word) > self.min_length]

    def plural_to_singular_stem(self, keywords):
        keywords = [self._plural_to_singular_stem(word) for word in keywords]
        return keywords

    def _camel_case_splitter(self, keywords: List[Text]) -> Iterator[Tuple[Text]]:
        for word in keywords:
            if not word.islower():
                pieces = self._camel_case_word_split(word)
                if len(pieces) > 1:
                    yield pieces

    @staticmethod
    @functools.cache
    def _camel_case_word_split(word):
        return BaseAnalyzer.camel_case_pattern.sub(r' \1', word).split()

    def _snake_case_splitter(self, keywords: List[Text]) -> Iterator[Tuple[Text]]:
        for word in keywords:
            pieces = word.split('_')
            if len(pieces) > 1:
                yield pieces

    def camel_case_split(self, keywords: List[Text]) -> List[Text]:
        pieces = list(itertools.chain(*self._camel_case_splitter(keywords)))
        return pieces

    def camel_case_trim(self, keywords: List[Text]) -> List[Text]:
        trimed = [''.join(pieces[1:]) for pieces in self._camel_case_splitter(keywords)]
        return trimed

    def snake_case_split(self, keywords: List[Text]) -> List[Text]:
        pieces = list(itertools.chain(*self._snake_case_splitter(keywords)))
        return pieces

    def snake_case_trim(self, keywords: List[Text]) -> List[Text]:
        trimed = [''.join(pieces[1:]) for pieces in self._snake_case_splitter(keywords)]
        return trimed

    @staticmethod
    @functools.cache
    def _plural_to_singular_stem(word):
        if word.endswith('s'):
            if word.endswith('ies'):
                if not (word.endswith('eies') or word.endswith('aies')):
                    return word[:-3] + 'y'  # Replace 'ies' with 'y'
            elif word.endswith('es'):
                if not word.endswith('ses'):
                    return word[:-2]  # Remove 'es'
            else:
                if not word.endswith('ss'):
                    return word[:-1]  # Remove 's'
        return word  # No changes for other cases


class DefaultAnalyzer(BaseAnalyzer):
    stopwords_filename = ('default',)

    def __init__(self):
        stopwords = []

        if isinstance(self.stopwords_filename, str):
            stopwords_filenames = (self.stopwords_filename,)
        else:
            stopwords_filenames = self.stopwords_filename

        for filename in stopwords_filenames:
            stopwords += self.load_stopwords_file(filename)

        super().__init__(stopwords=stopwords)

    def load_stopwords_file(self, stopwords_filename: str):
        current_dir = Path(__file__).parent
        filepath = current_dir.joinpath('stopwords', stopwords_filename)
        with open(filepath, 'r') as f:
            stopwords = [line.strip() for line in f.readlines() if len(line.strip()) > 0]
        return stopwords


class PythonAnalyzer(DefaultAnalyzer):
    stopwords_filename = 'python'


class JupyterAnalyzer(DefaultAnalyzer):
    stopwords_filename = ('jupyter', 'python', 'text')


class JavaAnalyzer(DefaultAnalyzer):
    stopwords_filename = 'java'


class JavaScriptAnalyzer(DefaultAnalyzer):
    stopwords_filename = 'javascript'


class CAnalyzer(DefaultAnalyzer):
    stopwords_filename = 'c'


class CSharpAnalyzer(DefaultAnalyzer):
    stopwords_filename = 'csharp'


class CppAnalyzer(DefaultAnalyzer):
    stopwords_filename = 'cpp'


class PHPAnalyzer(DefaultAnalyzer):
    stopwords_filename = 'php'


class XMLAnalyzer(DefaultAnalyzer):
    stopwords_filename = 'xml'

    def parse_keywords(self, content):
        keywords = super().parse_keywords(content)
        # Lowering term frequency
        keywords = list(set(keywords))
        return keywords


class HTMLAnalyzer(XMLAnalyzer):
    stopwords_filename = ('html', 'text')


class JSONAnalyzer(XMLAnalyzer):
    stopwords_filename = ('default', 'text')


class YAMLAnalyzer(XMLAnalyzer):
    stopwords_filename = 'default'


class TextAnalyzer(XMLAnalyzer):
    stopwords_filename = 'text'
