import re
from typing import Text, List, Tuple, Iterator
from pathlib import Path
import itertools


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
        keywords = re.findall(self.identifier_pattern, content)
        keywords = self.plural_to_singular_stem(keywords)
        keywords = self.word_filter(keywords)
        camel_split = self.camel_case_split(keywords)
        camel_trim = self.camel_case_trim(keywords)
        snake_split = self.snake_case_split(keywords)
        snake_trim = self.snake_case_trim(keywords)

        keywords = keywords + self.length_filter(camel_split + camel_trim + snake_split + snake_trim)
        keywords = keywords + [word.lower() for word in keywords]

        return ' '.join(keywords)

    def word_filter(self, keywords):
        return [word for word in keywords if (len(word) > self.min_length) and (word not in self.stopwords)]

    def length_filter(self, keywords):
        return [word for word in keywords if len(word) > self.min_length]

    def plural_to_singular_stem(self, keywords):
        keywords = [self._plural_to_singular_stem(word) for word in keywords]
        return keywords

    def _camel_case_splitter(self, keywords: List[Text]) -> Iterator[Tuple[Text]]:
        for word in keywords:
            pieces = re.sub(self.camel_case_pattern, r' \1', word).split()
            if len(pieces) > 1:
                yield pieces

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

    def _plural_to_singular_stem(self, word):
        def replace(match):
            if match.group(1):  # Matches (?!e|a)ies
                return 'y'
            elif match.group(2):  # Matches
                # es
                return ''
            elif match.group(3):  # Matches (?!s)s
                return ''
            return word  # No match (shouldn't happen with this regex)

        pattern = r'((?<!e|a)ies$)|(es$)|((?<!s)s$)'
        return re.sub(pattern, replace, word)


class DefaultAnalyzer(BaseAnalyzer):
    stopwords_filename = 'default'

    def __init__(self):
        current_dir = Path(__file__).parent
        filepath = current_dir.joinpath('stopwords', self.stopwords_filename)
        with open(filepath, 'r') as f:
            stopwords = [line.strip('\n') for line in f.readlines()]
        super().__init__(stopwords=stopwords)


class PythonAnalyzer(DefaultAnalyzer):
    stopwords_filename = 'Python'
