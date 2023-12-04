from tree_sitter_languages import get_parser

from parsing.base_parser import DefaultParser, CodeParser, TextParser

PARSER_REGISTRY = {
    'default': DefaultParser(),
    'Markdown': TextParser(),
    'Text': TextParser(),
}


def register_parser(language, *args, **kwargs):
    def _register_parser(cls):
        PARSER_REGISTRY[language] = cls(*args, **kwargs)
        return cls

    return _register_parser


def _preregister_parsers():
    tree_sitter_default_mapping = {
        'Bash': 'bash',
        'C': 'c',
        'C#': 'c_sharp',
        'Common Lisp': 'commonlisp',
        'C++': 'cpp',
        'CSS': 'css',
        'Dockerfile': 'dockerfile',
        'Graphviz (DOT)': 'dot',
        'Emacs Lisp': 'elisp',
        'Elixir': 'elixir',
        'Elm': 'elm',
        'HTML+ERB': 'embedded_template',
        'EJS': 'embedded_template',
        'Erlang': 'erlang',
        'Fortran': 'fortran',
        'Fortran Free Form': 'fortran',
        'Go': 'go',
        'Hack': 'hack',
        'Haskell': 'haskell',
        'HCL': 'hcl',
        'HTML': 'html',
        'Java': 'java',
        'JavaScript': 'javascript',
        'Julia': 'julia',
        'Kotlin': 'kotlin',
        'Lua': 'lua',
        'Makefile': 'make',
        'Objective-C': 'objc',
        'OCaml': 'ocaml',
        'Perl': 'perl',
        'PHP': 'php',
        'Python': 'python',
        'R': 'r',
        'reStructuredText': 'rst',
        'Ruby': 'ruby',
        'Rust': 'rust',
        'Scala': 'scala',
        'SQL': 'sql',
        'TOML': 'toml',
        'TSQL': 'tsq',
        'TSX': 'tsx',
        'TypeScript': 'typescript',
        'YAML': 'yaml',
    }

    for language, parser_alias in tree_sitter_default_mapping.items():
        parser_cls = type(f"_{parser_alias.title()}Parser", (CodeParser,), {
            "__init__": CodeParser.__init__,
            "tree_sitter_parser": get_parser(parser_alias),
        })
        register_parser(language)(parser_cls)


# Pre-register base parsers based on available precompiled tree-sitter parsers
# Note: connections are not handled by these parsers !!!
#       hence, the language specific parser should be reimplemented for connection analysis
_preregister_parsers()
