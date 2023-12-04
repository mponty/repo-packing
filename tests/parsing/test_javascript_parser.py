import pytest
from parsing.parsers_impl import JavaScriptParser


class Document:
    def __init__(self, content):
        self.content = content
        self.identifiers = []
        self.strings = []
        self.connections = []


# Test data for different import statements
_import_tests = [
    # Dynamic imports
    ('const module = import("my-dynamic-module");',
     [{'from_path': 'my-dynamic-module', 'name': 'module'}]),

    # Default import
    ('import MyDefault from "my-default-module";',
     [{'from_path': 'my-default-module', 'name': 'MyDefault'}]),

    # Named imports
    ('import { MyComponent, MyUtil } from "my-component-module";',
     [{'from_path': 'my-component-module', 'name': 'MyComponent'},
      {'from_path': 'my-component-module', 'name': 'MyUtil'}]),

    # Namespace import
    ('import * as MyModule from "my-module";',
     [{'from_path': 'my-module', 'name': 'MyModule'}]),

    # Mixed default and named imports
    ('import MyDefault, { MyComponent } from "my-mixed-module";',
     [{'from_path': 'my-mixed-module', 'name': 'MyDefault'},
      {'from_path': 'my-mixed-module', 'name': 'MyComponent'}]),

    # Import without module specifiers
    ('import "my-side-effect-module";',
     [{'from_path': 'my-side-effect-module', 'name': None}]),

]


@pytest.mark.parametrize("code,expected", _import_tests)
def test_import_statement(code, expected):
    parser = JavaScriptParser()  # Initialize your parser
    document = Document(content=code)
    parser.parse(document)  # Parse the test code

    assert document.connections == expected


# Test data for require statements in JavaScript
_require_tests = [
    ('const module = require("./myModule.js");',
     [{'from_path': './myModule.js', 'name': 'module'}]),

    ('require("./anotherModule");',
     [{'from_path': './anotherModule', 'name': None}]),

    ('var myVar = require("./varModule");',
     [{'from_path': './varModule', 'name': 'myVar'}]),

    # ('const { feature } = require("./featureModule");',
    #  [{'from_path': './featureModule', 'name': 'feature'}]),

    ('const feature = require("./featureModule").feature;',
     [{'from_path': './featureModule', 'name': 'feature'}]),

    ('function loadModule() { return require("./funcModule"); }',
     [{'from_path': './funcModule', 'name': None}]),

    ('const nested = require(require("./pathModule"));',
     [{'from_path': './pathModule', 'name': None}]),

    ('const template = require(`./templateModule`);',
     [{'from_path': './templateModule', 'name': 'template'}]),

]


@pytest.mark.parametrize("code,expected", _require_tests)
def test_require_statement(code, expected):
    parser = JavaScriptParser()  # Initialize your parser
    document = Document(content=code)
    parser.parse(document)  # Parse the test code

    assert document.connections == expected
