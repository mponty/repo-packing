import pytest
from parsing.parsers_impl import TypeScriptParser


class Document:
    def __init__(self, content):
        self.content = content
        self.identifiers = []
        self.strings = []
        self.connections = []




_test_cases = [
    # Test default import
    ('import MyDefault from "./myModule";',
     [{'from_path': './myModule', 'name': 'MyDefault'}]),

    # Test named imports
    ('import { MyComponent, MyHelper } from "./components";', [
        {'from_path': './components', 'name': 'MyComponent'},
        {'from_path': './components', 'name': 'MyHelper'}
    ]),

    # Test namespace import
    ('import * as MyModule from "./myModule";',
     [{'from_path': './myModule', 'name': 'MyModule'}]),

    # Test combined default and named imports
    ('import Default, { Named1, Named2 } from "./module";', [
        {'from_path': './module', 'name': 'Default'},
        {'from_path': './module', 'name': 'Named1'},
        {'from_path': './module', 'name': 'Named2'}
    ]),

    # Test empty import (shouldn't add any connections)
    ('import "./myModule";',
     [{'from_path': './myModule', 'name': None}]),

    # Relative path with double dots
    ('import Helper from "../Helper";',
     [{'from_path': '../Helper', 'name': 'Helper'}]),

    # Absolute path
    ('import Package from "/src/Package";',
     [{'from_path': '/src/Package', 'name': 'Package'}]),

    # Path without file extension
    ('import Module from "./Module";',
     [{'from_path': './Module', 'name': 'Module'}]),

    # Path with file extension
    ('import File from "./File.ts";',
     [{'from_path': './File.ts', 'name': 'File'}]),

    # Path in a subdirectory
    ('import SubComponent from "./subdir/SubComponent";',
     [{'from_path': './subdir/SubComponent', 'name': 'SubComponent'}]),

    # Path with a complex structure
    ('import Complex from "../parent/child/Complex";',
     [{'from_path': '../parent/child/Complex', 'name': 'Complex'}]),

]


@pytest.mark.parametrize("content, expected", _test_cases)
def test_typescript_parser(content, expected):
    parser = TypeScriptParser()
    doc = Document(content)
    parser.parse(doc)
    assert doc.connections == expected


# Test data for require statements
_require_tests = [
    # Simple require statement
    ('const module = require("./myModule.ts");',
     [{'from_path': './myModule.ts', 'name': 'module'}]),

    # Require without assignment
    ('require("./anotherModule");',
     [{'from_path': './anotherModule', 'name': None}]),

    # Basic require statement with assignment
    ('const module = require("./myModule");',
     [{'from_path': './myModule', 'name': 'module'}]),

    # Require without assignment
    ('require("./anotherModule");',
     [{'from_path': './anotherModule', 'name': None}]),

    # Require in a variable declaration without const or let
    ('var myVar = require("./varModule");',
     [{'from_path': './varModule', 'name': 'myVar'}]),

    # Require with destructuring assignment
    # ('const { feature } = require("./featureModule");',
    #  [{'from_path': './featureModule', 'name': 'feature'}]),

    # Require with property access
    ('const feature = require("./featureModule").feature;',
     [{'from_path': './featureModule', 'name': 'feature'}]),

    # Require within a function
    ('function loadModule() { return require("./funcModule"); }',
     [{'from_path': './funcModule', 'name': None}]),

    # Nested require
    ('const nested = require(require("./pathModule"));',
     [{'from_path': './pathModule', 'name': None}]),

    # Require with template literals
    ('const template = require(`./templateModule`);',
     [{'from_path': './templateModule', 'name': 'template'}]),

]


@pytest.mark.parametrize("code,expected", _require_tests)
def test_require_statement(code, expected):
    parser = TypeScriptParser()
    document = Document(content=code)
    parser.parse(document)  # Parse the test code

    assert document.connections == expected
