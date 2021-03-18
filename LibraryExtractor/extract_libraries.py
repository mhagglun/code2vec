#!/usr/bin/env python

import ast
import sys
import glob
import json
from tqdm import tqdm
from argparse import ArgumentParser
from collections import namedtuple, deque
from concurrent.futures import ThreadPoolExecutor, as_completed

Import = namedtuple("Import", ["parent", "name", "alias"])


def get_imports(tree):
    imports = []
    import_dict = {}

    # Extract import statements from parsed AST
    import_statements = []
    for node in tree:
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            import_statements.append(node)

    for node in import_statements:
        # Get parent library name
        if isinstance(node, ast.Import):
            module = node.names[0].name

            for n in node.names:
                imports.append(Import(module, n.name, n.asname))
                import_dict[n.name] = module
                if n.asname is not None:
                    import_dict[n.asname] = module

        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            module = node.module

            for n in node.names:
                imports.append(Import(module, n.name, n.asname))
                import_dict[n.name] = f"{module}.{n.name}"
                if n.asname is not None:
                    import_dict[n.asname] = f"{module}.{n.name}"
        else:
            continue

    return import_dict


def get_leaf_attribute(node):
    node = node.value
    path = deque()
    while isinstance(node, ast.Call) or isinstance(node, ast.Attribute) or isinstance(node, ast.Subscript):
        if isinstance(node, ast.Call):
            node = node.func
        elif isinstance(node, ast.Attribute) or isinstance(node, ast.Subscript):
            if isinstance(node, ast.Attribute):
                path.appendleft(node.attr)
            node = node.value
        else:
            break

    if isinstance(node, ast.Name):
        node_name = node.id
        if (node_name == 'self' and len(path) > 0):
            node_name += f".{path[0]}"
            path.popleft()
        return node_name, path
    else:
        return None, path


def get_target(node):
    node = node.targets[0]
    path = deque()
    while isinstance(node, ast.Subscript) or isinstance(node, ast.Attribute):
        if isinstance(node, ast.Attribute):
            path.appendleft(node.attr)
        node = node.value

    if isinstance(node, ast.Name):
        node_name = node.id
        if (node_name == 'self' and len(path) > 0):
            node_name += f".{path[0]}"
        return node_name
    else:
        return None


class NodeVisitor(ast.NodeVisitor):
    def __init__(self, fname, imports, filters):
        self.fname = fname
        self.imports = imports
        self.filters = filters
        self.method_libraries = []

    def visit_FunctionDef(self, node):
        method_name = node.name
        if (method_name in self.filters):
            library_references, path = [], deque()
            for body in node.body:
                # Get target name if possible
                if body.__dict__.get('targets') is not None:
                    target = get_target(body)
                else:
                    target = None

                # Get reference if possible
                if body.__dict__.get('value') is not None:
                    reference, path = get_leaf_attribute(body)
                else:
                    reference = None

                try:
                    # Create mapping between library and the assigned target if reference is from a library
                    if reference is not None and target is not None:
                        library = self.imports[reference]
                        self.imports[target] = library
                    # Look up assignment value if it exists
                    elif reference is not None:
                        library = self.imports[reference]
                    # Look up target if it exists
                    elif target is not None:
                        library = self.imports[target]
                    else:
                        continue
                except:
                    continue
                path.appendleft(library)
                library_path = ".".join(path)
                library_references.append(library_path)

        if len(library_references) > 0:
            self.method_libraries.append(
                {'file': self.fname, 'method_name': method_name, 'references': library_references})

    def generic_visit(self, node):
        ast.NodeVisitor.generic_visit(self, node)


class LibraryExtractor():
    def __init__(self, args):
        self.args = args
        self.method_libraries = []

    # Parses AST, returning library imports on a per method basis
    def extract_libraries(self, fname):
        try:
            with open(fname) as fh:
                root = ast.parse(fh.read(), fname)
        except Exception as e:
            if (args.verbose):
                print(f"Skipping problematic file {e}", fname, file=sys.stderr)
            return
        nodes = ast.walk(root)
        imports = get_imports(nodes)
        nodeVisitor = NodeVisitor(fname, imports, args.filters)
        for node in ast.iter_child_nodes(root):
            nodeVisitor.visit(node)

        extracted_libraries = nodeVisitor.method_libraries
        if len(extracted_libraries) > 0:
            self.method_libraries.extend(extracted_libraries)

    def extract(self):
        projects = glob.glob(self.args.directory+'/**/*.py', recursive=True)
        with tqdm(total=len(projects), unit=' Files', desc='Extracting libraries from files') as pbar:
            with ThreadPoolExecutor(max_workers=10) as executor:
                tasks = {executor.submit(
                    self.extract_libraries, file): file for file in projects}
                for task in as_completed(tasks):
                    pbar.update(1)

        with open(self.args.output, 'wt') as f:
            json.dump(self.method_libraries, f, indent=4)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-d", "--directory", dest="directory", type=str,
                        help="The directory of the projects to extract libraries from", required=True)
    parser.add_argument("-o", "--output", dest="output", type=str, default="./code2lib.json",
                        help="The output filepath", required=False)
    parser.add_argument("-v", "--verbose", dest="verbose", type=bool, default=False,
                        help="Increase verbosity of output", required=False)
    parser.add_argument("-f", "--filters", dest="filters", type=bool, default=['predict', 'save', 'train', 'transform'],
                        help="A comma separated list of method names to filter by", required=False)

    args = parser.parse_args()
    libraryExtractor = LibraryExtractor(args)
    libraryExtractor.extract()
    print("Done.")
