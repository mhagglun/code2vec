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


def get_leaf_attribute(node):
    node = node.value
    while isinstance(node, ast.Call) or isinstance(node, ast.Attribute) or isinstance(node, ast.Subscript):
        if isinstance(node, ast.Call):
            node = node.func
        elif isinstance(node, ast.Attribute) or isinstance(node, ast.Subscript):
            node = node.value
        else:
            break

    if isinstance(node, ast.Name):
        return node.id
    else:
        return None


def get_target(node):
    node = node.targets[0]
    while isinstance(node, ast.Subscript) or isinstance(node, ast.Attribute):
        node = node.value

    if isinstance(node, ast.Name):
        return node.id
    else:
        return None


class NodeVisitor(ast.NodeVisitor):
    def __init__(self, fname):
        self.fname = fname
        self.method_tokens = []

    def visit_FunctionDef(self, node):
        method_name = node.name
        tokens = []
        for body in node.body:
            for node in ast.walk(body):
                if isinstance(node, ast.Name):
                    tokens.append(node.id)
                elif isinstance(node, ast.Attribute):
                    tokens.append(node.attr)

        tokens = list(set(tokens))
        if len(tokens) > 0:
            self.method_tokens.append(
                {'file': self.fname, 'method_name': method_name, 'tokens': tokens})

    def generic_visit(self, node):
        ast.NodeVisitor.generic_visit(self, node)


class Extractor():
    def __init__(self, args):
        self.args = args
        self.tokens = []

    # Parses AST, returning library imports on a per method basis
    def token_extractor(self, fname):
        try:
            with open(fname) as fh:
                root = ast.parse(fh.read(), fname)
        except Exception as e:
            if (args.verbose):
                print(f"Skippings problematic file {e}", fname, file=sys.stderr)
            return

        nodeVisitor = NodeVisitor(fname)
        for node in ast.iter_child_nodes(root):
            nodeVisitor.visit(node)

        extracted_tokens = nodeVisitor.method_tokens
        if len(extracted_tokens) > 0:
            self.tokens.extend(extracted_tokens)

    def extract(self):
        projects = glob.glob(self.args.directory+'/**/*.py', recursive=True)
        with tqdm(total=len(projects), unit=' Files', desc='Extracting methods from files') as pbar:
            with ThreadPoolExecutor(max_workers=10) as executor:
                tasks = {executor.submit(
                    self.token_extractor, file): file for file in projects}

                for task in as_completed(tasks):
                    pbar.update(1)

        with open(self.args.output, 'wt') as f:
            json.dump(self.tokens, f, indent=4)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-d", "--directory", dest="directory", type=str,
                        help="The directory of the projects to extract libraries from", required=True)
    parser.add_argument("-o", "--output", dest="output", type=str, default="./tokens.json",
                        help="The output filepath", required=False)
    parser.add_argument("-v", "--verbose", dest="verbose", type=bool, default=False,
                        help="Increase verbosity of output", required=False)

    args = parser.parse_args()
    extractor = Extractor(args)
    extractor.extract()
    print("Done.")
