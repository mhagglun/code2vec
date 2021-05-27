#!/usr/bin/env python
import re
import ast
import sys
import glob
import json
import astunparse
from tqdm import tqdm
from argparse import ArgumentParser
from collections import namedtuple, deque
from concurrent.futures import ThreadPoolExecutor, as_completed


Import = namedtuple("Import", ["parent", "name", "alias"])

class NodeVisitor(ast.NodeVisitor):
    def __init__(self, fname):
        self.fname = fname
        self.method_tokens = []

    def visit_FunctionDef(self, node):
        method_name = node.name
        method_body = astunparse.unparse(node.body)

        tokens = re.findall(r"\w+", method_body)
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

    # Parses AST, returning tokens on a per method basis
    def token_extractor(self, fname):
        try:
            with open(fname) as fh:
                root = ast.parse(fh.read(), fname)
        except Exception as e:
            if (args.verbose):
                print(f"Skipping problematic file {e}", fname, file=sys.stderr)
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
            # with ThreadPoolExecutor(max_workers=10) as executor:
            #     tasks = {executor.submit(
            #         self.token_extractor, file): file for file in projects}

            for file in projects:
                self.token_extractor(file)
                pbar.update(1)

            # for task in as_completed(tasks):
            #     pbar.update(1)

        with open(self.args.output, 'wt') as f:
            json.dump(self.tokens, f, indent=4)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-d", "--directory", dest="directory", type=str,
                        help="The directory of the projects to extract method tokens from", required=True)
    parser.add_argument("-o", "--output", dest="output", type=str, default="./tokens.json",
                        help="The output filepath", required=False)
    parser.add_argument("-v", "--verbose", dest="verbose", type=bool, default=False,
                        help="Increase verbosity of output", required=False)

    args = parser.parse_args()
    extractor = Extractor(args)
    extractor.extract()
    print("Done.")
