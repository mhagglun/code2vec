#!/usr/bin/env python
import re
import ast
import csv
import sys
import glob
import json
from tqdm import tqdm
from argparse import ArgumentParser
from collections import namedtuple, deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import astunparse
from tensor2tensor.data_generators import text_encoder

import jsonlines
import code_to_subtokenized_sentences
import tokenizer_registry

from absl import app
from absl import flags
from filelock import FileLock
import logging

logging.getLogger('filelock').setLevel(logging.ERROR)
FLAGS = flags.FLAGS

flags.DEFINE_string('vocabulary_filepath', None,
                    'Path to the subword vocabulary.')

flags.DEFINE_string('directory', None,
                    'Path to the directory containing Python source code files.')

flags.DEFINE_string('output_filepath', None,
                    'Path to the output file of subtokenized source code.')


flags.DEFINE_boolean('verbose', False,
                     'The verbosity of the output')

flags.DEFINE_enum_class(
    'tokenizer',
    default=tokenizer_registry.TokenizerEnum.PYTHON,
    enum_class=tokenizer_registry.TokenizerEnum,
    help='The tokenizer to use.')


def camel_case_split(string):
    matches = re.finditer(
        '.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)',
        string,
    )
    return f'|'.join([m.group(0).lower() for m in matches])


def snake_case_split(string):
    return f'|'.join([x for x in string.split('_') if x != ''])


class Extractor():
    def __init__(self, tokenizer, subword_tokenizer):
        self.tokenizer = tokenizer
        self.subword_tokenizer = subword_tokenizer

    def token_extractor(self, fname):
        with open(fname, 'r') as input_file:
            try:
                root = ast.parse(input_file.read(), fname)
            except Exception as e:
                if FLAGS.verbose:
                    print(
                        f"Skipping problematic file {e}", fname, file=sys.stderr)
                return
        try:
            for node in ast.walk(root):
                if isinstance(node, ast.FunctionDef):
                    method_name = camel_case_split(snake_case_split(node.name))
                    node.name = "f"  # Obfuscate method name before tokenizing
                    method_string = astunparse.unparse(node)

                    subtokenized_sentences = (
                        code_to_subtokenized_sentences.code_to_cubert_sentences(
                            code=method_string,
                            initial_tokenizer=self.tokenizer,
                            subword_tokenizer=self.subword_tokenizer))

                    with FileLock(f"{FLAGS.output_filepath}.lock"):
                        with jsonlines.open(FLAGS.output_filepath, mode='a') as writer:
                            writer.write(json.dumps(
                                {"file": fname, "method_name": method_name, "tokens": subtokenized_sentences}))
        except Exception as e:
            print(e)

    def extract(self):
        projects = glob.glob(FLAGS.directory+'/**/*.py', recursive=True)
        with tqdm(total=len(projects), unit=' Files', desc='Extracting libraries from files') as pbar:
            with ThreadPoolExecutor(max_workers=10) as executor:
                tasks = {executor.submit(
                    self.token_extractor, file): file for file in projects}

                for task in as_completed(tasks):
                    pbar.update(1)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    # The value of the `TokenizerEnum` is a `CuBertTokenizer` subclass.
    tokenizer = FLAGS.tokenizer.value()
    subword_tokenizer = text_encoder.SubwordTextEncoder(
        FLAGS.vocabulary_filepath)

    extractor = Extractor(tokenizer, subword_tokenizer)
    extractor.extract()


if __name__ == '__main__':
    flags.mark_flag_as_required('vocabulary_filepath')
    flags.mark_flag_as_required('directory')
    flags.mark_flag_as_required('output_filepath')
    app.run(main)
