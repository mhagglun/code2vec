#!/usr/bin/env bash
# Makes predictions for the files in the given project directory

mkdir -p predictions
python3 code2vec.py --load models/$1/saved_model --predict
rm -rf predictions