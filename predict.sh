#!/usr/bin/env bash
# Makes predictions for the files in the given project directory

mkdir -p predictions
python3 code2vec.py --framework keras --load models/py_model/saved_model --predict
rm -rf predictions