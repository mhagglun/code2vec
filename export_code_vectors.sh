#!/usr/bin/env bash
###########################################################
# Exports the code vectors for the test data
# Pass the model name as the first argument and the dataset name as the second
python3 code2vec.py --load models/$1/saved_model --test data/$2/$2.test.c2v --export_code_vectors
