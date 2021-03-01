#!/usr/bin/env bash
###########################################################
# Exports the code vectors for the test data
# Pass the model name as the first argument and the dataset name as the second
python3 code2vec.py --load $1 --test data/$2/$2.test.c2v --export_code_vectors
python3 code2vec.py --load $1 --save_w2v data/$2/libraries.txt --save_t2v data/$2/targets.txt
