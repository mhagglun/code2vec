#!/usr/bin/env bash
###########################################################
python3 code2vec.py --load $1 --predict
rm -rf predictions/