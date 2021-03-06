#!/usr/bin/env bash
###########################################################
# Change the following values to preprocess a new dataset.
# TRAIN_DIR, VAL_DIR and TEST_DIR should be paths to      
#   directories containing sub-directories with .java files
#   each of {TRAIN_DIR, VAL_DIR and TEST_DIR} should have sub-dirs,
#   and data will be extracted from .java files found in those sub-dirs).
# DATASET_NAME is just a name for the currently extracted 
#   dataset.                                              
# MAX_CONTEXTS is the number of contexts to keep for each 
#   method (by default 200).                              
# WORD_VOCAB_SIZE, PATH_VOCAB_SIZE, TARGET_VOCAB_SIZE -   
#   - the number of words, paths and target words to keep 
#   in the vocabulary (the top occurring words and paths will be kept). 
#   The default values are reasonable for a Tesla K80 GPU 
#   and newer (12 GB of board memory).
# NUM_THREADS - the number of parallel threads to use. It is 
#   recommended to use a multi-core machine for the preprocessing 
#   step and set this value to the number of cores.
# PYTHON - python3 interpreter alias.
PROJECT_DIR=/path/to/projects
DATASET_NAME=dataset_name
LANG=py
MAX_CONTEXTS=200
WORD_VOCAB_SIZE=1301136
PATH_VOCAB_SIZE=911417
TARGET_VOCAB_SIZE=261245
PYTHON=python3
JAVA=java
###########################################################

mkdir -p data
mkdir -p data/${DATASET_NAME}

PROJECT_DATA_FILE=data/${DATASET_NAME}

TRAIN_DATA_FILE=${PROJECT_DATA_FILE}/${LANG}/train.csv
VAL_DATA_FILE=${PROJECT_DATA_FILE}/${LANG}/val.csv
TEST_DATA_FILE=${PROJECT_DATA_FILE}/${LANG}/test.csv
EXTRACTOR_JAR=astminer/shadow/lib-0.6.jar

echo "Extracting paths from dataset..."
${JAVA} -jar ${EXTRACTOR_JAR} code2vec --lang ${LANG} --project ${PROJECT_DIR} --output ${PROJECT_DATA_FILE} --maxL 8 --maxW 2 --maxContexts ${MAX_CONTEXTS} --maxTokens ${WORD_VOCAB_SIZE} --maxPaths ${PATH_VOCAB_SIZE} --split-tokens --granularity method
sed -i 's/__//g' ${PROJECT_DATA_FILE}/${LANG}/path_contexts.csv
sed -i 's/_/|/g' ${PROJECT_DATA_FILE}/${LANG}/path_contexts.csv
echo "Finished extracting paths from dataset"

# Shuffle rows of csv
echo "Shuffling data..."
shuf ${PROJECT_DATA_FILE}/${LANG}/path_contexts.csv > ${PROJECT_DATA_FILE}/${LANG}/path_contexts_shuffled.csv


# Partition the large csv into csvs for train / val / test
echo "Partitioning into datasets..."
${PYTHON} create_datasets.py --filepath ${PROJECT_DATA_FILE}/${LANG}/path_contexts_shuffled.csv --output ${PROJECT_DATA_FILE}/${LANG}

TARGET_HISTOGRAM_FILE=data/${DATASET_NAME}/${DATASET_NAME}.histo.tgt.c2v
ORIGIN_HISTOGRAM_FILE=data/${DATASET_NAME}/${DATASET_NAME}.histo.ori.c2v
PATH_HISTOGRAM_FILE=data/${DATASET_NAME}/${DATASET_NAME}.histo.path.c2v

echo "Creating histograms from the training data"
cat ${TRAIN_DATA_FILE} | cut -d' ' -f1 | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${TARGET_HISTOGRAM_FILE}
cat ${TRAIN_DATA_FILE} | cut -d' ' -f2- | tr ' ' '\n' | cut -d',' -f1,3 | tr ',' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${ORIGIN_HISTOGRAM_FILE}
cat ${TRAIN_DATA_FILE} | cut -d' ' -f2- | tr ' ' '\n' | cut -d',' -f2 | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${PATH_HISTOGRAM_FILE}

${PYTHON} preprocess.py --train_data ${TRAIN_DATA_FILE} --test_data ${TEST_DATA_FILE} --val_data ${VAL_DATA_FILE} \
  --max_contexts ${MAX_CONTEXTS} --word_vocab_size ${WORD_VOCAB_SIZE} --path_vocab_size ${PATH_VOCAB_SIZE} \
  --target_vocab_size ${TARGET_VOCAB_SIZE} --word_histogram ${ORIGIN_HISTOGRAM_FILE} \
  --path_histogram ${PATH_HISTOGRAM_FILE} --target_histogram ${TARGET_HISTOGRAM_FILE} --output_name data/${DATASET_NAME}/${DATASET_NAME}

# If all went well, the raw data files can be deleted, because preprocess.py creates new files 
# with truncated and padded number of paths for each example.
# rm -rf ${TRAIN_DATA_FILE} ${VAL_DATA_FILE} ${TEST_DATA_FILE} ${TARGET_HISTOGRAM_FILE} ${ORIGIN_HISTOGRAM_FILE} \
#   ${PATH_HISTOGRAM_FILE}

