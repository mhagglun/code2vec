import csv
import config
import shutil
import numpy as np
import pandas as pd

from tqdm import tqdm
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split

MAX_CONTEXTS = 200

def create_dataset(args):

    # shutil.rmtree(f'{args.output}/train.csv')
    # shutil.rmtree(f'{args.output}/val.csv')
    # shutil.rmtree(f'{args.output}/test.csv')

    df = pd.read_csv(args.filepath, delimiter=' ', header=None, names=np.arange(0, MAX_CONTEXTS+1), iterator=True, chunksize=10000)
    train_size, val_size, test_size = 0.9, 0.05, 0.05

    for chunk in tqdm(df, desc="Writing to datasets"):
        train, remainder = train_test_split(chunk, test_size=(1-train_size), shuffle=True)
        validate, test =  train_test_split(remainder, test_size=test_size/(test_size + val_size))


        train.to_csv(f'{args.output}/train.csv', encoding='utf-8', mode="a", sep=" ", index=False, header=None, quoting = csv.QUOTE_NONE, escapechar = ' ')
        validate.to_csv(f'{args.output}/val.csv', encoding='utf-8', mode="a", sep=" ", index=False, header=None, quoting = csv.QUOTE_NONE, escapechar = ' ')
        test.to_csv(f'{args.output}/test.csv', encoding='utf-8', mode="a", sep=" ", index=False, header=None, quoting = csv.QUOTE_NONE, escapechar = ' ')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--filepath", dest="filepath", type=str,
                        help="The filepath to the csv containing the path contexts", required=True)
    parser.add_argument("-o", "--output", dest="output", type=str, default="./data",
                        help="The path to output directory", required=True)
    args = parser.parse_args()
    create_dataset(args)