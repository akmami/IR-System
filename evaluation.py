import logging
import os
import pandas as pd 
import numpy as np
from preprocess import read_dataset

# Lower logging level to give info to user
logging.getLogger().setLevel(logging.INFO)


# Get current directory
cwd = os.getcwd()

DOCS_DIR = os.path.join(cwd, "docs.csv")
GROUND_TRUTH_PATH = os.path.join(cwd, "dataset", "ground_truth.csv")

exists = os.path.exists(GROUND_TRUTH_PATH)

if not exists:
    logging.error("ground_truth.csv file is missing.")
    exit(0)

(_, ground_truth, _) = read_dataset(onlyColumn=True)


#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
# MARK: Methods
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
def evaluate(ranks_dir=DOCS_DIR):
    
    logging.info("evaluate function execution started.")
        
    if not os.path.exists(ranks_dir):
        
        logging.error("ranks_dir does not exists.")
        logging.info("evaluate function execution ended.")
        exit(0)
    
    np_docs = np.loadtxt(ranks_dir, delimiter=",")

    gt_docs = dict()

    for index, row in ground_truth.iterrows():
        if row[0] in gt_docs:
            gt_docs[row[0]].append(row[1])
        else:
            gt_docs[row[0]] = [row[1]]

    total = 0
    count = 0
    query_index = 0
    for row in np_docs:
        query_index += 1

        if query_index not in gt_docs.keys():
            continue
        
        tp = 0
        index = 0
        for doc in row:
            if doc in gt_docs[query_index]:
                tp += 1
            index += 1 
        print(row, tp, query_index)
        count += tp
        total += index
    
    print(count)
    print(total)

    logging.info("evaluate function execution ended.")


if __name__ == "__main__":
    evaluate()