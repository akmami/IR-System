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
def evaluate(ranks_dir=DOCS_DIR, logs=logging.INFO):

    logging.getLogger().setLevel(logs)
    
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
    mean_avr_prec = 0.0
    for row in np_docs:
        query_index += 1

        if query_index not in gt_docs.keys():
            continue
        
        tp = 0
        index = 0
        mean_avr_prec_query = 0.0
        for doc in row:
            if doc in gt_docs[query_index]:
                tp += 1
                mean_avr_prec_query += tp / (index + 1.0)
            index += 1 
        count += tp
        total += index

        if tp > 0:
            mean_avr_prec += mean_avr_prec_query / tp
    
    mean_avr_prec /= query_index
    print(count, total, "MAP: " + ( "%.3f" % mean_avr_prec ) )

    logging.info("evaluate function execution ended.")


if __name__ == "__main__":
    evaluate()