import logging
import os
import pandas as pd 
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric, remove_stopwords, strip_short, stem_text


# Lower logging level to give info to user
logging.getLogger().setLevel(logging.INFO)

# get current directory
cwd = os.getcwd()

# Define NLP functions to be done in preprocess_string
filters = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric, remove_stopwords, strip_short, stem_text]

DOCUMENTS_DIR = os.path.join(cwd, "dataset", "documents.csv")
GROUND_TRUTH_DIR = os.path.join(cwd, "dataset", "ground_truth.csv")
QUERIES_DIR = os.path.join(cwd, "dataset", "queries.csv")

exists = os.path.exists(DOCUMENTS_DIR)
exists = exists and os.path.exists(GROUND_TRUTH_DIR)
exists = exists and os.path.exists(QUERIES_DIR)

if not exists:
    logging.error("Dataset is not complete. Under the dataset directory, there should be documents.csv, ground_truth.csv and queries.csv")
    exit(0)

logging.info("No missig files in dataset.")


#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
# MARK: Methods
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
# Read dataset
def read_dataset(onlyColumn=False):

    logging.info("read_dataset function execution started.")

    documents = pd.read_csv(DOCUMENTS_DIR)
    ground_truth = pd.read_csv(GROUND_TRUTH_DIR)
    queries = pd.read_csv(QUERIES_DIR)

    # Convert NaN entries to ""
    # This avoids the error when gensim function tries to preprocess it.
    documents = documents.where(pd.notnull(documents), "")
    queries = queries.where(pd.notnull(queries), "")

    if onlyColumn:
        documents = documents["text"]
        queries = queries["text"]
        del ground_truth["index"]

    logging.info("Dataset loaded.")
    logging.info("read_dataset function execution ended.")

    return (documents, ground_truth, queries)


# Pre-process dataset
def preprocess(onlyColumn=False):

    logging.info("preprocess function execution started.")

    (documents, ground_truth, queries) = read_dataset(onlyColumn)

    # Tokenize, lowercase and remove stop words for title, author and text columns in documents
    documents = [preprocess_string(text, filters=filters) for text in documents]

    # Tokenize, lowercase and remove stop words for query column in queries
    queries = [preprocess_string(query, filters=filters) for query in queries]

    logging.info("preprocess function execution ended.")

    return (documents, ground_truth, queries)


# Pre-process documents (list)
def preprocess_documents(documents):

    logging.info("preprocess_documents function execution started.")

    processed_documents = list()
    
    for document in documents:
        processed_documents.append(preprocess_string(document, filters=filters))
    
    logging.info("preprocess_documents function execution ended.")

    return processed_documents