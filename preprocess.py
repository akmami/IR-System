import logging
import os
import pandas as pd 
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric, remove_stopwords, strip_short, stem_text


# Lower logging level
logging.getLogger().setLevel(logging.INFO)


# get current directory
cwd = os.getcwd()


# check whether datasets exists
def check_dataset():
    exists = os.path.exists(os.path.join(cwd, "dataset/documents.csv"))
    exists = exists and os.path.exists(os.path.join(cwd, "dataset/ground_truth.csv"))
    exists = exists and os.path.exists(os.path.join(cwd, "dataset/queries.csv"))

    if not exists:
        logging.error("Dataset is not complete. Under the dataset directory, there should be documents.csv, ground_truth.csv and queries.csv")
        exit(-1)

    logging.info("No missig files in dataset.")


# Read dataset
def read_dataset():
    
    check_dataset()

    documents = pd.read_csv("dataset/documents.csv") 
    ground_truth = pd.read_csv("dataset/ground_truth.csv") 
    queries = pd.read_csv("dataset/queries.csv") 

    logging.info("Dataset loaded.")

    return (documents, ground_truth, queries)


# Pre-process dataset
def preprocess():

    (documents, ground_truth, queries) = read_dataset()

    # Convert NaN entries to ""
    # This avoids the error when gensim function tries to preprocess it.
    documents = documents.where(pd.notnull(documents), "")

    '''
    documents["title"] = [stem_text(title) for title in documents["title"]]
    documents["title"] = [list(tokenize(title, lower=True)) for title in documents["title"]]
    documents["title"] = [remove_stopword_tokens(title, stopwords=STOPWORDS) for title in documents["title"]]
    '''

    # Define NLP functions to be done in preprocess_string
    filters = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric, remove_stopwords, strip_short, stem_text]

    documents = documents["text"]
    queries = queries["text"]

    # Tokenize, lowercase and remove stop words for title, author and text columns in documents
    # documents["title"] = [preprocess_string(title, filters=filters) for title in documents["title"]]
    # documents["author"] = [preprocess_string(author, filters=filters) for author in documents["author"]]
    documents = [preprocess_string(text, filters=filters) for text in documents]

    # Tokenize, lowercase and remove stop words for query column in queries
    queries = [preprocess_string(query, filters=filters) for query in queries]

    return (documents, ground_truth, queries)