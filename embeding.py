import logging
import os
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from preprocess import read_dataset, preprocess_documents


# Lower logging level to give info to user
logging.getLogger().setLevel(logging.INFO)

# Get current directory
cwd = os.getcwd()

# Define embeding vectors' directory 
DOCUMENTS_EMBEDDING_WORD2VEC_DIR = os.path.join(cwd, "dataset", "documents_embedding_word2vec.csv")
DOCUMENTS_EMBEDDING_PRETRAINED_DIR = os.path.join(cwd, "dataset", "documents_embedding_pretrained.csv")
WORD2VEC_MODEL_DIR = os.path.join(cwd, "model", "word2vec.model")
WORD2VEC_PRETRAINED_MODEL_DIR = os.path.join(cwd, "model", "word2vec-google-news-300", "word2vec-google-news-300.gz")

# Validate models
exists = os.path.exists(WORD2VEC_MODEL_DIR)
exists = exists and os.path.exists(WORD2VEC_PRETRAINED_MODEL_DIR)

if not exists:
    logging.error("Models are not complete. Please make sure that Word2Vec and pre-trained one exists under model directory.")
    exit(0)

logging.info("Loading models.")

# Increase logging level to prevent gensim logs
logging.getLogger().setLevel(logging.WARN)

word2vec = KeyedVectors.load_word2vec_format(WORD2VEC_MODEL_DIR, binary=True)
pre_trained = KeyedVectors.load_word2vec_format(WORD2VEC_PRETRAINED_MODEL_DIR, binary=True)

# Lower logging level to give info to user
logging.getLogger().setLevel(logging.INFO)

logging.info("Models loaded successfully.")


#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
# MARK: Methods
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
def embed(document, model="word2vec"):
    
    document_tokens = document
    
    if type(document) is str:
        document_tokens = document.split()
    
    if model == "word2vec":
        embeddings = []
        if len(document_tokens)<1:
            return np.zeros(word2vec.vector_size)
        else:
            for token in document_tokens:
                if token in word2vec.index_to_key:
                    embeddings.append(word2vec.get_vector(token))
                else:
                    embeddings.append(np.random.rand(word2vec.vector_size))
            # mean the vectors of individual words to get the vector of the document
            return np.mean(embeddings, axis=0)
    elif model == "pretrained":
        embeddings = []
        if len(document_tokens)<1:
            return np.zeros(pre_trained.vector_size)
        else:
            for token in document_tokens:
                if token in pre_trained.index_to_key:
                    embeddings.append(pre_trained.get_vector(token))
                else:
                    embeddings.append(np.random.rand(pre_trained.vector_size))
            # mean the vectors of individual words to get the vector of the document
            return np.mean(embeddings, axis=0)
    else:
        logging.error("Invalid embeding model name provided. Please enter word2vec or pretrained.")


def get_embeddings(allowLoad=True):

    logging.info("get_embeddings function execution started.")

    if allowLoad:
        if not os.path.exists(DOCUMENTS_EMBEDDING_WORD2VEC_DIR) or not os.path.exists(DOCUMENTS_EMBEDDING_PRETRAINED_DIR):
            logging.info("Embeding Word vectors could not be located.")
        else:
            logging.info("Embedings found in local directory.")
            logging.info("Loading Word2Vec from {}.".format(DOCUMENTS_EMBEDDING_WORD2VEC_DIR))

            pd_w2v = np.loadtxt(DOCUMENTS_EMBEDDING_WORD2VEC_DIR, delimiter=",")

            logging.info("Loading Pre-trained Word2Vec from {}.".format(DOCUMENTS_EMBEDDING_PRETRAINED_DIR))

            pd_pre = np.loadtxt(DOCUMENTS_EMBEDDING_PRETRAINED_DIR, delimiter=",")

            logging.info("Models loaded.")
            logging.info("get_embeddings function execution ended.")

            return (pd_w2v, pd_pre)

    (documents, _, _) = read_dataset(onlyColumn=True)
    
    documents = preprocess_documents(documents)
    
    documents_embedding_word2vec = []
    documents_embedding_pretrained = []
    
    for document in documents:
        documents_embedding_word2vec.append(embed(document, model="word2vec"))
        documents_embedding_pretrained.append(embed(document, model="pretrained"))

    logging.info("All documents embedded.")

    pd_w2v = pd.DataFrame(documents_embedding_word2vec)
    pd_pre = pd.DataFrame(documents_embedding_pretrained)

    logging.info("Ebmeddings saved to {} and {}.".format(DOCUMENTS_EMBEDDING_WORD2VEC_DIR, DOCUMENTS_EMBEDDING_PRETRAINED_DIR))

    pd_w2v.to_csv(DOCUMENTS_EMBEDDING_WORD2VEC_DIR, index=False, header=False)
    pd_pre.to_csv(DOCUMENTS_EMBEDDING_PRETRAINED_DIR, index=False, header=False)

    logging.info("get_embeddings function execution ended.")

    return (np.array(documents_embedding_word2vec), np.array(documents_embedding_pretrained))
