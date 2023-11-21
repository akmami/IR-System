import logging
import os
import numpy as np
import pandas as pd
import csv
from gensim.models import KeyedVectors
from preprocess import read_dataset, preprocess_documents


# Lower logging level
logging.getLogger().setLevel(logging.INFO)


# Get current directory
cwd = os.getcwd()


# Define embeding vectors' directory 
DOCUMENTS_EMBEDDING_WORD2VEC_DIR = os.path.join(cwd, "dataset", "documents_embedding_word2vec.csv")
DOCUMENTS_EMBEDDING_PRETRAINED_DIR = os.path.join(cwd, "dataset", "documents_embedding_pretrained.csv")


word2vec = KeyedVectors.load_word2vec_format("model/word2vec.model", binary=True)
pre_trained = KeyedVectors.load_word2vec_format("model/word2vec-google-news-300/word2vec-google-news-300.gz", binary=True)


# Lower logging level
logging.getLogger().setLevel(logging.INFO)


# test_document = """Editions of the Dewey Decimal Classifications,"Comaromi, J.P.","   The present study is a history of the DEWEY Decimal Classification.  The first edition of the DDC was published in 1876, the eighteenth edition in 1971, and future editions will continue to appear as needed.  In spite of the DDC's long and healthy life, however, its full story has never been told.  There have been biographies of Dewey that briefly describe his system, but this is the first attempt to provide a detailed history of the work that more than any other has spurred the growth of librarianship in this country and abroad."""


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
    
    if allowLoad:
        if not os.path.exists(DOCUMENTS_EMBEDDING_WORD2VEC_DIR) or not os.path.exists(DOCUMENTS_EMBEDDING_PRETRAINED_DIR):
            logging.info("Embeding Word vectors could not be located.")
        else:
            pd_w2v = np.loadtxt(DOCUMENTS_EMBEDDING_WORD2VEC_DIR, delimiter=",")
            pd_pre = np.loadtxt(DOCUMENTS_EMBEDDING_PRETRAINED_DIR, delimiter=",")

            return (pd_w2v, pd_pre)

    (documents, _, _) = read_dataset(onlyColumn=True)
    
    documents = preprocess_documents(documents)
    
    documents_embedding_word2vec = []
    documents_embedding_pretrained = []
    
    for document in documents:
        documents_embedding_word2vec.append(embed(document, model="word2vec"))
        documents_embedding_pretrained.append(embed(document, model="pretrained"))

    logging.info("Embeddings retrieved")

    pd_w2v = pd.DataFrame(documents_embedding_word2vec)
    pd_pre = pd.DataFrame(documents_embedding_pretrained)

    pd_w2v.to_csv(DOCUMENTS_EMBEDDING_WORD2VEC_DIR, index=False, header=False)
    pd_pre.to_csv(DOCUMENTS_EMBEDDING_PRETRAINED_DIR, index=False, header=False)

    return (np.array(documents_embedding_word2vec), np.array(documents_embedding_pretrained))