import logging
from gensim.models import Word2Vec

word2vec = Word2Vec.load("model/word2vec.model")
pre_trained = Word2Vec.loac("model/word2vec-google-news-300/word2vec-google-news-300.gz")


def embed(sentence, model="word2vec"):
    if model == "word2vec":
        pass
    elif model == "pre_trained":
        pass
    else:
        logging.error("Invalid embeding model name provided. Please enter word2vec or pretrained.")
    