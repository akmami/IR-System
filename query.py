import numpy as np
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import read_dataset


(documents, _, _) = read_dataset(onlyColumn=True, preprocess=False)
tfidfvectorizer = TfidfVectorizer(analyzer='word',stop_words= 'english')
tfidf_wm = tfidfvectorizer.fit_transform(documents).todense()
names = tfidfvectorizer.get_feature_names_out()

WEIGHT_TFIDF = 0.4
WEIGHT_BM25 = 0.4
WEIGHT_WE = 0.2

def query(sentence):
    # Scores will be calculated with cosine similarity function

    # TODO: get scores from TF-IDF
    print(tfidf_wm)
    print(names)

    # TODO: get scores from BM25

    # TODO: get scores from word embedings

    # Combine scores with some weights

    # Get top 10

    pass


def cosine(vec1, vec2):
    return np.dot(vec1, vec2) / ( norm(vec1) * norm(vec2) )