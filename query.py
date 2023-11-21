import logging
import numpy as np
import pandas as pd
import heapq
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from preprocess import read_dataset
from embeding import embed, get_embeddings


# Lower logging level to give info to user
logging.getLogger().setLevel(logging.INFO)

# Hyperparameters
WEIGHT_TFIDF = 0.4
WEIGHT_BM25 = 0.4
WEIGHT_W2V = 0.1
WEIGHTS_PRE = 0.1


# Get dataset
(documents, ground_truth, queries) = read_dataset(onlyColumn=True)

# Initialize TfidfVectorizer
tfidfvectorizer = TfidfVectorizer(analyzer='word',stop_words= 'english')
tfidf_wm = tfidfvectorizer.fit_transform(documents)

# Initialize BM25Okapi
tokenized_docs = [document.split(" ") for document in documents]
bm25_wm = BM25Okapi(tokenized_docs)

# Initialize word embedding documents
(word2vec, pretrained_wrod2vec) = get_embeddings()


#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
# MARK: Methods
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
def query(sentence, weight_tfidf=WEIGHT_TFIDF, weight_bm25=WEIGHT_BM25, weight_w2v=WEIGHT_W2V, weight_pre=WEIGHTS_PRE):
    # Scores will be calculated with cosine similarity function

    # Get scores from TF-IDF
    query_scores_tfidf = tfidfvectorizer.transform([sentence])
    query_scores_tfidf = cosine_similarity(query_scores_tfidf, tfidf_wm).flatten()

    # Get scores from BM25
    query_scores_bm25 = bm25_wm.get_scores(sentence)
    # Normalize scores
    query_scores_bm25 = query_scores_bm25 /  np.linalg.norm(query_scores_bm25)
    
    # Get scores from word embedings
    query_embedding_word2vec = embed(sentence, model="word2vec")
    query_embedding_pretrained = embed(sentence, model="pretrained")

    query_scores_embeddings_word2vec = []
    query_scores_embeddings_pretrained = []
    
    for embeding in word2vec:
        query_scores_embeddings_word2vec.append(cosine_for_1d(embeding, query_embedding_word2vec).flatten())
    
    for embeding in pretrained_wrod2vec:
        query_scores_embeddings_pretrained.append(cosine_for_1d(embeding, query_embedding_pretrained).flatten())
    
    query_scores_embeddings_word2vec = np.array(query_scores_embeddings_word2vec).flatten()
    query_scores_embeddings_pretrained = np.array(query_scores_embeddings_pretrained).flatten()

    # Combine scores with some weights
    # Update weights
    query_scores_tfidf = query_scores_tfidf * weight_tfidf
    query_scores_bm25 = query_scores_bm25 * weight_bm25
    query_scores_embeddings_word2vec = query_scores_embeddings_word2vec * weight_w2v
    query_scores_embeddings_pretrained = query_scores_embeddings_pretrained * weight_pre
    # Sum wieghts
    total_scores = np.add(query_scores_tfidf, query_scores_bm25)
    total_scores = np.add(total_scores, query_scores_embeddings_word2vec)
    total_scores = np.add(total_scores, query_scores_embeddings_pretrained)

    # Get top 10
    top10 = heapq.nlargest(10, np.ndenumerate(total_scores), key=lambda x: x[1])
    top10 = [ (index[0]+1, rank) for (index, rank) in top10 ]
    
    return(top10)


# query function for multiple queries
def query_all(weight_tfidf=WEIGHT_TFIDF, weight_bm25=WEIGHT_BM25, weight_w2v=WEIGHT_W2V, weight_pre=WEIGHTS_PRE):
    
    ranks = []

    for q in queries:
        ranks.append(query(q, weight_tfidf, weight_bm25, weight_w2v, weight_pre))
    
    return ranks


# cosine similarity function for vectors
def cosine_for_1d(vec1, vec2):
    return np.dot(vec1, vec2) / ( max(norm(vec1) * norm(vec2), 0.000000000000001) ) # for some cases, norms of vec1 or vec2 can be zero, which yields to division by zero


#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
# MARK: Main
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
if __name__ == "__main__":
    test_query = "What problems and concerns are there in making up descriptive titles? What difficulties are involved in automatically retrieving articles from approximate titles? What is the usual relevance of the content of articles to their titles?"
    query(test_query)