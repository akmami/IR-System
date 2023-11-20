import multiprocessing
import logging
from gensim.models import Word2Vec
from preprocess import preprocess

# Lower logging level
logging.getLogger().setLevel(logging.WARN)


def train():
    (documents, ground_truth, queries) = preprocess()
    
    cores = multiprocessing.cpu_count()
    
    # Create model
    w2v_model = Word2Vec(epochs=30, min_count=20, window=2, vector_size=10, sample=6e-5, alpha=0.03, min_alpha=0.0007, negative=20, workers=cores-1)
    
    # Build vocabluary.
    # This is new in Word3Vec 4.0.0
    w2v_model.build_vocab(documents)

    # Train model
    w2v_model.train(documents, total_examples=len(documents), epochs=5, report_delay=1)

    # save model
    w2v_model.save('model/word2vec.model')


train()