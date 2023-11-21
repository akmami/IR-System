import multiprocessing
import logging
from gensim.models import Word2Vec
from preprocess import preprocess


# Lower logging level
logging.getLogger().setLevel(logging.WARN)


def train():
    (documents, _, _) = preprocess(onlyColumn=True, preprocess=True)
    
    cores = multiprocessing.cpu_count()
    
    # Create model
    w2v_model = Word2Vec(epochs=10, min_count=20, window=2, vector_size=100, sample=6e-5, alpha=0.03, min_alpha=0.0007, negative=20, workers=cores-1)
    
    # Build vocabluary.
    # This is new in Word3Vec 4.0.0
    w2v_model.build_vocab(documents)

    # Train model
    w2v_model.train(documents, total_examples=len(documents), epochs=5, report_delay=1)

    # Lower logging level to give info to user
    logging.getLogger().setLevel(logging.INFO)
    
    # Keys
    logging.info("Terms count: {}".format(len(w2v_model.wv.key_to_index)))

    # save model
    w2v_model.wv.save_word2vec_format("model/word2vec.model", binary=True)

    # Change logging level
    logging.getLogger().setLevel(logging.WARN)


train()