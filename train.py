import os
import multiprocessing
import logging
from gensim.models import Word2Vec
from preprocess import preprocess


# Lower logging level to give info to user
logging.getLogger().setLevel(logging.INFO)

# get current directory
cwd = os.getcwd()

WORD2VEC_MODEL_PATH = os.path.join(cwd, "model", "word2vec.model")


#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
# MARK: Methods
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
def train(check=False):

    logging.info("train function execution started.")

    (documents, _, _) = preprocess(onlyColumn=True)
    
    cores = multiprocessing.cpu_count()
    
    if check and os.path.exists(WORD2VEC_MODEL_PATH):
        logging.info("Word2Vec model exists.")
        logging.info("train function execution ended.")
        return
    
    # Increase logging level to prevent gensim logs
    logging.getLogger().setLevel(logging.WARN)

    # Create model
    w2v_model = Word2Vec(epochs=10, min_count=20, window=2, vector_size=100, sample=6e-5, alpha=0.03, min_alpha=0.0007, negative=20, workers=cores-1)
    
    # Build vocabluary. This is new in Word3Vec 4.0.0
    w2v_model.build_vocab(documents)

    # Train model
    w2v_model.train(documents, total_examples=len(documents), epochs=5, report_delay=1)

    term_count = len(w2v_model.wv.key_to_index)

    # Lower logging level to give info to user
    logging.getLogger().setLevel(logging.INFO)
    
    # Keys
    logging.info("Terms count: {}".format(term_count))

    # save model
    w2v_model.wv.save_word2vec_format(WORD2VEC_MODEL_PATH, binary=True)

    logging.info("train function execution ended.")


#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
# MARK: Main
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
if __name__ == "__main__":
    train(check=True)