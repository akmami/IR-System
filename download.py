import logging
import os
import shutil
import gensim.downloader as api


# Lower logging level to give info to user
logging.getLogger().setLevel(logging.INFO)

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
# MARK: Methods
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
def download():

    logging.info("download function execution started.")

    cwd = os.getcwd()
    exists = os.path.exists(os.path.join(cwd, "model", "word2vec-google-news-300", "word2vec-google-news-300.gz"))

    if exists:
        logging.info("Word2Vec dataset already installed.")
        logging.info("download function execution ended.")
        
        return

    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    
    os.environ["GENSIM_DATA_DIR"] = cwd

    model_path = api.load("word2vec-google-news-300", return_path=True)
    destination_path = os.path.join(cwd, "model")

    logging.info("Model downloaded to {}".format(model_path))
    logging.info("Moving model to {}/model/".format(cwd))

    if not os.path.exists(destination_path):
        os.mkdir(destination_path)

    if not os.path.exists(os.path.join(destination_path, "word2vec-google-news-300")):
        os.mkdir(os.path.join(destination_path, "word2vec-google-news-300"))

    shutil.copy(model_path, os.path.join(destination_path, "word2vec-google-news-300", "word2vec-google-news-300.gz"))
    shutil.copy(os.path.join(os.path.abspath(os.path.join(model_path, "..", "..")), "information.json"), destination_path)

    logging.info("Model moved. Deleting initial directory of download started.")

    shutil.rmtree(os.path.abspath(os.path.join(model_path, "..", "..")))

    logging.info("Directory {} deleted.".format(os.path.abspath(os.path.join(model_path, "..", ".."))))
    logging.info("download function execution ended.")


#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
# MARK: Main
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
if __name__ == "__main__":
    download()