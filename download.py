import logging
import os
import shutil
import gensim.downloader as api

cwd = os.getcwd()
exists = os.path.exists(os.path.join(cwd, "model", "word2vec-google-news-300", "word2vec-google-news-300.gz"))

if exists:
    logging.error("Word2Vec dataset already installed.")
    exit(0)

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

shutil.rmtree(os.path.abspath(os.path.join(model_path, "..", "..")))