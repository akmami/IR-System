import os
import logging
import pandas as pd


# Lower logging level
logging.getLogger().setLevel(logging.INFO)


#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
# MARK: Methods
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
def main():

    from download import download
    
    # Download dataset
    # If dataset is already installed, then the command will be ignored
    download()

    from train import train
    
    train(check=True)

    from query import query_all

    query_results = query_all(weight_tfidf=0, weight_bm25=1, weight_w2v=0, weight_pre=0)
    docs = [[result[0] for result in query_result] for query_result in query_results]

    cwd = os.getcwd()
    QUERY_RESULTS = os.path.join(cwd, "query_results.csv")
    DOCS = os.path.join(cwd, "docs.csv")

    # save ranks
    pd_query_results = pd.DataFrame(query_results)
    pd_query_results.to_csv(QUERY_RESULTS, index=False, header=False)
    pd_docs = pd.DataFrame(docs)
    pd_docs.to_csv(DOCS, index=False, header=False)

    from evaluation import evaluate

    evaluate()


#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
# MARK: Main
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()