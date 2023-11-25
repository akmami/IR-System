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
    
    ensemble_weights = [(1.0, 0.0, 0.0, 0.0),
                        (0.0, 1.0, 0.0, 0.0),
                        (0.0, 0.0, 1.0, 0.0),
                        (0.0, 0.0, 0.0, 1.0),
                        (1.0, 1.0, 1.0, 1.0),
                        (6.0, 0.4, 0.0, 0.0),
                        (6.0, 0.3, 0.0, 0.0),
                        (6.0, 0.2, 0.0, 0.0),
                        (6.0, 0.1, 0.0, 0.0),
                        (6.0, -0.1, 0.0, 0.0),
                        (6.0, -0.2, 0.0, 0.0),
                        (6.0, -0.3, 0.0, 0.0),
                        (6.0, -0.4, 0.0, 0.0),
                        (6.0, 0.0, 0.4, 0.0),
                        (6.0, 0.0, 0.3, 0.0),
                        (6.0, 0.0, 0.2, 0.0),
                        (6.0, 0.0, 0.1, 0.0),
                        (6.0, 0.0, -0.1, 0.0),
                        (6.0, 0.0, -0.2, 0.0),
                        (6.0, 0.0, -0.3, 0.0),
                        (6.0, 0.0, -0.4, 0.0),
                        (6.0, 0.0, 0.0, 0.4),
                        (6.0, 0.0, 0.0, 0.3),
                        (6.0, 0.0, 0.0, 0.2),
                        (6.0, 0.0, 0.0, 0.1),
                        (6.0, 0.0, 0.0, -0.1),
                        (6.0, 0.0, 0.0, -0.2),
                        (6.0, 0.0, 0.0, -0.3),
                        (6.0, 0.0, 0.0, -0.4), # fundamental experiments done
                        (6.0, -0.4, 0.0, 0.1),
                        (6.0, -0.4, 0.0, 0.2),
                        (6.0, -0.4, 0.0, 0.3),
                        (6.0, -0.4, 0.1, 0.0),
                        (6.0, -0.4, 0.2, 0.0),
                        (6.0, -0.4, 0.3, 0.0),
                        (6.0, -0.4, 0.0, -0.1),
                        (6.0, -0.4, 0.0, -0.2),
                        (6.0, -0.4, 0.0, -0.3),
                        (6.0, -0.4, -0.1, 0.0),
                        (6.0, -0.4, -0.2, 0.0),
                        (6.0, -0.4, -0.3, 0.0)
                        (6.0, -0.2, 0.2, 0.4),
                        (6.0, 0.0, 0.2, 0.4),
                        (6.0, 0.2, 0.2, 0.4),
                        (6.0, -0.4, -0.2, 0.4),
                        (6.0, -0.4, 0.0, 0.4),
                        (6.0, -0.4, 0.2, 0.4)]
    
    for weights in ensemble_weights:

        query_results = query_all(weight_tfidf=weights[0], weight_bm25=weights[1], weight_w2v=weights[2], weight_pre=weights[3])
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

        print(weights)

        evaluate(logs=logging.WARN)


#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
# MARK: Main
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()