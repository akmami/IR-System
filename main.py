import os
import logging
import pandas as pd
from download import download
from train import train
from query import query_all

# Lower logging level
logging.getLogger().setLevel(logging.INFO)


#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
# MARK: Methods
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
def main():

    # Download dataset
    # If dataset is already installed, then the command will be ignored
    download()

    train(check=True)

    ranks = query_all()

    cwd = os.getcwd()
    RANKS = os.path.join(cwd, "ranks.csv")

    # save ranks
    pd_ranks = pd.DataFrame(ranks)
    pd_ranks.to_csv(RANKS, index=False, header=False)


#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
# MARK: Main
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()