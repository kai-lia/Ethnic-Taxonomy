from utils.cleaning_data.clean_data import run as clean_data
from utils.log_odds.log_odds import run as log_odds
from utils.entropy.entropy import run as entropy
from utils.inheritance.run_inheritance import run as inheritance
from utils.semantic_drift.embeddings import run as embeddings
from utils.semantic_drift.analysis import run as analysis


def run():
    # basic cleaning and data prep
    print("Prepping data ================")
    clean_data()
    log_odds()

    # calc entropy
    print("Calculating Entropy ================")
    entropy()

    # calc inheritance
    ("Calculating Inheritance ================")
    inheritance()

    ("Calculating Drift ================")
    print("This will run embeddings and take 10-30 minutes")
    ans = input("Run embeddings? (y/n): ").lower()
    if ans == "y":
        embeddings()
        analysis()
    print("done")


if __name__ == "__main__":
    """execute all analysis run python -m utils.main
    WARNING: there will be a popup"""
    run()
