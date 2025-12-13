import pickle
import sys
from pathlib import Path
from collections import Counter




import numpy as np
from collections import Counter
import math

from utils.cleaning_data.clean_data import main as cleaning_pipeline
from utils.save_load import fetch_duck_df, save_duck_df


# setting to main 
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# paths to data
DB_PATH = Path("data/clean/ethnicity_clean.duckdb")


# combining counts for log odds calc
def combine_counts(ethnicity_df, col):
    """takes a df and a column name (adjs, verbs, nouns) and creates a merged Counter"""
    merged = Counter()
    for ctr in ethnicity_df[col]:
        merged.update(ctr)

    print(f"global {col} counter created with {len(merged)} unique {col}")
    return merged

def top_k_log_odds_terms(log_odds_dict, k=20, min_score=0.0):
    """return top-k terms with log-odds >= min_score"""
    if not isinstance(log_odds_dict, dict):
        return {}

    items = [
        (term, score)
        for term, score in log_odds_dict.items()
        if score >= min_score
    ]

    items = sorted(items, key=lambda x: x[1], reverse=True)[:k]

    return dict(items)


def weighted_log_odds(counts_group, counts_global, alpha=0.01):
    """ Monroe et al. 2008 weighted log-odds with Dirichlet prior:
    counts_group: Counter of words in target group
    counts_global: Counter of words in all other groups
    """

    vocab = set(counts_group)|set(counts_global)
    V = len(vocab)

    # totals
    N1 = sum(counts_group.values())
    N2 = sum(counts_global.values())

    # alpha prior: proportional to total frequency
    alpha_vec = {w: alpha*counts_global.get(w, 0) for w in vocab}
    A0 = sum(alpha_vec.values())

    scores = {}

    for w in vocab:
        c1 = counts_group.get(w, 0)
        c2 = counts_global.get(w, 0)

        # posterior means
        p1 = (c1 + alpha_vec[w]) / (N1 + A0)
        p2 = (c2 + alpha_vec[w]) / (N2 + A0)

        # weighted logodds
        delta = np.log(p1 / (1 - p1)) - np.log(p2 / (1 - p2))

        # variance gets rid of rare words
        var = 1/(c1 + alpha_vec[w]) + 1/(c2 + alpha_vec[w])

        scores[w] = delta / math.sqrt(var)

    return scores



if __name__ == "__main__":
    """ run via the command python -m utils.log_odds.log_odds"""
    if not DB_PATH.exists():
        # making sure clean is there
        cleaning_pipeline()


    ethnicity_df = fetch_duck_df(DB_PATH, "ethnicity_clean")

    global_verb_counter = combine_counts(ethnicity_df, "Verbs") # creating verb global
    global_adj_counter = combine_counts(ethnicity_df, "Adjs") # creating adj global

    # getting regular log odds
    ethnicity_df['Verbs Log-Odds'] = ethnicity_df['Verbs'].apply(lambda x: weighted_log_odds(x, global_verb_counter))
    ethnicity_df['Adjs Log-Odds'] = ethnicity_df['Adjs'].apply(lambda x: weighted_log_odds(x, global_adj_counter))

    # top words for each ethnicity, a
    ethnicity_df["Top Adjs Log-Odds"] = ethnicity_df["Adjs Log-Odds"].apply(lambda x: top_k_log_odds_terms(x))
    ethnicity_df["Top Verbs Log-Odds"] = ethnicity_df["Verbs Log-Odds"].apply(lambda x: top_k_log_odds_terms(x))

    
    save_duck_df(DB_PATH, ethnicity_df, "ethnicity_log_odds")

    
