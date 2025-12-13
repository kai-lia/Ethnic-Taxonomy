import sys
import math
import numpy as np

from pathlib import Path
from collections import Counter

# from my code
from utils.cleaning_data.clean_data import run as cleaning_pipeline
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


def top_k_log_odds_terms(log_odds_dict, k=0, min_score=0.0):
    """return top-k terms with log-odds >= min_score
    result: most distinct words per word group"""
    if not isinstance(log_odds_dict, dict):
        return {}
    # sorting and getting the top terms of log odds
    scores = np.array(list(log_odds_dict.values()))
    median_score = np.median(scores)

    items = [
        (term, score) for term, score in log_odds_dict.items() if score >= median_score
    ]

    if k:
        items = sorted(items, key=lambda x: x[1], reverse=True)[:k]
    else:
        items = sorted(items, key=lambda x: x[1], reverse=True)

    return dict(items)


def weighted_log_odds(group_counter, global_counter, alpha=0.01):
    """Monroe et al. 2008 weighted log-odds with Dirichlet prior:
    counts_group: Counter of words in target group
    counts_global: Counter of words in all other groups
    """

    vocab = set(group_counter) | set(global_counter)
    V = len(vocab)

    # totals
    group_size = sum(group_counter.values())
    global_size = sum(global_counter.values())

    # alpha prior: proportional to total frequency
    alpha_vec = {w: alpha * global_counter.get(w, 0) for w in vocab}
    alpah_prior = sum(alpha_vec.values())

    scores = {}

    for word in vocab:
        # how many times word appears
        group_word_count = group_counter.get(word, 0)  # word appearance # in group
        global_word_count = global_counter.get(word, 0)  # word appearance # in global

        # posterior means
        grp_prb = (group_word_count + alpha_vec[word]) / (group_size + alpah_prior)
        glbl_prb = (global_word_count + alpha_vec[word]) / (global_size + alpah_prior)

        # weighted logodds
        delta = np.log(grp_prb / (1 - grp_prb)) - np.log(glbl_prb / (1 - glbl_prb))

        # variance gets rid of rare words
        var = 1 / (group_word_count + alpha_vec[word]) + 1 / (
            global_word_count + alpha_vec[word]
        )

        scores[word] = delta / math.sqrt(var)

    return scores


def run():
    if not DB_PATH.exists():
        # making sure clean is there
        cleaning_pipeline()

    ethnicity_df = fetch_duck_df(DB_PATH, "ethnicity_clean")

    # creating verb global, all verbs together
    global_verb_counter = combine_counts(ethnicity_df, "Verbs")
    # creating adj global, all adj together
    global_adj_counter = combine_counts(ethnicity_df, "Adjs")

    # getting regular log odds
    ethnicity_df["Verbs Log-Odds"] = ethnicity_df["Verbs"].apply(
        lambda x: weighted_log_odds(x, global_verb_counter)
    )
    ethnicity_df["Adjs Log-Odds"] = ethnicity_df["Adjs"].apply(
        lambda x: weighted_log_odds(x, global_adj_counter)
    )

    # top words for each ethnicity from log odds
    ethnicity_df["Top Adjs Log-Odds"] = ethnicity_df["Adjs Log-Odds"].apply(
        lambda x: top_k_log_odds_terms(x, 50)
    )
    ethnicity_df["Top Verbs Log-Odds"] = ethnicity_df["Verbs Log-Odds"].apply(
        lambda x: top_k_log_odds_terms(x, 50)
    )

    save_duck_df(DB_PATH, ethnicity_df, "ethnicity_log_odds")


if __name__ == "__main__":
    """run via the command python -m utils.log_odds.log_odds"""
    run()
