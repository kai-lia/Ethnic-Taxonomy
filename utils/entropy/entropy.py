import matplotlib.pyplot as plt
import math
import pickle
import sys
from pathlib import Path
from collections import Counter
import pandas as pd 
import ast
import duckdb

from utils.save_load import fetch_duck_df


# setting to main 
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


DB_PATH = Path("data/clean/ethnicity_clean.duckdb")


def shannon_entropy(counts: dict) -> float | None:
    """Ccmpute Shannon entropy from a {term: count} dict."""
    if not isinstance(counts, dict) or len(counts) == 0:
        return None

    total = sum(counts.values())
    if total == 0:
        return None

    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log2(p)

    return entropy



def compute_entropy_by_node(
    df,
    node_col="Ethnicity",
    adj_col="Adjs",
    verb_col="Verbs",
):
    rows = []

    for _, row in df.iterrows():
        rows.append({
            "node": row[node_col],
            "adj_entropy": shannon_entropy(row[adj_col]),
            "verb_entropy": shannon_entropy(row[verb_col]),
            "adj_vocab_size": len(row[adj_col]) if isinstance(row[adj_col], dict) else 0,
            "verb_vocab_size": len(row[verb_col]) if isinstance(row[verb_col], dict) else 0,
        })

    return pd.DataFrame(rows)



def infer_taxonomy_level(row):
    if pd.isna(row["Race"]) and pd.isna(row["Region"]):
        return "Race"
    if pd.notna(row["Race"]) and pd.isna(row["Region"]):
        return "Region"
    return "Ethnicity"



def visualize_entropy(summary):
    order = ["Race", "Region", "Ethnicity"]
    summary = summary.set_index("level").loc[order].reset_index()

    Path("figures").mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))

    plt.plot(summary["level"], summary["adj_entropy"], marker="o", label="Adjectives")
    plt.plot(summary["level"], summary["verb_entropy"], marker="o", label="Verbs")

    plt.ylabel("Mean Shannon Entropy")
    plt.xlabel("Taxonomy Level")
    plt.title("Taxonomic Entropy Gradient")

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "figures/taxonomic_entropy_gradient.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


def main(): 
    ethnicity_df = fetch_duck_df(DB_PATH, "ethnicity_log_odds")

    entropy_df = compute_entropy_by_node(ethnicity_df)
    entropy_df["level"] = ethnicity_df.apply(infer_taxonomy_level, axis=1)
    entropy_df = entropy_df[
    (entropy_df["adj_vocab_size"] >= 10) &
    (entropy_df["verb_vocab_size"] >= 10)]
    summary = (
        entropy_df
        .groupby("level")[["adj_entropy", "verb_entropy"]]
        .mean()
        .reset_index())
    visualize_entropy(summary)


if __name__ == "__main__":
    """ run via the command python -m utils.entropy.entropy"""
    main()
   
