import sys
import math
import pandas as pd
import matplotlib.pyplot as plt


from pathlib import Path
from collections import Counter
from utils.save_load import fetch_duck_df

# setting path root for reading or writing data
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# path to access
DB_PATH = Path("data/clean/ethnicity_clean.duckdb")


def shannon_entropy(counts: dict) -> float | None:
    """Compute Shannon entropy from a {term: count} dict."""
    # no shared term
    if not isinstance(counts, dict) or len(counts) == 0:
        return None

    total = sum(counts.values())  # total amount of words
    if total == 0:
        return None

    entropy = 0.0
    for count in counts.values():
        prob = count / total
        entropy -= prob * math.log2(prob)
    return entropy


def compute_entropy_by_node(
    df,
    node_col="Ethnicity",
    adj_col="Adjs",
    verb_col="Verbs",
):
    """entropy across all terms"""
    rows = []

    for _, row in df.iterrows():
        rows.append(
            {
                "node": row["Ethnicity"],
                "adj_entropy": shannon_entropy(row["Adjs"]),
                "verb_entropy": shannon_entropy(row["Verbs"]),
                "adj_vocab_size": (
                    len(row[adj_col]) if isinstance(row["Adjs"], dict) else 0
                ),
                "verb_vocab_size": (
                    len(row["Verbs"]) if isinstance(row["Verbs"], dict) else 0
                ),
            }
        )

    return pd.DataFrame(rows)


def infer_taxonomy_level(row):
    """returns the level on taxonomical hiearchy belongs to"""
    # race is at the highest level of the taxonomy so none above
    if pd.isna(row["Race"]) and pd.isna(row["Region"]):
        return "Race"
    # region is the second level and has a race above it
    if pd.notna(row["Race"]) and pd.isna(row["Region"]):
        return "Region"
    # ethnicity is at lowest level so has both above
    return "Ethnicity"


def visualize_entropy(summary):
    """ "
    plots race region and ethnicitys entropy
    saves: in figures file
    note:(not creatign a separate moduel because it is only one function)"""
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

    plt.savefig("figures/taxonomic_entropy_gradient.png", dpi=300, bbox_inches="tight")


def run():
    # load file
    ethnicity_df = fetch_duck_df(DB_PATH, "ethnicity_log_odds")

    entropy_df = compute_entropy_by_node(ethnicity_df)

    # adding level
    entropy_df["level"] = ethnicity_df.apply(infer_taxonomy_level, axis=1)
    # removing low counts
    entropy_df = entropy_df[
        (entropy_df["adj_vocab_size"] >= 10) & (entropy_df["verb_vocab_size"] >= 10)
    ]
    # for each level of taxonomy, calc entropy
    summary = (
        entropy_df.groupby("level")[["adj_entropy", "verb_entropy"]]
        .mean()
        .reset_index()
    )
    visualize_entropy(summary)


if __name__ == "__main__":
    """run via the command python -m utils.entropy.entropy"""
    run()
