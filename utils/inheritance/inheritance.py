import sys
import random
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from utils.save_load import fetch_duck_df



ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DB_PATH = Path("data/clean/ethnicity_clean.duckdb")



def top_log_odds_terms(d, k=20, min_score=0.0):
    """return top-k terms by log-odds score."""
    if not isinstance(d, dict):
        return []

    return [
        term for term, score in
        sorted(d.items(), key=lambda x: x[1], reverse=True)
        if score >= min_score
    ][:k]


def bootstrap_inheritance(child_terms, parent_terms, n_boot=1000, seed=42):
    """bootstrap overlap between child and parent signature sets"""
    if not child_terms:
        return None, None, None

    rng = random.Random(seed)
    parent_terms = set(parent_terms)

    vals = []
    for _ in range(n_boot):
        sample = [rng.choice(child_terms) for _ in range(len(child_terms))]
        vals.append(len(set(sample) & parent_terms) / len(sample))

    s = pd.Series(vals)
    return s.mean(), s.quantile(0.025), s.quantile(0.975)


def build_lookup(df, key="Ethnicity"):
    """row lookup by ethnicity label"""
    return {
        row[key]: row
        for _, row in df.iterrows()
        if pd.notna(row[key])
    }


def get_child_terms(row, k, min_score):
    """extract top child adjective and verbs """
    return {
        "adj": top_log_odds_terms(row["Adjs Log-Odds"], k, min_score),
        "verb": top_log_odds_terms(row["Verbs Log-Odds"], k, min_score),
    }


def inheritance_at_level(
    child_terms,
    parent_label,
    lookup,
    k,
    min_score,
    n_boot,
):
    """compute inheritance stats against a parent level."""
    if pd.isna(parent_label) or parent_label not in lookup:
        return {
            "adj": (None, None, None),
            "verb": (None, None, None),
        }

    parent = lookup[parent_label]

    parent_adj = top_log_odds_terms(parent["Adjs Log-Odds"], k, min_score)
    parent_verb = top_log_odds_terms(parent["Verbs Log-Odds"], k, min_score)

    return {
        "adj": bootstrap_inheritance(child_terms["adj"], parent_adj, n_boot),
        "verb": bootstrap_inheritance(child_terms["verb"], parent_verb, n_boot),
    }


def unpack_stats(prefix, stats):
    """make mean, lo, hi into flat columns."""
    mean, lo, hi = stats
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_lo": lo,
        f"{prefix}_hi": hi,
    }



def inheritance_with_bootstrap(
    df,
    k=20,
    min_score=0.0,
    n_boot=1000,
): 
    """" calculated the inclustion, and calcs a bootstrap for randomization"""
    rows = []
    lookup = build_lookup(df)

    for _, row in df.iterrows():
        eth = row["Ethnicity"]
        region = row["Region"]
        race = row["Race"]

        child_terms = get_child_terms(row, k, min_score)

        region_stats = inheritance_at_level(
            child_terms, region, lookup, k, min_score, n_boot
        )
        race_stats = inheritance_at_level(
            child_terms, race, lookup, k, min_score, n_boot
        )

        rows.append({
            "Ethnicity": eth,
            "Region": region,
            "Race": race,

            # region
            **unpack_stats("adj_region", region_stats["adj"]),
            **unpack_stats("verb_region", region_stats["verb"]),

            # race
            **unpack_stats("adj_race", race_stats["adj"]),
            **unpack_stats("verb_race", race_stats["verb"]),
        })

    return pd.DataFrame(rows)



def plot_by_parent(df, parent_col, mean_cols, lo_cols, hi_cols, out_dir, title_prefix):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for parent, sub in df.groupby(parent_col):
        sub = sub.dropna(subset=mean_cols + lo_cols + hi_cols)
        if len(sub) < 3:
            continue

        sub = sub.assign(
            diff=(sub[mean_cols[0]] - sub[mean_cols[1]]).abs()
        ).sort_values("diff", ascending=False)

        y = range(len(sub))
        plt.figure(figsize=(6, max(3, 0.35 * len(sub))))

        # adjectives
        plt.hlines(y, sub[lo_cols[0]], sub[hi_cols[0]], alpha=0.4)
        plt.scatter(sub[mean_cols[0]], y, label="Adjectives", zorder=3)

        # verbs
        plt.hlines(y, sub[lo_cols[1]], sub[hi_cols[1]], alpha=0.4)
        plt.scatter(sub[mean_cols[1]], y, label="Verbs", zorder=3)

        plt.yticks(y, sub["Ethnicity"])
        plt.xlabel("Signature inheritance")
        plt.title(f"{title_prefix}: {parent}")
        plt.legend()
        plt.tight_layout()

        fname = parent.lower().replace(" ", "_")
        plt.savefig(
            f"{out_dir}/inheritance_{fname}.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()




def main():
    """ plotting my inheritance"""
    df = fetch_duck_df(DB_PATH, "ethnicity_log_odds")

    inherit_df = inheritance_with_bootstrap(
        df,
        k=15,
        min_score=0.0,
        n_boot=1000,
    )

    plot_by_parent(
        inherit_df,
        parent_col="Region",
        mean_cols=["adj_region_mean", "verb_region_mean"],
        lo_cols=["adj_region_lo", "verb_region_lo"],
        hi_cols=["adj_region_hi", "verb_region_hi"],
        out_dir="figures/regions",
        title_prefix="Region Level Inheritance",
    )

    plot_by_parent(
        inherit_df,
        parent_col="Race",
        mean_cols=["adj_race_mean", "verb_race_mean"],
        lo_cols=["adj_race_lo", "verb_race_lo"],
        hi_cols=["adj_race_hi", "verb_race_hi"],
        out_dir="figures/races",
        title_prefix="Race Level Inheritance",
    )


if __name__ == "__main__":
    """Run via: python -m utils.inheritance.inheritance"""

    main()
