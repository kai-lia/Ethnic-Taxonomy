import sys
import random
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


from utils.save_load import fetch_duck_df



ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DB_PATH = Path("data/clean/ethnicity_clean.duckdb")


def bootstrap_inheritance(child_terms, parent_terms, n_boot=1000, seed=42):
    """bootstrap overlap between child and parent signature sets"""
    if not child_terms:
        return None, None, None

    child_terms = list(child_terms)  
    parent_terms = set(parent_terms)

    rng = random.Random(seed)
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

def get_signature_terms(row):
    """ extract precomputed signature sets (done in log odds func)"""
    adj = row.get("Top Adjs Log-Odds", {})
    verb = row.get("Top Verbs Log-Odds", {})

    return {
        "adj": set(adj.keys()) if isinstance(adj, dict) else set(),
        "verb": set(verb.keys()) if isinstance(verb, dict) else set(),
    }


def unpack_stats(prefix, stats):
    """make mean, lo, hi into flat columns"""
    mean, lo, hi = stats
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_lo": lo,
        f"{prefix}_hi": hi,
    }


def inheritance_with_bootstrap(df, n_boot=1000):
    rows = []
    lookup = build_lookup(df)

    for _, row in df.iterrows():
        eth = row["Ethnicity"]
        region = row["Region"]
        race = row["Race"]

        child_terms = get_signature_terms(row)

        # Region
        if pd.notna(region) and region in lookup:
            parent_terms = get_signature_terms(lookup[region])

            adj_region = bootstrap_inheritance(
                child_terms["adj"], parent_terms["adj"], n_boot
            )
            verb_region = bootstrap_inheritance(
                child_terms["verb"], parent_terms["verb"], n_boot
            )
        else:
            adj_region = verb_region = (None, None, None)

        # Race
        if pd.notna(race) and race in lookup:
            parent_terms = get_signature_terms(lookup[race])

            adj_race = bootstrap_inheritance(
                child_terms["adj"], parent_terms["adj"], n_boot
            )
            verb_race = bootstrap_inheritance(
                child_terms["verb"], parent_terms["verb"], n_boot
            )
        else:
            adj_race = verb_race = (None, None, None)

        rows.append({
            "Ethnicity": eth,
            "Region": region,
            "Race": race,

            # region
            **unpack_stats("adj_region", adj_region),
            **unpack_stats("verb_region", verb_region),

            # race
            **unpack_stats("adj_race", adj_race),
            **unpack_stats("verb_race", verb_race),
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





def plot_two_panel_inheritance(df, parent_col, adj_cols, verb_cols, out_dir, title_prefix, min_n=3,):
    """ Two pannel plots
    - panel left: adjective inheritance (mean + CI)
    - panel right: verb inheritance (mean + CI)
    """

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for parent, sub in df.groupby(parent_col):
        sub = sub.dropna(subset=list(adj_cols + verb_cols))
        if len(sub) < min_n:
            continue

        # sort by adjective mean (stable + intuitive)
        sub = sub.sort_values(adj_cols[0])

        y = range(len(sub))

        fig, axes = plt.subplots(ncols=2, sharey=True, figsize=(9, max(3, 0.35 * len(sub))))

        # adjs plot creation
        ax = axes[0]
        ax.hlines(y,sub[adj_cols[1]],sub[adj_cols[2]], color="tab:blue", alpha=0.5)
        ax.scatter(sub[adj_cols[0]], y, color="tab:blue", zorder=3)
        ax.set_title("Adjective inheritance")
        ax.set_xlabel("Inheritance")
        ax.set_yticks(y)
        ax.set_yticklabels(sub["Ethnicity"])

       # verbs plot creation
        ax = axes[1]
        ax.hlines(y, sub[verb_cols[1]], sub[verb_cols[2]], color="tab:orange", alpha=0.5)
        ax.scatter(sub[verb_cols[0]], y, color="tab:orange", zorder=3)
        ax.set_title("Verb inheritance")
        ax.set_xlabel("Inheritance")

        # shared format
        fig.suptitle(f"{title_prefix}: {parent}", y=1.02)
        plt.tight_layout()

        fname = parent.lower().replace(" ", "_")
        plt.savefig(
            f"{out_dir}/inheritance_{fname}.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close(fig)




def main():
    df = fetch_duck_df(DB_PATH, "ethnicity_log_odds")

    REQUIRED = {
        "Top Adjs Log-Odds",
        "Top Verbs Log-Odds",
        "Region",
        "Race",
    }
    assert REQUIRED.issubset(df.columns), df.columns

    inherit_df = inheritance_with_bootstrap(
        df,
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
    plot_two_panel_inheritance(
    inherit_df,
    parent_col="Region",
    adj_cols=("adj_region_mean", "adj_region_lo", "adj_region_hi"),
    verb_cols=("verb_region_mean", "verb_region_lo", "verb_region_hi"),
    out_dir="figures/regions",
    title_prefix="Region-level signature inheritance",
    )

    plot_two_panel_inheritance(
        inherit_df,
        parent_col="Race",
        adj_cols=("adj_race_mean", "adj_race_lo", "adj_race_hi"),
        verb_cols=("verb_race_mean", "verb_race_lo", "verb_race_hi"),
        out_dir="figures/races",
        title_prefix="Race-level signature inheritance",
    )



if __name__ == "__main__":
    """Run via: python -m utils.inheritance.inheritance"""

    main()
