import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from scipy.stats import spearmanr

from utils.save_load import fetch_duck_df


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DB_PATH = Path("data/clean/ethnicity_clean.duckdb")
TABLE = "ethnicity_semantic"


def compute_correlations(df):
    """
    Correlate lexical inheritance with semantic drift.
    """
    pairs = [
        ("adj_region_mean", "semantic_drift_region", "Adj → Region"),
        ("verb_region_mean", "semantic_drift_region", "Verb → Region"),
        ("adj_race_mean", "semantic_drift_race", "Adj → Race"),
        ("verb_race_mean", "semantic_drift_race", "Verb → Race"),
    ]

    rows = []

    for x, y, label in pairs:
        if x not in df.columns or y not in df.columns:
            continue

        sub = df[[x, y]].dropna()
        if len(sub) < 5:
            continue

        rho = sub[x].corr(sub[y], method="spearman")
        pearson = sub[x].corr(sub[y], method="pearson")

        rows.append(
            {
                "comparison": label,
                "n": len(sub),
                "spearman_r": rho,
                "pearson_r": pearson,
            }
        )

    return pd.DataFrame(rows)


def add_pca_coords(df, emb_col="embedding", n_components=2):
    X = np.vstack(df[emb_col].values)
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(X)

    df = df.copy()
    df["pc1"] = coords[:, 0]
    df["pc2"] = coords[:, 1]
    return df, pca


def build_parent_lookup(df, parent_col):
    return {
        row[parent_col]: (row["pc1"], row["pc2"])
        for _, row in df.iterrows()
        if isinstance(row.get(parent_col), str)
    }


def plot_semantic_field(
    df,
    parent_col,
    title,
    out_path,
):
    """
    PCA semantic field with drift arrows:
    ethnicity → parent (region or race)
    """
    df = df.dropna(subset=["pc1", "pc2", parent_col])

    parent_lookup = build_parent_lookup(df, parent_col)

    plt.figure(figsize=(7, 6))

    # ethnicity points
    plt.scatter(
        df["pc1"],
        df["pc2"],
        alpha=0.6,
        s=40,
        label="Ethnicities",
    )

    # arrows
    for _, row in df.iterrows():
        parent = row[parent_col]
        if parent not in parent_lookup:
            continue

        px, py = parent_lookup[parent]

        plt.arrow(
            row["pc1"],
            row["pc2"],
            px - row["pc1"],
            py - row["pc2"],
            alpha=0.4,
            width=0.002,
            head_width=0.03,
            length_includes_head=True,
            color="gray",
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_inheritance_vs_drift(
    df,
    x_col,
    y_col,
    region_col="Region",
    title=None,
    out_path=None,
):
    """
    Scatter plot of lexical inheritance vs semantic drift.
    Each point = one ethnicity.
    Colored by region.
    """

    sub = df[[x_col, y_col, region_col]].dropna()

    if sub.empty:
        print("No data to plot.")
        return

    # spearman correlation
    rho, p = spearmanr(sub[x_col], sub[y_col])

    plt.figure(figsize=(6, 5))

    for region, g in sub.groupby(region_col):
        plt.scatter(
            g[x_col],
            g[y_col],
            alpha=0.75,
            label=region,
        )

    # median reference lines
    plt.axvline(sub[x_col].median(), linestyle="--", alpha=0.3)
    plt.axhline(sub[y_col].median(), linestyle="--", alpha=0.3)

    # labels (FIXED, not passed in)
    plt.xlabel("Adjective Signature Inheritance (Region)")
    plt.ylabel("Semantic Drift from Region")

    if title:
        plt.title(title)

    # atats annotation
    plt.text(
        0.02,
        0.98,
        f"Spearman ρ = {rho:.2f}\np = {p:.3f}",
        transform=plt.gca().transAxes,
        ha="left",
        va="top",
    )

    plt.legend(title="Region", fontsize=8)
    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300)
        plt.close()
    else:
        plt.show()


def main():

    df_sem = fetch_duck_df(DB_PATH, "ethnicity_semantic")
    df_inh = fetch_duck_df(DB_PATH, "ethnicity_inheritance")  # or whatever you named it

    df = df_sem.merge(
        df_inh,
        on="Ethnicity",
        how="inner",
        suffixes=("", "_inherit"),
    )

    print("Columns in merged df:")
    print(sorted(df.columns))

    corr_df = compute_correlations(df)
    print("\nSemantic Drift ↔ Lexical Inheritance (Spearman):")
    print(corr_df.to_string(index=False))

    df, _ = add_pca_coords(df)

    plot_semantic_field(
        df,
        parent_col="Region",
        title="Semantic Drift Field: Ethnicity → Region",
        out_path="figures/semantic_field_region.png",
    )
    print("hit display save")
    plot_semantic_field(
        df,
        parent_col="Race",
        title="Semantic Drift Field: Ethnicity → Race",
        out_path="figures/semantic_field_race.png",
    )
    plot_inheritance_vs_drift(
        df=df,
        x_col="adj_region_mean",
        y_col="semantic_drift_region",
        title="Lexical Inheritance vs Semantic Drift (Adjectives)",
        out_path="figures/drift/adj_region_inheritance_vs_drift.png",
    )

    plot_inheritance_vs_drift(
        df,
        x_col="verb_region_mean",
        y_col="semantic_drift_region",
        region_col="Region",
        title="Lexical Inheritance vs Semantic Drift (Verbs)",
        out_path="figures/drift/verb_inheritance_vs_drift.png",
    )

    print("final")


if __name__ == "__main__":
    """
    Run via:
    python -m utils.semantic_drift.analysis
    """
    main()
