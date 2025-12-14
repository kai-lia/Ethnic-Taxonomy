import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score

# my own functs
from utils.save_load import fetch_duck_df


# for loading in
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DB_PATH = Path("data/clean/ethnicity_clean.duckdb")
TABLE = "ethnicity_semantic"


def compute_correlations(df):
    """compute correlations between lexical inheritance and semantic drift
    output: df
    one row per comparison including:
        - comparison label
        - sample size
        - Spearman rho
        - Pearson r
    """
    pairs = [
        ("adj_region_mean", "semantic_drift_region", "Adj to Region"),
        ("verb_region_mean", "semantic_drift_region", "Verb to Region"),
        ("adj_race_mean", "semantic_drift_race", "Adj to Race"),
        ("verb_race_mean", "semantic_drift_race", "Verb to Race"),
    ]

    rows = []

    for x, y, label in pairs:
        if x not in df.columns or y not in df.columns:
            continue

        sub = df[[x, y]].dropna()

        if len(sub) < 5:
            continue

        rows.append(
            {
                "comparison": label,
                "n": len(sub),
                "spearman_r": sub[x].corr(sub[y], method="spearman"),
                "pearson_r": sub[x].corr(sub[y], method="pearson"),
            }
        )

    return pd.DataFrame(rows)


def add_pca_coords(df, emb_col="embedding", n_components=2):
    """high dimensional embeddings into a low dim PCA space"""
    X = np.vstack(df[emb_col].values)
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(X)

    df = df.copy()
    df["pc1"] = coords[:, 0]
    df["pc2"] = coords[:, 1]

    return df, pca


def compute_silhouette_for_plot(df, x_col, y_col, label_col):
    """compute mean silhouette score in inheritancedrift space"""
    sub = df[[x_col, y_col, label_col]].dropna()

    if sub[label_col].nunique() < 2:
        return None

    X = sub[[x_col, y_col]].values
    labels = sub[label_col].values

    return silhouette_score(X, labels)


def format_p(p, thresh=0.001):
    """pretty p-value formatting"""
    if p < thresh:
        return f"{p:.2e}"
    return f"{p:.3f}"


def plot_inheritance_vs_drift(
    df,
    x_col,
    y_col,
    region_col="Region",
    title=None,
    out_path=None,
):
    """
    scatter plot of lexical inheritance vs semantic drift

    - Each point = one ethnicity
    - colored by region or race
    - annotates:
        - Spearman correlation
        - p-value
        - silhouette score
    """

    sub = df[[x_col, y_col, region_col]].dropna()

    if sub.empty:
        print("No data to plot.")
        return

    # correlations
    rho, p = spearmanr(sub[x_col], sub[y_col])
    p_str = format_p(p)

    # silhouette score
    sil = compute_silhouette_for_plot(df, x_col, y_col, region_col)
    sil_str = f"{sil:.2f}" if sil is not None else "NA"

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

    # labels
    plt.xlabel("Adjective Signature Inheritance")
    plt.ylabel("Semantic Drift")

    if title:
        plt.title(title)

    # annotation block
    plt.text(
        0.02,
        0.98,
        f"Spearman ρ = {rho:.2f}\n" f"p = {p_str}\n" f"Silhouette = {sil_str}",
        transform=plt.gca().transAxes,
        ha="left",
        va="top",
    )

    plt.legend(title=region_col, fontsize=8)
    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300)
        plt.close()
    else:
        plt.show()


def run():
    df_sem = fetch_duck_df(DB_PATH, "ethnicity_semantic")
    df_inh = fetch_duck_df(DB_PATH, "ethnicity_inheritance")

    df = df_sem.merge(
        df_inh,
        on="Ethnicity",
        how="inner",
        suffixes=("", "_inherit"),
    )

    # summary correlations
    corr_df = compute_correlations(df)
    print("\nSemantic Drift vs Lexical Inheritance:")
    print(corr_df.to_string(index=False))

    df, _ = add_pca_coords(df)

    # region level
    plot_inheritance_vs_drift(
        df,
        x_col="adj_region_mean",
        y_col="semantic_drift_region",
        region_col="Region",
        title="Lexical Inheritance vs Semantic Drift (Adjectives, Region)",
        out_path="figures/drift/adj_region_inheritance_vs_drift.png",
    )

    plot_inheritance_vs_drift(
        df,
        x_col="verb_region_mean",
        y_col="semantic_drift_region",
        region_col="Region",
        title="Lexical Inheritance vs Semantic Drift (Verbs, Region)",
        out_path="figures/drift/verb_region_inheritance_vs_drift.png",
    )

    # race level
    plot_inheritance_vs_drift(
        df,
        x_col="adj_race_mean",
        y_col="semantic_drift_race",
        region_col="Race",
        title="Lexical Inheritance vs Semantic Drift (Adjectives, Race)",
        out_path="figures/drift/adj_race_inheritance_vs_drift.png",
    )

    plot_inheritance_vs_drift(
        df,
        x_col="verb_race_mean",
        y_col="semantic_drift_race",
        region_col="Race",
        title="Lexical Inheritance vs Semantic Drift (Verbs, Race)",
        out_path="figures/drift/verb_race_inheritance_vs_drift.png",
    )

    print("Finished")


if __name__ == "__main__":
    """
    Run via:
    python -m utils.semantic_drift.analysis
    """
    run()
