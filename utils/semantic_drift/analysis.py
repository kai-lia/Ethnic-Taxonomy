import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

# my own functs
from utils.save_load import fetch_duck_df
from sklearn.metrics import silhouette_samples, silhouette_score

# setting path root for reading or writing data
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# setting path root for reading
DB_PATH = Path("data/clean/ethnicity_clean.duckdb")
TABLE = "ethnicity_semantic"


def compute_correlations(df):
    """compute correlations between lexical inheritance and semantic drift
    output: df
    one row per comparison, including:
        - comparison label
        - sample size
        - Spearman rho
        - Pearson r"""
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

        # avoid unstable correlations on very small samples
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
    """high dimensional embeddings into a low dim PCA space
    output: pca columns"""
    X = np.vstack(df[emb_col].values)
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(X)

    df = df.copy()
    df["pc1"] = coords[:, 0]
    df["pc2"] = coords[:, 1]
    return df, pca


def build_parent_lookup(df, parent_col):
    """building tble mapping a parent category (region or race)
    to its semantic embedding coordinates"""
    return {
        row[parent_col]: (row["pc1"], row["pc2"])
        for _, row in df.iterrows()
        if isinstance(row.get(parent_col), str)
    }


def compute_region_silhouette(df, x_col, y_col, label_col="Region"):
    """
    compute silhouette scores using region labels
    in inheritance–drift space
    output:
    - mean_score
    - out
        df with per-point silhouette scores

    """
    sub = df[[x_col, y_col, label_col]].dropna()

    # need at least 2 regions
    if sub[label_col].nunique() < 2:
        print("Not enough regions for silhouette score.")
        return None

    X = sub[[x_col, y_col]].values
    labels = sub[label_col].values

    mean_score = silhouette_score(X, labels)
    sample_scores = silhouette_samples(X, labels)

    out = sub.copy()
    out["silhouette"] = sample_scores

    return mean_score, out


""" visuals """


def format_p(p, thresh=0.001):
    if p < thresh:
        return f"{p:.2e}"
    else:
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
    Scatter plot of lexical inheritance vs semantic drift
    - Each point = one ethnicity
    - Colored by region.
    - reports the Spearman correlation between
    inheritance and drift
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

    p_str = format_p(p)

    # atats annotation
    plt.text(
        0.02,
        0.98,
        f"Spearman ρ = {rho:.2f}\np = {p_str}",
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


def run():
    #
    df_sem = fetch_duck_df(DB_PATH, "ethnicity_semantic")
    df_inh = fetch_duck_df(DB_PATH, "ethnicity_inheritance")

    df = df_sem.merge(
        df_inh,
        on="Ethnicity",
        how="inner",
        suffixes=("", "_inherit"),
    )

    corr_df = compute_correlations(df)
    print("\nSemantic Drift vs Lexical Inheritance (Spearman):")
    print(corr_df.to_string(index=False))

    df, _ = add_pca_coords(df)

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

    result = compute_region_silhouette(
        df,
        x_col="verb_region_mean",
        y_col="semantic_drift_region",
        label_col="Region",
    )

    if result:
        mean_sil, sil_df = result
        print(f"\nMean silhouette score (Region clustering, verbs): {mean_sil:.3f}")

    result = compute_region_silhouette(
        df,
        x_col="adj_region_mean",
        y_col="semantic_drift_region",
        label_col="Region",
    )

    if result:
        mean_sil, sil_df = result
        print(f"\nMean silhouette score (Region clustering, adj): {mean_sil:.3f}")

    print("Finished")


if __name__ == "__main__":
    """
    Run via:
    python -m utils.semantic_drift.analysis
    """
    run()
