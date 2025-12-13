# utils/inheritance/visualization.py
# for running chech run_inheritance.py

from pathlib import Path
import matplotlib.pyplot as plt


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
        plt.savefig(f"{out_dir}/inheritance_{fname}.png", dpi=300, bbox_inches="tight")
        plt.close()


def plot_two_panel_inheritance(
    df,
    parent_col,
    adj_cols,
    verb_cols,
    out_dir,
    title_prefix,
    min_n=3,
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for parent, sub in df.groupby(parent_col):
        sub = sub.dropna(subset=list(adj_cols + verb_cols))
        if len(sub) < min_n:
            continue

        sub = sub.sort_values(adj_cols[0])
        y = range(len(sub))

        fig, axes = plt.subplots(
            ncols=2, sharey=True, figsize=(9, max(3, 0.35 * len(sub)))
        )

        # adjectives
        ax = axes[0]
        ax.hlines(y, sub[adj_cols[1]], sub[adj_cols[2]], alpha=0.5)
        ax.scatter(sub[adj_cols[0]], y, zorder=3)
        ax.set_title("Adjective inheritance")
        ax.set_xlabel("Inheritance")
        ax.set_yticks(y)
        ax.set_yticklabels(sub["Ethnicity"])

        # verbs
        ax = axes[1]
        ax.hlines(y, sub[verb_cols[1]], sub[verb_cols[2]], alpha=0.5)
        ax.scatter(sub[verb_cols[0]], y, zorder=3)
        ax.set_title("Verb inheritance")
        ax.set_xlabel("Inheritance")

        fig.suptitle(f"{title_prefix}: {parent}", y=1.02)
        plt.tight_layout()

        fname = parent.lower().replace(" ", "_")
        plt.savefig(f"{out_dir}/inheritance_{fname}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
