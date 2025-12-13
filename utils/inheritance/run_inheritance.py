import sys
from pathlib import Path

from utils.save_load import fetch_duck_df, save_duck_df
from utils.inheritance.inheritance import inheritance_with_bootstrap
from utils.inheritance.visualization import (
    plot_by_parent,
    plot_two_panel_inheritance,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DB_PATH = Path("data/clean/ethnicity_clean.duckdb")


def main():
    df = fetch_duck_df(DB_PATH, "ethnicity_log_odds")

    REQUIRED = {
        "Top Adjs Log-Odds",
        "Top Verbs Log-Odds",
        "Region",
        "Race",
    }
    assert REQUIRED.issubset(df.columns)

    inherit_df = inheritance_with_bootstrap(df, n_boot=1000)

    save_duck_df(
        DB_PATH,
        inherit_df,
        name="ethnicity_inheritance",
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
    main()
