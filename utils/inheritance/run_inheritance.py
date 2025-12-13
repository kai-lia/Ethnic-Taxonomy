import sys
from pathlib import Path

from utils.save_load import fetch_duck_df, save_duck_df
from utils.inheritance.inheritance import inheritance_with_bootstrap
from utils.inheritance.visualize import (
    plot_by_parent,
    plot_two_panel_inheritance,
)

# for saving purposes
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DB_PATH = Path("data/clean/ethnicity_clean.duckdb")


def run():
    df = fetch_duck_df(DB_PATH, "ethnicity_log_odds")

    # calculate bootstap and inheritance
    inherit_df = inheritance_with_bootstrap(df, n_boot=1000)

    # save for second part
    save_duck_df(
        DB_PATH,
        inherit_df,
        name="ethnicity_inheritance",
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
    """python -m utils.inheritance.run_inheritance"""
    run()
