import pickle
import sys
from pathlib import Path
from collections import Counter
import pandas as pd 
import ast
import duckdb

# setting to main 
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))



def fetch_duck_df(path, name): 
    """load in data from duckdb"""
    print("Loading Data... ")
    with duckdb.connect(str(path), read_only=True) as con:
        df = con.execute(f"SELECT * FROM {name}").fetchdf()
    return df


def save_duck_df(path, df, name):
    print("Saving df as duckdb ...")
    path.parent.mkdir(parents=True, exist_ok=True)

    counter_like_cols = [
        "Adjs",
        "Verbs",
        "Nouns",
        "Adjs Log-Odds",
        "Verbs Log-Odds", 
        "Top Adjs Log-Odds",
        "Top Verbs Log-Odds",
    ] # all

    for col in counter_like_cols:
        if col in df.columns:
            df[col] = df[col].apply(dict)

    con = duckdb.connect(str(path))
    con.execute(f"""
        CREATE OR REPLACE TABLE {name} AS
        SELECT * FROM df
    """)
    con.close()