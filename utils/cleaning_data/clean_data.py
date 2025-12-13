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


from utils.save_load import fetch_duck_df, save_duck_df

# paths to data
PKL_PATH = Path("data/ethnicity_class/taxonomy_structure.pkl")
DB_PATH = Path("data/input/ethnicity_pos.duckdb")
OUT_DB_PATH = Path("data/clean/ethnicity_clean.duckdb")

ETHNICITY_ALIAS = { # for the sake of another project these are separate in my data, am combining them
    "asian indian": "indian",
    "native hawaiian": "hawaiian",
}



def load_pickle(path):
    path = Path(path)
    with path.open("rb") as f:
        return pickle.load(f)


def clean_ethnicity(df): 
    """i formatted things as south_asian for tokenizing purposes"""
    def normalize_ethnicity(eth):
        """ grouping hawaiian and native hawaiian + more"""
        if not isinstance(eth, str):
            return eth
        eth = eth.lower().strip()
        return ETHNICITY_ALIAS.get(eth, eth)
    
    df["ethnicity"] = df["ethnicity"].str.replace("_", " ", regex=False)
    df["ethnicity"] = df["ethnicity"].apply(normalize_ethnicity) 
    


def parse_list(x):
    """input df saved the lists as strings, reverting to lists"""
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []

def add_taxonomy(df): 
    """ check out file cleaning_data/create_taxomomy.py for more details
    adding lables for hierarical taxonomy"""
    def get_classification(eth, taxonomy_struct):
        # if term is missing → None, None
        if eth not in taxonomy_struct:
            return None, None

        values = taxonomy_struct[eth]

        # values may be [] or shorter/longer lists
        race  = values[0] if len(values) > 0 else None
        region = values[1] if len(values) > 1 else None

        return race, region
    print("loading pikcle ")
    taxonomy_structure = load_pickle(PKL_PATH) # checkout 

    df[["Race", "Region"]] = df["Ethnicity"].apply(lambda x: pd.Series(get_classification(x, taxonomy_structure)))
    return df


def sanitize_token(t):
    "was running into emoji issues, will clean here"
    if isinstance(t, str):
        return t.encode("utf-8", "ignore").decode("utf-8")
    return t



    
def format_to_counters(df): 
    """data base is currently very unstructured
    input (sentance level parses): 
       ethnicity adjs        verbs                nouns   count    has_dup
       chinese["cool"]	  ["found", "win"]	  ["people"]	1	  False

    out: 
    	Ethnicity	Adjs	                  Verbs	                 Nouns	               Race	   Region
	    chinese	   Counter("lovely": 5, ...) Counter("eat": 10, ...) Counter("kid": 10, ..) Asian  East Asian
    """
    results = []
    # had some formating issues with str
    clean_ethnicity(df) # remove whitespace
    
    df["adjs"]  = df["adjs"].apply(parse_list)
    df["verbs"]  = df["verbs"].apply(parse_list)
    df["nouns"]  = df["nouns"].apply(parse_list)

    for eth, group in df.groupby("ethnicity"):
        print(eth)
        adj_counter = Counter()
        verb_counter = Counter()
        noun_counter = Counter()

        for _, row in group.iterrows():
            c = row["count"] # some are duplicates

            # add weighted adjective counts
            for adj in row["adjs"]:
                adj_counter[sanitize_token(adj)] += c

            #add weighted verb counts
            for verb in row["verbs"]:
                verb_counter[sanitize_token(verb)] += c

            # add weighted noun counts
            for noun in row["nouns"]:
                noun_counter[sanitize_token(noun)] += c

        results.append({
            "Ethnicity": eth,
            "Adjs": adj_counter,
            "Verbs": verb_counter,
            "Nouns": noun_counter
        })
    ethnicity_df = pd.DataFrame(results)
    return ethnicity_df


def main(): 
    
    df = fetch_duck_df(DB_PATH, "ethnicity_sentence_modifiers")

    print("Formatting and Cleaning ...")
    ethnicity_df = format_to_counters(df)
    print("Adding Taxonomy ...")
    ethnicity_df = add_taxonomy(ethnicity_df)


    save_duck_df(OUT_DB_PATH, ethnicity_df, "ethnicity_clean")


if __name__ == "__main__":
    """ run via the command python -m utils.cleaning_data.clean_data"""
    main()
   
