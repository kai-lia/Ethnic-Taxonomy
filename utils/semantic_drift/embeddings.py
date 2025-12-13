import sys
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from typing import List
from transformers import OlmoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# my own functions
from utils.save_load import fetch_duck_df, save_duck_df

# setting path root for reading or writing data
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DB_PATH = Path("data/clean/ethnicity_clean.duckdb")

# using this model because of data and size
MODEL_NAME = "allenai/OLMo-1B-hf"


def build_pseudodoc(row):
    """collecting the top verbs and adjs"""
    terms = set()

    adj = row.get("Top Adjs Log-Odds", {})
    verb = row.get("Top Verbs Log-Odds", {})

    if isinstance(adj, dict):
        terms |= set(list(adj.keys()))
    if isinstance(verb, dict):
        terms |= set(list(verb.keys()))

    return " ".join(sorted(terms))


def add_pseudodocs(df):
    """ "creating a doc with k top items"""
    df = df.copy()
    df["pseudodoc"] = df.apply(build_pseudodoc, axis=1)
    return df


def embed_texts_olmo(texts: List[str], device=None):
    """loading allen ai's olmo"""
    print("Loading Olmo...")
    # run on cuda if possible
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    # load OLMo model
    # output_hidden_states=False because we only need the final layer
    model = OlmoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        output_hidden_states=False,
    ).to(device)

    model.eval()  # set model to eval mode
    embeddings = []

    print("Computing Embeddings...")
    # disable gradient tracking for faster inference + lower memory usage
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            inputs.pop("token_type_ids", None)  # OLMo does not use
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # forward pass
            outputs = model(**inputs)

            # extract token embeddings from the final hidden layer
            # shape: (sequence_length, hidden_dim)
            h = outputs.last_hidden_state[0]
            emb = h.mean(dim=0)  # mean pooling over tokens
            emb = emb / emb.norm()

            embeddings.append(emb.cpu().numpy())  # store on CPU as a NumPy array

    return embeddings


def compute_embeddings(df):
    "formatting embedings for understanding"
    df = df.copy()
    texts = df["pseudodoc"].fillna("").tolist()
    df["embedding"] = embed_texts_olmo(texts)
    return df


def compute_semantic_drift(df):
    def cosine_drift(a, b):
        """calculating the cosine similarity"""
        similarity = cosine_similarity(np.array([a]), np.array([b]))
        return (1 - similarity)[0][0]

    df = df.copy()  # avoid mutation

    lookup = {
        row["Ethnicity"]: row["embedding"]
        for _, row in df.iterrows()
        if isinstance(row.get("embedding"), np.ndarray)
    }

    drift_region = []
    drift_race = []

    for _, row in df.iterrows():
        emb_e = row["embedding"]

        # drift relative to region parent
        region = row.get("Region")
        if isinstance(region, str) and region in lookup:

            drift_region.append(cosine_drift(emb_e, lookup[region]))
        else:
            # no parent
            drift_region.append(None)

        # drift relative to race parent
        race = row.get("Race")
        if isinstance(race, str) and race in lookup:
            drift_race.append(cosine_drift(emb_e, lookup[race]))
        else:
            drift_race.append(None)

    df["semantic_drift_region"] = drift_region
    df["semantic_drift_race"] = drift_race

    return df


def run():
    df = fetch_duck_df(DB_PATH, "ethnicity_log_odds")

    df = add_pseudodocs(df)
    df = compute_embeddings(df)
    df = compute_semantic_drift(df)

    # saving for future use
    save_duck_df(DB_PATH, df, "ethnicity_semantic")


if __name__ == "__main__":
    """run via: python -m utils.semantic_drift.embeddings"""
    run()
