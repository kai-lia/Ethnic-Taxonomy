import random
import pandas as pd


def bootstrap_inheritance(child_terms, parent_terms, n_boot=1000, seed=42):
    """bootstrap overlap between child and parent signature sets
    caulaiting interval of uncertainty
    output: mean, 1st quantile, 3rd quantial"""
    if not child_terms:
        return None, None, None

    child_terms = list(child_terms)  # lower level on the taxonomy
    parent_terms = set(parent_terms)  # higher level on the taxonomy

    rng = random.Random(seed)  # for consistency
    vals = []

    # adding bootstrap
    for _ in range(n_boot):
        sample = [rng.choice(child_terms) for _ in range(len(child_terms))]
        vals.append(len(set(sample) & parent_terms) / len(sample))

    s = pd.Series(vals)
    return s.mean(), s.quantile(0.025), s.quantile(0.975)


def build_lookup(df, key="Ethnicity"):
    """row lookup by ethnicity label"""
    return {row[key]: row for _, row in df.iterrows() if pd.notna(row[key])}


def get_signature_terms(row):
    """extract words most signature to the term
    output: set of top log values"""
    adj = row.get("Top Adjs Log-Odds", {})
    verb = row.get("Top Verbs Log-Odds", {})

    return {
        "adj": set(adj.keys()) if isinstance(adj, dict) else set(),
        "verb": set(verb.keys()) if isinstance(verb, dict) else set(),
    }


def unpack_stats(prefix, stats):
    mean, lo, hi = stats
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_lo": lo,
        f"{prefix}_hi": hi,
    }


def inheritance_with_bootstrap(df, n_boot=1000):
    """
    for the each of the taxonomy levels, perform bootstrap

    out: new df with bootstrap values
    """
    rows = []
    lookup = build_lookup(df)

    for _, row in df.iterrows():
        eth = row["Ethnicity"]
        region = row["Region"]
        race = row["Race"]

        child_terms = get_signature_terms(row)

        # region-level inheritance and boots
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

        # race-level inheritance and boots
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

        rows.append(
            {
                "Ethnicity": eth,
                "Region": region,
                "Race": race,
                **unpack_stats("adj_region", adj_region),
                **unpack_stats("verb_region", verb_region),
                **unpack_stats("adj_race", adj_race),
                **unpack_stats("verb_race", verb_race),
            }
        )

    return pd.DataFrame(rows)
