# 6_outdegree_merge_enjoy.py
# Author: Yolanda Pan (xpan02@uchicago.edu)
# Last Edited: 2025/12/25
"""
Conversation-level features:
  - avg_out_degree + n_nodes from per-conversation topic transition edges
  - avg_how_enjoyable from survey.ALL.parquet
  - conv_length_min from survey.ALL.parquet (min across the two participants)
  - sex_pair from survey.ALL.parquet (FF / MM / FM; unordered; UNK if missing)
Outputs a CSV under ./data/ next to this script.
"""

import os
import pandas as pd
import numpy as np

from parquet_helper import read_parquet_any


def script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def ensure_data_dir() -> str:
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    d = os.path.abspath(d)
    os.makedirs(d, exist_ok=True)
    return d


def compute_avg_outdegree_per_convo(edges: pd.DataFrame) -> pd.DataFrame:
    """
    edges must have: conversation_id (or convo_id), source, target [, weight]

    Outdegree definition:
      outdeg(node) = number of outgoing edges from node (unweighted)
      avg_out_degree(convo) = mean outdeg across ALL nodes in that convo,
                              including nodes with 0 outdegree (e.g., last topic).
    """
    # normalize convo column name
    if "conversation_id" in edges.columns:
        edges = edges.rename(columns={"conversation_id": "convo_id"}).copy()
    elif "convo_id" in edges.columns:
        edges = edges.copy()
    else:
        raise ValueError("Edges must contain 'conversation_id' or 'convo_id'.")

    required = {"convo_id", "source", "target"}
    missing = required - set(edges.columns)
    if missing:
        raise ValueError(f"Edges missing required columns: {missing}")

    # All nodes per convo = union of source/target
    nodes = pd.concat(
        [
            edges[["convo_id", "source"]].rename(columns={"source": "node"}),
            edges[["convo_id", "target"]].rename(columns={"target": "node"}),
        ],
        ignore_index=True,
    ).drop_duplicates()

    # Outdegree per (convo, node) from source counts
    outdeg = (
        edges.groupby(["convo_id", "source"])
        .size()
        .reset_index(name="out_degree")
        .rename(columns={"source": "node"})
    )

    # Left-join so nodes with 0 outdegree appear
    node_outdeg = nodes.merge(outdeg, on=["convo_id", "node"], how="left")
    node_outdeg["out_degree"] = node_outdeg["out_degree"].fillna(0).astype(int)

    convo_feats = (
        node_outdeg.groupby("convo_id", as_index=False)
        .agg(
            avg_out_degree=("out_degree", "mean"),
            n_nodes=("node", "nunique"),
        )
    )

    return convo_feats


def _normalize_sex_to_fm(x):
    """Map sex to 'F'/'M'/NaN robustly across common encodings."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()

    # common string encodings
    if s in {"f", "female", "woman", "girl"}:
        return "F"
    if s in {"m", "male", "man", "boy"}:
        return "M"

    # common numeric encodings (guessy; adjust if you know the codebook)
    # e.g., 1/2 or 0/1
    if s in {"1", "0"}:
        # don't guess too hard—treat unknown numeric as NaN unless you're sure
        return np.nan
    if s in {"2"}:
        return np.nan

    return np.nan


def _sex_pair_from_list(sex_list):
    """
    sex_list: iterable of sex values for (unique) participants in a convo.
    Returns: 'FF'/'MM'/'FM' or 'UNK' if not enough info.
    """
    # keep only F/M
    vals = [v for v in sex_list if v in {"F", "M"}]
    vals = list(dict.fromkeys(vals))  # unique, preserve order
    # BUT we actually want the two participants, not unique sexes.
    # So instead: use original list after normalization + dropna, then take first two.
    # We'll implement correctly below in compute_convo_survey_feats.
    return "UNK"


def compute_convo_survey_feats(surveys: pd.DataFrame) -> pd.DataFrame:
    """
    Build conversation-level survey aggregates:
      - avg_how_enjoyable: mean across both participants
      - conv_length_min: min across both participants
      - sex_pair: FF/MM/FM (unordered); UNK if missing
    """
    needed = {"convo_id", "user_id", "how_enjoyable", "conv_length", "sex"}
    missing = needed - set(surveys.columns)
    if missing:
        raise ValueError(f"Survey missing required columns: {missing}")

    # one row per participant per convo (avoid duplicates if survey has repeats)
    s = surveys[["convo_id", "user_id", "how_enjoyable", "conv_length", "sex"]].copy()
    s = s.drop_duplicates(subset=["convo_id", "user_id"])

    # normalize sex
    s["sex_norm"] = s["sex"].apply(_normalize_sex_to_fm)

    # enjoyment mean + conv_length min
    agg_basic = (
        s.groupby("convo_id", as_index=False)
        .agg(
            avg_how_enjoyable=("how_enjoyable", "mean"),
            conv_length_min=("conv_length", "min"),
            n_participants=("user_id", "nunique"),
        )
    )

    # sex_pair: take the two participants' sex (after normalization), unordered
    def sex_pair(g: pd.DataFrame) -> str:
        vals = g["sex_norm"].dropna().tolist()
        # if we have more than 2 for any reason, take first two
        vals = vals[:2]
        if len(vals) < 2:
            return "UNK"
        a, b = sorted(vals)  # unordered
        if a == "F" and b == "F":
            return "FF"
        if a == "M" and b == "M":
            return "MM"
        if a == "F" and b == "M":
            return "FM"
        return "UNK"

    sex_pairs = (
        s.groupby("convo_id", as_index=False)
        .apply(lambda g: pd.Series({"sex_pair": sex_pair(g)}))
        .reset_index(drop=True)
    )

    out = agg_basic.merge(sex_pairs, on="convo_id", how="left")
    return out


def main() -> None:
    # -------- paths (edit if needed) --------
    INPUT_DATA_PATH = "/project/macs40123/yolanda/candor_parquet"
    EDGES_PARQUET = os.path.join(INPUT_DATA_PATH, "topic_edges_by_convo.parquet")
    SURVEY_PARQUET = os.path.join(INPUT_DATA_PATH, "survey.ALL.parquet")

    # output under ./data next to this script
    out_csv = os.path.join(ensure_data_dir(), "convo_features_outdegree_enjoyable.csv")

    # -------- load edges & compute convo outdegree features --------
    edges = read_parquet_any(EDGES_PARQUET)
    convo_outdeg = compute_avg_outdegree_per_convo(edges)

    # -------- load survey & compute convo-level survey feats --------
    surveys = read_parquet_any(SURVEY_PARQUET)
    convo_survey = compute_convo_survey_feats(surveys)

    # -------- merge --------
    final = (
        convo_outdeg
        .merge(convo_survey, on="convo_id", how="left")
        .sort_values("convo_id")
        .reset_index(drop=True)
    )

    final.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    print("Rows:", len(final))
    print(final.head(5))


if __name__ == "__main__":
    main()
