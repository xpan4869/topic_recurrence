# 8_merge_outputs.py
# Author: Yolanda Pan (xpan02@uchicago.edu)
# Merge outputs from 7_build_topic_network_cluster.py and survey/transcript.
# Input: cluster_edges_by_convo.parquet (from 7), survey.ALL.parquet (enjoyment, sex),
#        transcript_backbiter.ALL.parquet (start/stop for conv_length: (last_stop - first_start)/60).
# Output: convo_features_outdegree_enjoyable.csv (includes conv_length_min from transcript).

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

from parquet_helper import read_parquet_any


CANDOR_DIR = Path("/project/ycleong/datasets/CANDOR")
TRANSCRIPT_FILE = "transcript_backbiter.ALL.parquet"
PROJECT_DATA = Path("/home/xpan02/topic_recurrence/data")


def ensure_data_dir() -> Path:
    d = PROJECT_DATA
    d.mkdir(parents=True, exist_ok=True)
    return d


def compute_avg_outdegree_per_convo(edges: pd.DataFrame) -> pd.DataFrame:
    """
    edges must have: conversation_id (or convo_id), source, target [, weight]

    Outdegree definition:
      outdeg(node) = number of outgoing edges from node (unweighted)
      avg_out_degree(convo) = mean outdeg across ALL nodes in that convo,
                              including nodes with 0 outdegree (e.g., last cluster).
    """
    # normalize convo column name (use conversation_id consistently)
    if "convo_id" in edges.columns and "conversation_id" not in edges.columns:
        edges = edges.rename(columns={"convo_id": "conversation_id"}).copy()
    elif "conversation_id" not in edges.columns:
        raise ValueError("Edges must contain 'conversation_id' or 'convo_id'.")
    else:
        edges = edges.copy()

    required = {"conversation_id", "source", "target"}
    missing = required - set(edges.columns)
    if missing:
        raise ValueError(f"Edges missing required columns: {missing}")

    # All nodes per convo = union of source/target
    nodes = pd.concat(
        [
            edges[["conversation_id", "source"]].rename(columns={"source": "node"}),
            edges[["conversation_id", "target"]].rename(columns={"target": "node"}),
        ],
        ignore_index=True,
    ).drop_duplicates()

    # Outdegree per (conversation_id, node) from source counts
    outdeg = (
        edges.groupby(["conversation_id", "source"])
        .size()
        .reset_index(name="out_degree")
        .rename(columns={"source": "node"})
    )

    # Left-join so nodes with 0 outdegree appear
    node_outdeg = nodes.merge(outdeg, on=["conversation_id", "node"], how="left")
    node_outdeg["out_degree"] = node_outdeg["out_degree"].fillna(0).astype(int)

    convo_feats = (
        node_outdeg.groupby("conversation_id", as_index=False)
        .agg(
            avg_out_degree=("out_degree", "mean"),
            n_nodes=("node", "nunique"),
        )
        .rename(columns={"conversation_id": "convo_id"})  # Keep convo_id for merge with survey
    )

    return convo_feats


def compute_conv_length_from_transcript(transcript: pd.DataFrame) -> pd.DataFrame:
    """
    Compute conversation length in minutes from transcript: (last stop - first start) / 60 per conversation.
    transcript must have: conversation_id, start, stop (times in seconds).
    Returns DataFrame with convo_id, conv_length_min.
    """
    for col in ("conversation_id", "start", "stop"):
        if col not in transcript.columns:
            raise ValueError(f"Transcript must have columns conversation_id, start, stop; missing: {col}")
    agg = (
        transcript.groupby("conversation_id", as_index=False)
        .agg(first_start=("start", "min"), last_stop=("stop", "max"))
    )
    agg["conv_length_min"] = ((agg["last_stop"] - agg["first_start"]) / 60.0).astype(np.float64)
    return agg[["conversation_id", "conv_length_min"]].rename(columns={"conversation_id": "convo_id"})


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
      - sex_pair: FF/MM/FM (unordered); UNK if missing
    conv_length_min is no longer from survey; use compute_conv_length_from_transcript() instead.
    """
    if "conversation_id" in surveys.columns and "convo_id" not in surveys.columns:
        surveys = surveys.rename(columns={"conversation_id": "convo_id"}).copy()
    needed = {"convo_id", "user_id", "how_enjoyable", "sex"}
    missing = needed - set(surveys.columns)
    if missing:
        raise ValueError(f"Survey missing required columns: {missing}")

    s = surveys[["convo_id", "user_id", "how_enjoyable", "sex"]].copy()
    s = s.drop_duplicates(subset=["convo_id", "user_id"])

    s["sex_norm"] = s["sex"].apply(_normalize_sex_to_fm)

    agg_basic = (
        s.groupby("convo_id", as_index=False)
        .agg(
            avg_how_enjoyable=("how_enjoyable", "mean"),
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
    parser = argparse.ArgumentParser(description="Compute conversation features from cluster edges and survey data.")
    parser.add_argument(
        "--edges_parquet",
        type=Path,
        default=CANDOR_DIR / "cluster_edges_by_convo.parquet",
        help="Edges parquet from 5_build_topic_network_cluster.py",
    )
    parser.add_argument(
        "--survey_parquet",
        type=Path,
        default=CANDOR_DIR / "survey.ALL.parquet",
        help="Survey parquet (how_enjoyable, sex; conv_length no longer used)",
    )
    parser.add_argument(
        "--transcript_parquet",
        type=Path,
        default=CANDOR_DIR / TRANSCRIPT_FILE,
        help="Transcript parquet for conv_length: (last stop - first start)/60 per conversation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DATA / "convo_features_outdegree_enjoyable.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    # -------- load edges & compute convo outdegree features --------
    if not args.edges_parquet.exists():
        raise FileNotFoundError(f"Edges parquet not found: {args.edges_parquet}")
    edges = read_parquet_any(str(args.edges_parquet))
    convo_outdeg = compute_avg_outdegree_per_convo(edges)

    # -------- load survey & compute convo-level survey feats (no conv_length) --------
    if not args.survey_parquet.exists():
        raise FileNotFoundError(f"Survey parquet not found: {args.survey_parquet}")
    surveys = read_parquet_any(str(args.survey_parquet))
    convo_survey = compute_convo_survey_feats(surveys)

    # -------- conv_length from transcript: (last stop - first start) / 60 per convo --------
    if not args.transcript_parquet.exists():
        raise FileNotFoundError(f"Transcript parquet not found: {args.transcript_parquet}")
    transcript = read_parquet_any(str(args.transcript_parquet))
    conv_length_df = compute_conv_length_from_transcript(transcript)

    # -------- merge --------
    final = (
        convo_outdeg
        .merge(convo_survey, on="convo_id", how="left")
        .merge(conv_length_df, on="convo_id", how="left")
        .sort_values("convo_id")
        .reset_index(drop=True)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.output, index=False)
    print("Saved:", args.output)
    print("Rows:", len(final))
    print(final.head(5))


if __name__ == "__main__":
    main()
