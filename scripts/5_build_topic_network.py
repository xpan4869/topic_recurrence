# 5_build_topic_network.py
# Author: Yolanda Pan (xpan02@uchicago.edu)
# Last Edited: 2025/12/12
"""
Build per-conversation topic-transition networks.

Input (final-topic.parquet) should contain:
  - conversation_id (or convo_id)
  - chunk_id
  - general_topic
  - topic_id (optional; used to drop -1 outliers)

Outputs:
  - topic_edges_by_convo.parquet : conversation_id, source, target, weight
  - topic_nodes_by_convo.parquet : conversation_id, node_id, n_chunks

Definition:
  - Node = general_topic
  - Directed edge = transition from topic_t -> topic_{t+1} within a conversation
  - Consecutive duplicates are collapsed (A,A,B -> A,B)
  - Missing topics (and optionally outliers topic_id == -1) are excluded
"""

import os
import argparse
import numpy as np
import pandas as pd

from parquet_helper import read_parquet_any, write_parquet_any


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-conversation topic transition networks.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/project/macs40123/yolanda/candor_parquet",
    )
    parser.add_argument(
        "--in_parquet",
        type=str,
        default="final-topic.parquet",
    )
    parser.add_argument(
        "--out_edges",
        type=str,
        default="topic_edges_by_convo.parquet",
    )
    parser.add_argument(
        "--out_nodes",
        type=str,
        default="topic_nodes_by_convo.parquet",
    )
    parser.add_argument(
        "--drop_outliers",
        action="store_true",
        help="Drop rows where topic_id == -1 (BERTopic outliers), if topic_id exists.",
    )
    args = parser.parse_args()

    # Resolve paths
    in_path = args.in_parquet if os.path.isabs(args.in_parquet) else os.path.join(args.input_dir, args.in_parquet)
    out_edges = args.out_edges if os.path.isabs(args.out_edges) else os.path.join(args.input_dir, args.out_edges)
    out_nodes = args.out_nodes if os.path.isabs(args.out_nodes) else os.path.join(args.input_dir, args.out_nodes)

    df = read_parquet_any(in_path)

    # Accept conversation_id or convo_id
    if "conversation_id" in df.columns:
        convo_col = "conversation_id"
    elif "convo_id" in df.columns:
        convo_col = "convo_id"
    else:
        raise ValueError("Input must contain 'conversation_id' or 'convo_id'.")

    required = {convo_col, "chunk_id", "general_topic"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    topics = df.copy()

    # Clean general_topic
    topics["topic_clean"] = topics["general_topic"].replace({"None": np.nan, "nan": np.nan})

    # Drop BERTopic outliers if topic_id exists
    if args.drop_outliers:
        if "topic_id" not in topics.columns:
            raise ValueError("--drop_outliers set but input has no 'topic_id' column.")
        topics.loc[topics["topic_id"] == -1, "topic_clean"] = np.nan

    # Sort within conversation
    topics = topics.sort_values([convo_col, "chunk_id"]).reset_index(drop=True)

    # Drop missing topics
    t = topics.dropna(subset=["topic_clean"]).copy()

    # Collapse consecutive duplicates within conversation
    t["prev_topic"] = t.groupby(convo_col)["topic_clean"].shift(1)
    t = t[t["topic_clean"] != t["prev_topic"]].copy()

    # Next topic within conversation
    t["next_topic"] = t.groupby(convo_col)["topic_clean"].shift(-1)

    # ---- Edges (per conversation) ----
    edges_raw = (
        t.dropna(subset=["next_topic"])[[convo_col, "topic_clean", "next_topic"]]
        .rename(columns={convo_col: "conversation_id", "topic_clean": "source", "next_topic": "target"})
        .reset_index(drop=True)
    )

    # Aggregate within each conversation
    edges_by_convo = (
        edges_raw
        .groupby(["conversation_id", "source", "target"])
        .size()
        .reset_index(name="weight")
        .sort_values(["conversation_id", "weight"], ascending=[True, False])
        .reset_index(drop=True)
    )

    # ---- Nodes (per conversation) ----
    nodes_by_convo = (
        t.groupby([convo_col, "topic_clean"])
        .size()
        .reset_index(name="n_chunks")
        .rename(columns={convo_col: "conversation_id", "topic_clean": "node_id"})
        .sort_values(["conversation_id", "n_chunks"], ascending=[True, False])
        .reset_index(drop=True)
    )

    write_parquet_any(edges_by_convo, out_edges)
    write_parquet_any(nodes_by_convo, out_nodes)

    print("Done. Saved:")
    print(f"  - {out_edges}")
    print(f"  - {out_nodes}")
    print(f"Edges rows: {len(edges_by_convo)} | Nodes rows: {len(nodes_by_convo)}")


if __name__ == "__main__":
    main()
