# 7_build_topic_network_cluster.py
# Author: Yolanda Pan (xpan02@uchicago.edu)
# Build per-conversation topic-transition networks using cluster_id from topic_cluster_map.csv.
# nodes = cluster_id, edges = transitions within conversation.
#
# Input: CANDOR chunk_topic.parquet (conversation_id, chunk_id, topic) + topic_cluster_map.csv (topic_id -> cluster_id).
# Output: cluster_edges_by_convo.parquet, cluster_nodes_by_convo.parquet.

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from parquet_helper import read_parquet_any, write_parquet_any


CANDOR_DIR = Path("/project/ycleong/datasets/CANDOR")
PROJECT_DATA = Path("/home/xpan02/topic_recurrence/data")


def build_edges_nodes(
    df: pd.DataFrame,
    convo_col: str,
    drop_topic_id_minus_one: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build edges/nodes: use cluster_id directly, drop noise, collapse consecutive dupes, build network."""
    if "cluster_id" not in df.columns:
        raise ValueError(f"Column 'cluster_id' not found in dataframe. Available columns: {list(df.columns)}")
    
    topics = df.copy()
    
    # Drop noise: topic == -1 (from parquet) or cluster_id == -1
    if drop_topic_id_minus_one:
        # Check both topic_id (from merge) and topic (original parquet column)
        if "topic_id" in topics.columns:
            topics.loc[topics["topic_id"] == -1, "cluster_id"] = np.nan
        elif "topic" in topics.columns:
            topics.loc[topics["topic"] == -1, "cluster_id"] = np.nan
    # Drop cluster_id == -1 (noise cluster)
    topics.loc[topics["cluster_id"] == -1, "cluster_id"] = np.nan
    
    # Convert cluster_id to string for consistent node labels (handle NaN)
    topics["cluster_id"] = topics["cluster_id"].astype(str).replace({"nan": np.nan, "None": np.nan, "<NA>": np.nan})

    topics = topics.sort_values([convo_col, "chunk_id"]).reset_index(drop=True)
    t = topics.dropna(subset=["cluster_id"]).copy()

    # Collapse consecutive duplicates within conversation
    t["prev_cluster"] = t.groupby(convo_col)["cluster_id"].shift(1)
    t = t[t["cluster_id"] != t["prev_cluster"]].copy()
    t["next_cluster"] = t.groupby(convo_col)["cluster_id"].shift(-1)

    # Build edges
    edges_raw = (
        t.dropna(subset=["next_cluster"])[[convo_col, "cluster_id", "next_cluster"]]
        .rename(columns={convo_col: "conversation_id", "cluster_id": "source", "next_cluster": "target"})
        .reset_index(drop=True)
    )
    edges_by_convo = (
        edges_raw.groupby(["conversation_id", "source", "target"])
        .size()
        .reset_index(name="weight")
        .sort_values(["conversation_id", "weight"], ascending=[True, False])
        .reset_index(drop=True)
    )

    # Build nodes
    nodes_by_convo = (
        t.groupby([convo_col, "cluster_id"])
        .size()
        .reset_index(name="n_chunks")
        .rename(columns={convo_col: "conversation_id", "cluster_id": "node_id"})
        .sort_values(["conversation_id", "n_chunks"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return edges_by_convo, nodes_by_convo


def main() -> None:
    parser = argparse.ArgumentParser(description="Build topic transition networks using cluster_id from topic_cluster_map.csv.")
    parser.add_argument(
        "--parquet_dir",
        type=Path,
        default=CANDOR_DIR,
        help="Directory containing chunk_topic.parquet",
    )
    parser.add_argument(
        "--in_parquet",
        type=str,
        default="chunk_topic.parquet",
        help="Chunk-level parquet with conversation_id, chunk_id, topic",
    )
    parser.add_argument(
        "--cluster_map_csv",
        type=Path,
        default=PROJECT_DATA / "topic_cluster_map.csv",
        help="CSV: topic_id -> cluster_id",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=CANDOR_DIR,
        help="Directory for output parquets",
    )
    parser.add_argument(
        "--out_edges",
        type=str,
        default="cluster_edges_by_convo.parquet",
        help="Output edges filename",
    )
    parser.add_argument(
        "--out_nodes",
        type=str,
        default="cluster_nodes_by_convo.parquet",
        help="Output nodes filename",
    )
    parser.add_argument(
        "--drop_outliers",
        action="store_true",
        help="Drop rows where topic_id == -1",
    )
    args = parser.parse_args()

    in_path = args.parquet_dir / args.in_parquet if not os.path.isabs(args.in_parquet) else Path(args.in_parquet)
    if not in_path.exists():
        raise FileNotFoundError(f"Parquet not found: {in_path}")

    if not args.cluster_map_csv.exists():
        raise FileNotFoundError(f"Cluster map CSV not found: {args.cluster_map_csv}")

    chunk_df = read_parquet_any(str(in_path))
    if "conversation_id" not in chunk_df.columns and "convo_id" in chunk_df.columns:
        chunk_df = chunk_df.rename(columns={"convo_id": "conversation_id"})
    convo_col = "conversation_id"
    if convo_col not in chunk_df.columns:
        raise ValueError("Parquet must contain conversation_id or convo_id.")
    if "chunk_id" not in chunk_df.columns or "topic" not in chunk_df.columns:
        raise ValueError("Parquet must contain chunk_id and topic.")

    cluster_map = pd.read_csv(args.cluster_map_csv)
    if "topic_id" not in cluster_map.columns or "cluster_id" not in cluster_map.columns:
        raise ValueError("Cluster map CSV must contain topic_id and cluster_id.")

    # Map topic (from parquet) -> topic_id (for merge with CSV)
    chunk_df["topic_id"] = chunk_df["topic"].astype(int)
    merged = chunk_df.merge(cluster_map[["topic_id", "cluster_id"]], on="topic_id", how="left")
    
    # Ensure cluster_id column exists after merge
    if "cluster_id" not in merged.columns:
        raise ValueError("Merge failed: cluster_id column not found after merging with cluster_map CSV.")

    edges_by_convo, nodes_by_convo = build_edges_nodes(
        merged,
        convo_col=convo_col,
        drop_topic_id_minus_one=args.drop_outliers,
    )

    out_edges_path = args.out_dir / args.out_edges if not os.path.isabs(args.out_edges) else Path(args.out_edges)
    out_nodes_path = args.out_dir / args.out_nodes if not os.path.isabs(args.out_nodes) else Path(args.out_nodes)
    out_edges_path.parent.mkdir(parents=True, exist_ok=True)
    out_nodes_path.parent.mkdir(parents=True, exist_ok=True)

    write_parquet_any(edges_by_convo, str(out_edges_path))
    write_parquet_any(nodes_by_convo, str(out_nodes_path))

    print("Done (cluster_id). Saved:")
    print(f"  - {out_edges_path}")
    print(f"  - {out_nodes_path}")
    print(f"Edges rows: {len(edges_by_convo)} | Nodes rows: {len(nodes_by_convo)}")


if __name__ == "__main__":
    main()
