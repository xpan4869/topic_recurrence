# 4_map_general_topic.py
# Author: Yolanda Pan (xpan02@uchicago.edu)
# Last Edited: 2025/12/12

# Map BERTopic topic IDs to higher-level general topics and attach them to chunk-level data.
# Inputs:
#   - chunk_topic-num.parquet              (chunk-level topic assignments)
#   - topic-label_all.csv                  (topic_id -> short_label/summary/keywords)
#   - topic_general_mapping.csv            (topic_id OR short_label -> general_topic)
# Output:
#   - final-topic.parquet                  (chunk_id, conversation_id, chunk_text, topic, general_topic)

import os
import argparse
import pandas as pd

from parquet_helper import read_parquet_any, write_parquet_any

def main() -> None:
    parser = argparse.ArgumentParser(description="Map topic_id to general_topic and merge into chunk-level parquet.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/project/macs40123/yolanda/candor_parquet",
        help="Directory containing chunk_topic-num.parquet",
    )
    parser.add_argument(
        "--feat_parquet",
        type=str,
        default="chunk_topic-num.parquet",
        help="Chunk-level parquet filename (within input_dir) or full path",
    )
    parser.add_argument(
        "--label_csv",
        type=str,
        default="/home/xpan02/CASNL/macs-40123-xpan4869/itr_4-automated-labeling/topic-label_all.csv",
        help="CSV with topic_id labels",
    )
    parser.add_argument(
        "--mapping_csv",
        type=str,
        default="/home/xpan02/CASNL/macs-40123-xpan4869/itr_4-automated-labeling/topic_general_mapping.csv",
        help="CSV mapping to general_topic",
    )
    parser.add_argument(
        "--out_parquet",
        type=str,
        default="final-topic.parquet",
        help="Output parquet filename (within input_dir) or full path",
    )
    args = parser.parse_args()

    # Resolve paths
    feat_path = args.feat_parquet
    if not os.path.isabs(feat_path):
        feat_path = os.path.join(args.input_dir, feat_path)

    out_path = args.out_parquet
    if not os.path.isabs(out_path):
        out_path = os.path.join(args.input_dir, out_path)

    # Load inputs
    chunk_topic = read_parquet_any(feat_path)
    label = pd.read_csv(args.label_csv)
    mapping = pd.read_csv(args.mapping_csv)

    # Basic schema checks
    if "topic" not in chunk_topic.columns:
        raise ValueError("chunk_topic-num.parquet must contain a 'topic' column.")
    if "topic_id" not in label.columns:
        raise ValueError("label CSV must contain 'topic_id'.")
    if "general_topic" not in mapping.columns:
        raise ValueError("mapping CSV must contain 'general_topic'.")

    # Ensure topic_id exists in chunk_topic
    chunk_topic = chunk_topic.copy()
    chunk_topic["topic_id"] = chunk_topic["topic"]

    # Decide merge key for mapping:
    # Prefer mapping on topic_id if available; otherwise try short_label.
    if "topic_id" in mapping.columns:
        # label + mapping on topic_id
        label_new = label.merge(mapping[["topic_id", "general_topic"]], on="topic_id", how="left")
    elif "short_label" in mapping.columns:
        # label + mapping on short_label
        if "short_label" not in label.columns:
            raise ValueError("mapping uses short_label, but label CSV has no 'short_label' column.")
        label_new = label.merge(mapping[["short_label", "general_topic"]], on="short_label", how="left")
    else:
        raise ValueError("mapping CSV must have either 'topic_id' or 'short_label' as a key column.")

    label_renamed = label_new[["topic_id", "general_topic"]].drop_duplicates()

    # Merge into chunk_topic
    chunk_topic_labeled = chunk_topic.merge(label_renamed, on="topic_id", how="left")

    out_df = (
        chunk_topic_labeled[["chunk_id", "conversation_id", "chunk_text", "topic", "general_topic"]]
        .sort_values(["conversation_id", "chunk_id"])
        .reset_index(drop=True)
    )

    write_parquet_any(out_df, out_path)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
