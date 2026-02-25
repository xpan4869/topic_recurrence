# 2_topic_modeling.py
# Author: Yolanda Pan
# Date: 2026/2/17
# This script:
#   - Loads chunk-level texts and embeddings (MPNet from backbiter_chunk_embed.parquet)
#   - Fits BERTopic using precomputed embeddings (static topic induction)
#   - Preserves within-conversation chunk order for downstream analysis
#   - Assigns a topic label to each chunk
#   - Saves chunk-level Parquet with chunk_id, conversation_id, chunk_text, topic, embedding
# Output: chunk_topic.parquet (same file updated by 2b_rate_depth.py with depth_score).

import os
from ast import literal_eval

import numpy as np
import pandas as pd
from bertopic import BERTopic
from pathlib import Path

from parquet_helper import read_parquet_any, write_parquet_any


# ----------------------------- Helper Functions -----------------------------
def ensure_embedding_col(
    df: pd.DataFrame,
    src_col: str = "chunk_vector",
    out_col: str = "embedding",
) -> pd.DataFrame:
    """
    Ensure df[out_col] contains np.ndarray(float32) embeddings.

    Handles cases where df[src_col] is:
      - already a list/np.ndarray, OR
      - stored as a string like "[0.1, 0.2, ...]".
    """
    if len(df) == 0:
        raise ValueError("Input dataframe is empty; nothing to embed.")
    if src_col not in df.columns:
        raise ValueError(f"Expected '{src_col}' column in dataframe.")

    x0 = df[src_col].iloc[0]
    if isinstance(x0, str):
        df[out_col] = df[src_col].apply(
            lambda x: np.array(literal_eval(x), dtype="float32")
        )
    else:
        df[out_col] = df[src_col].apply(
            lambda x: np.array(x, dtype="float32")
        )

    return df


def reindex_chunks_within_conversation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort by (conversation_id, chunk_id) and create a 0..T-1 index within each conversation.

    Keeps the original chunk_id as chunk_id_orig and overwrites chunk_id
    to reflect within-conversation order. Temporal structure is preserved
    for downstream sequence or transition analysis, but is not used by the model.
    """
    required = {"conversation_id", "chunk_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.sort_values(["conversation_id", "chunk_id"]).reset_index(drop=True).copy()
    df["chunk_id_orig"] = df["chunk_id"]
    df["chunk_id"] = df.groupby("conversation_id").cumcount()
    return df


# ----------------------------- Main -----------------------------

def main() -> None:
    base = Path("/project/ycleong/datasets/CANDOR")
    in_path = base / "backbiter_chunk_embed.parquet"
    out_path = base / "chunk_topic.parquet"

    # sanity
    print("Reading:", in_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Missing: {in_path}")

    # 1) Load chunk-level data
    df = read_parquet_any(str(in_path))

    # 2) Preserve within-conversation ordering
    df = reindex_chunks_within_conversation(df)

    # 3) Ensure embeddings are real arrays
    df = ensure_embedding_col(df, src_col="chunk_vector", out_col="embedding")

    # 4) Fit BERTopic using precomputed embeddings
    docs = df["chunk_text"].fillna("").astype(str).tolist()
    embeddings = np.vstack(df["embedding"].to_numpy())

    topic_model = BERTopic(verbose=True)
    topics, _ = topic_model.fit_transform(docs, embeddings)
    df["topic"] = topics

    # 5) Save chunk-level output (include embedding column for downstream avg per topic)
    out_df = (
        df[["chunk_id", "conversation_id", "chunk_text", "topic", "embedding"]]
        .sort_values(["conversation_id", "chunk_id"])
        .reset_index(drop=True)
    )
    write_parquet_any(out_df, str(out_path))

    print("Done. Saved:")
    print(f"  - {out_path}")


if __name__ == "__main__":
    main()
