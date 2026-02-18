# 2_topic_modeling.py
# Author: Yolanda Pan
# Date: 2026/01/20
#
# Loads Friends chunk-level texts + embeddings
# Fits BERTopic using precomputed embeddings
# Preserves within-scene order (chunk_id)
# Saves:
#   (1) chunk-level parquet with topic + topic_words
#   (2) topic-level parquet (topic -> top words, counts)
#
# NOTE:
# - BERTopic model is NOT saved
# - Topic interpretation is preserved via topic_words column

from __future__ import annotations

from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
from bertopic import BERTopic

from parquet_helper import read_parquet_any, write_parquet_any


# ============================================================
# Helpers
# ============================================================

def ensure_embedding_col(
    df: pd.DataFrame,
    src_col: str = "chunk_vector",
    out_col: str = "embedding",
) -> pd.DataFrame:
    """Ensure embeddings are np.ndarray(float32) with consistent dimensionality."""
    if src_col not in df.columns:
        raise ValueError(f"Missing column: {src_col}")

    x0 = df[src_col].iloc[0]
    if isinstance(x0, str):
        df[out_col] = df[src_col].apply(lambda x: np.array(literal_eval(x), dtype="float32"))
    else:
        df[out_col] = df[src_col].apply(lambda x: np.array(x, dtype="float32"))

    dim = df[out_col].iloc[0].shape[0]
    bad = df[out_col].apply(
        lambda v: v is None or np.asarray(v).ndim != 1 or np.asarray(v).shape[0] != dim
    )
    if bad.any():
        raise ValueError(f"Invalid embeddings at rows: {bad[bad].index[:10].tolist()}")

    return df


def reindex_chunks_within_scene(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make chunk_id be 0..T-1 within each scene_id.
    Original chunk_id preserved as chunk_id_orig.
    """
    if "scene_id" not in df.columns:
        raise ValueError("Missing column: scene_id")

    if "start_turn_id" in df.columns and "end_turn_id" in df.columns:
        sort_cols = ["scene_id", "start_turn_id", "end_turn_id"]
    elif "chunk_id" in df.columns:
        sort_cols = ["scene_id", "chunk_id"]
    else:
        raise ValueError("Need ordering columns for chunks.")

    df = df.sort_values(sort_cols).reset_index(drop=True).copy()
    df["chunk_id_orig"] = df["chunk_id"]
    df["chunk_id"] = df.groupby("scene_id").cumcount()
    return df


def extract_topic_words(topic_model: BERTopic, top_n: int = 15) -> pd.DataFrame:
    """Build topic → words table."""
    info = topic_model.get_topic_info()
    rows = []

    for t in info["Topic"].tolist():
        if t == -1:
            continue
        words = topic_model.get_topic(t) or []
        rows.append({
            "topic": int(t),
            "n_chunks": int(info.loc[info["Topic"] == t, "Count"].values[0]),
            "top_words": ", ".join(w for w, _ in words[:top_n]),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("n_chunks", ascending=False)
        .reset_index(drop=True)
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    base = Path("/project/ycleong/datasets/Friends")

    in_path = base / "friends_chunk_embed.parquet"
    out_chunk_path = base / "friends_chunk_topic.parquet"
    out_topic_path = base / "friends_topics.parquet"

    if not in_path.exists():
        raise FileNotFoundError(in_path)

    # 1) Load
    df = read_parquet_any(str(in_path))
    required = {"chunk_text", "chunk_vector", "scene_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # 2) Preserve within-scene temporal order
    df = reindex_chunks_within_scene(df)

    # 3) Ensure embeddings
    df = ensure_embedding_col(df)

    # 4) Fit BERTopic with precomputed embeddings
    docs = df["chunk_text"].fillna("").astype(str).tolist()
    embeddings = np.vstack(df["embedding"].to_numpy())

    topic_model = BERTopic(verbose=True)
    topics, _ = topic_model.fit_transform(docs, embeddings)
    df["topic"] = topics

    # 5) Extract topic words (model disposable after this)
    topics_df = extract_topic_words(topic_model, top_n=15)
    topic_word_map = dict(zip(topics_df["topic"], topics_df["top_words"]))
    df["topic_words"] = df["topic"].map(topic_word_map)

    # 6) Save outputs
    keep_cols = [
        "scene_id",
        "chunk_id",
        "chunk_text",
        "topic",
        "topic_words",
    ]
    for extra in [
        "season_id",
        "episode_id",
        "start_turn_id",
        "end_turn_id",
        "n_words",
        "chunk_id_orig",
    ]:
        if extra in df.columns:
            keep_cols.insert(0, extra)

    out_df = (
        df[keep_cols]
        .sort_values(["scene_id", "chunk_id"])
        .reset_index(drop=True)
    )

    write_parquet_any(out_df, str(out_chunk_path))
    write_parquet_any(topics_df, str(out_topic_path))


if __name__ == "__main__":
    main()