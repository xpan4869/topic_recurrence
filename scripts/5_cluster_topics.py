# 5_cluster_topics.py
# Author: Yolanda Pan (xpan02@uchicago.edu)
# Last Edited: 2026/2/17
# Cluster topics from topic-label_all.csv using L2-normalized topic_embedding.
# Find best number of clusters (silhouette or Davies-Bouldin), apply AgglomerativeClustering,
# and write topic_id -> cluster_id mapping to CSV.

import argparse
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import normalize


def parse_embedding(s: str) -> np.ndarray:
    """Parse topic_embedding string to float32 array. Handles '[0.1, ...]' or '[np.float64(0.1), ...]'."""
    s = str(s).strip()
    if s.startswith("[") and "np.float64" in s:
        s = s.replace("np.float64(", "").replace(")", "")
    return np.array(literal_eval(s), dtype=np.float32)


def load_embeddings(csv_path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load topic-label CSV and build normalized embedding matrix for clustering.
    Returns: (df_full, X_normalized, mask_cluster, topic_ids_for_cluster)
    - df_full: full dataframe (all rows; rows with null topic_embedding kept but not clustered)
    - X_normalized: (n_cluster, dim) L2-normalized embeddings (excluding noise topic -1)
    - mask_cluster: boolean index into df_full for rows that were clustered
    - topic_ids_for_cluster: topic_id array aligned with X_normalized
    """
    df = pd.read_csv(csv_path)
    if "topic_embedding" not in df.columns:
        raise ValueError(
            "topic_embedding column missing. Regenerate CSV with "
            "scripts/3_label_bertopics.py (and ensure chunk_topic.parquet has embedding from 2_topic_modeling.py)"
        )
    # Keep all rows; only use rows with valid embedding for clustering
    df = df.copy()
    has_emb = df["topic_embedding"].notna()
    df.loc[has_emb, "emb"] = df.loc[has_emb, "topic_embedding"].apply(parse_embedding)

    rows_cluster = has_emb & (df["topic_id"] != -1)
    if rows_cluster.sum() == 0:
        raise ValueError("No rows left after excluding topic_id -1 and missing embeddings.")

    X = np.vstack(df.loc[rows_cluster, "emb"].to_numpy()).astype(np.float32)
    X_normalized = normalize(X, norm="l2")
    topic_ids_cluster = df.loc[rows_cluster, "topic_id"].values
    return df, X_normalized, rows_cluster.values, topic_ids_cluster


def find_optimal_k(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 30,
    criterion: str = "silhouette",
) -> int:
    """Find best k in [k_min, k_max] using silhouette (max) or davies_bouldin (min)."""
    k_max = min(k_max, len(X) // 2)
    if k_max < k_min:
        k_max = k_min
    k_range = range(k_min, k_max + 1)
    sil_scores = []
    db_scores = []

    for k in k_range:
        agg = AgglomerativeClustering(n_clusters=k)
        labels = agg.fit_predict(X)
        sil_scores.append(silhouette_score(X, labels))
        db_scores.append(davies_bouldin_score(X, labels))

    if criterion == "silhouette":
        best_idx = int(np.argmax(sil_scores))
        best_k = k_range[best_idx]
        print(f"Optimal k (silhouette): {best_k} (score={sil_scores[best_idx]:.3f})")
    else:
        best_idx = int(np.argmin(db_scores))
        best_k = k_range[best_idx]
        print(f"Optimal k (Davies-Bouldin): {best_k} (score={db_scores[best_idx]:.3f})")
    return best_k


def main():
    parser = argparse.ArgumentParser(
        description="Cluster topics from topic-label_all.csv; output topic_id -> cluster_id map."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/topic-label_all.csv"),
        help="Input CSV with topic_id, topic_embedding (and optional short_label, etc.).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/topic_cluster_map.csv"),
        help="Output CSV: topic_id, cluster_id.",
    )
    parser.add_argument(
        "--criterion",
        choices=["silhouette", "davies_bouldin"],
        default="silhouette",
        help="Metric to choose best number of clusters (silhouette: higher is better; davies_bouldin: lower).",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=None,
        help="Override: fix number of clusters (skip optimal-k search).",
    )
    parser.add_argument(
        "--k_min",
        type=int,
        default=2,
        help="Min k when searching for optimal number of clusters.",
    )
    parser.add_argument(
        "--k_max",
        type=int,
        default=30,
        help="Max k when searching for optimal number of clusters.",
    )
    args = parser.parse_args()

    csv_path = args.input.resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    print(f"Loading {csv_path}...")
    df_full, X_normalized, mask_cluster, topic_ids_cluster = load_embeddings(csv_path)
    n_cluster = X_normalized.shape[0]
    print(f"Clustering {n_cluster} topics (topic_id -1 excluded).")

    if args.n_clusters is not None:
        n_clusters = args.n_clusters
        print(f"Using fixed n_clusters={n_clusters}")
    else:
        n_clusters = find_optimal_k(
            X_normalized,
            k_min=args.k_min,
            k_max=args.k_max,
            criterion=args.criterion,
        )

    agg = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_ids = agg.fit_predict(X_normalized)

    # Build full mapping: one row per topic_id in the input; cluster_id -1 for noise / missing
    out = df_full[["topic_id"]].drop_duplicates().sort_values("topic_id").reset_index(drop=True)
    cluster_map = pd.Series(cluster_ids.astype(int), index=topic_ids_cluster)
    out["cluster_id"] = out["topic_id"].map(cluster_map).fillna(-1).astype(int)

    out_path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {out.shape[0]} rows to {out_path}")
    print(f"Cluster IDs: 0..{n_clusters - 1} ({n_clusters} clusters); topic_id -1 -> cluster_id -1.")


if __name__ == "__main__":
    main()
