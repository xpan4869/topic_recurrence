# What Makes Conversations Enjoyable?

This repository contains code and analysis for the project **“What Makes Conversations Enjoyable?”** (Yolanda Pan, 2026-05-01).

The workflow builds **topic structure** from conversations, aggregates topics into **clusters**, constructs **within-conversation transition networks** over clusters, and merges these with **survey enjoyment** and basic transcript-derived metadata to produce conversation-level features.

## Repository contents

- `analysis.ipynb`: Main analysis notebook (loads intermediate artifacts and produces plots/stats).
- `scripts/`: End-to-end pipeline scripts (see below).
- `data/`: Lightweight outputs and mappings saved by this repo (CSV).
  - `data/topic-label_all.csv`: topic labels generated from sampled chunks
  - `data/topic_cluster_map.csv`: topic -> cluster mapping
  - `data/cluster_labels.csv`: cluster labels
  - `data/convo_features_outdegree_enjoyable.csv`: merged conversation-level feature table

## Data dependencies (not in this repo)

The source data come from the CANDOR dataset (Reece et al., 2023), available via `https://betterup-data-requests.herokuapp.com/`. For convenience we compile the raw files into Parquet, but most scripts assume the CANDOR dataset lives at:

- `/project/ycleong/datasets/CANDOR`

Key expected inputs (Parquet) include:

- `transcript_backbiter.ALL.parquet` (turn-level transcript with timestamps)
- `survey.ALL.parquet` (includes `how_enjoyable`, `sex`)

Intermediate Parquets are written back into the same `CANDOR_DIR` as the pipeline runs (see “Pipeline”).

## Pipeline (scripts)

The main pipeline is implemented as numbered scripts under `scripts/`. A typical run order is:

1. `scripts/1_generate_embeddings.py`
   - Chunks conversations into word-bounded chunks and embeds each chunk with MPNet.
   - Writes chunk-level embeddings (e.g., `backbiter_chunk_embed.parquet`) under `CANDOR_DIR`.
2. `scripts/2_topic_modeling.py`
   - Fits BERTopic using precomputed embeddings and writes `chunk_topic.parquet` under `CANDOR_DIR`.
3. `scripts/3_label_topics.py`
   - Samples chunk text per topic and uses a local LLM to generate `short_label`, `summary`, and `keywords`.
   - Writes `data/topic-label_all.csv` (and includes mean topic embedding).
4. `scripts/4_rate_depth.py`
   - Adds `depth_score` per chunk (LLM-based, or `--no_llm` placeholder) and writes `chunk_topic_depth.parquet` under `CANDOR_DIR`.
5. `scripts/5_cluster_topics.py`
   - Clusters topics using their mean embeddings and writes `data/topic_cluster_map.csv`.
6. `scripts/6_label_clusters.py`
   - Uses a local LLM to label each topic-cluster and writes `data/cluster_labels.csv`.
7. `scripts/7_build_topic_network_cluster.py`
   - Builds per-conversation transition networks over `cluster_id`.
   - Writes `cluster_edges_by_convo.parquet` and `cluster_nodes_by_convo.parquet` under `CANDOR_DIR`.
8. `scripts/8_merge_outputs.py`
   - Merges network features with survey enjoyment and transcript-derived conversation length.
   - Writes `data/convo_features_outdegree_enjoyable.csv`.

## Running

### Local (interactive)

If you have access to the expected CANDOR parquet files and a Python environment with the required ML/NLP dependencies, you can run scripts directly, for example:

```bash
python3 scripts/8_merge_outputs.py
```

### Slurm / Midway-style run

`scripts/run.sh` is an example Slurm submission script that loads an Anaconda + CUDA module stack and runs the merge step:

```bash
sbatch scripts/run.sh
```

## Environment variables / `.env`

This repo supports an optional `.env` file (ignored by git) for values like `HF_TOKEN` when using gated Hugging Face models.

Expected format:

```bash
HF_TOKEN=...
```

## Notes

- The default labeling/depth scripts are designed to run **locally** using an open weights LLM (no paid API).
- Several scripts hardcode the CANDOR base path (`/project/ycleong/datasets/CANDOR`). If you are running elsewhere, update `CANDOR_DIR` in the relevant scripts.

