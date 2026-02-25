# 3_label_topics.py
# Author: Yolanda Pan (xpan02@uchicago.edu)
# Last Edited: 2026/2/16

# Random-sample chunks within each BERTopic topic and use an LLM to generate:
#   short_label (2–5 words), one-sentence summary, keywords (3–8 comma-separated)
# Runs locally (no paid API): uses GPU when available (device_map="auto"), else CPU.
# For sbatch on Midway3: request a GPU (e.g. --partition=gpu --gres=gpu:1)
# Default model is non-gated so no HF token needed. Optional: --model_path to load from local dir (no HF at all).
# Input: chunk_topic.parquet from 2_topic_modeling (topic, chunk_text, embedding = MPNet).
# Output: topic-label_all.csv with topic_id, short_label, summary, keywords, topic_embedding (avg MPNet).

import os
import sys
import argparse
from ast import literal_eval
from pathlib import Path
from typing import Optional

import re
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from parquet_helper import read_parquet_any

# ----------------------------- Env & API Key -----------------------------
# Paths derived from this script's location 
project_root = Path("/home/xpan02/topic_recurrence")
script_dir = Path(project_root / "scripts")
_default_env_path = project_root / ".env"

def _load_env(path: Path) -> None:
    """Read KEY=VALUE lines from path and set os.environ."""
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            if key:
                os.environ[key] = value


def get_hf_token_optional(env_path: Optional[str] = None, hf_token_arg: Optional[str] = None) -> Optional[str]:
    """HF token if available (from --hf_token or .env). Not required for non-gated models."""
    if hf_token_arg:
        return hf_token_arg
    path = Path(env_path) if env_path else _default_env_path
    _load_env(path)
    return os.getenv("HF_TOKEN") or None

DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
model = None
tokenizer = None

# ----------------------------- Prompt -----------------------------

PROMPT = """
You are labeling a latent conversation topic.

All text chunks below belong to the same topic.

Topic ID: {topic_id}

Examples:
{chunk_examples}

Task:
Infer the shared underlying theme.

Output ONLY one Markdown table row:

| topic_id | short_label | summary | keywords |
|----------|-------------|---------|----------|
| {topic_id} | 2–5 word specific label | one clear sentence | 3–8 comma-separated keywords |

Rules:
- Be specific (avoid vague labels like "general discussion")
- Base the label only on the examples
- No extra text
""".strip()


# ----------------------------- Helpers -----------------------------
def _ensure_embedding_arrays(df: pd.DataFrame, col: str) -> np.ndarray:
    """Convert col to (n, dim) float32 array. Handles list/array or string '[0.1, ...]'."""
    if len(df) == 0:
        raise ValueError("DataFrame is empty; cannot infer embedding dtype.")
    x0 = df[col].iloc[0]
    if isinstance(x0, str):
        arr = np.array([np.array(literal_eval(x), dtype=np.float32) for x in df[col]], dtype=np.float32)
    else:
        arr = np.vstack(df[col].apply(lambda x: np.asarray(x, dtype=np.float32)))
    return arr


def compute_topic_mean_embeddings(
    df: pd.DataFrame,
    embedding_parquet: Optional[str] = None,
    embed_col: str = "chunk_vector",
) -> Optional[pd.DataFrame]:
    """
    Return a DataFrame with topic_id and one column topic_embedding (mean MPNet embedding per topic).
    Value is a string "[float, float, ...]" like the embedding column in chunk_embed.parquet.
    Uses df's 'embedding' column if present (from chunk_topic.parquet), else merges with
    embedding_parquet on (conversation_id, chunk_id). Returns None if no embeddings available.
    """
    if df.empty:
        return None
    # Prefer embedding in main df (chunk_topic.parquet from 2_topic_modeling)
    if "embedding" in df.columns:
        col = "embedding"
    elif embed_col in df.columns:
        col = embed_col
    else:
        col = None
    if col is not None:
        df = df.copy()
        emb_arr = _ensure_embedding_arrays(df, col)
        df["_emb"] = list(emb_arr)
        topic_means = df.groupby("topic")["_emb"].apply(lambda x: np.mean(np.vstack(x), axis=0)).reset_index()
    elif embedding_parquet:
        emb_df = read_parquet_any(embedding_parquet)
        if emb_df.empty:
            return None
        if embed_col not in emb_df.columns and "embedding" in emb_df.columns:
            embed_col_use = "embedding"
        else:
            embed_col_use = embed_col
        if embed_col_use not in emb_df.columns:
            return None
        join_cols = [c for c in ["conversation_id", "chunk_id"] if c in df.columns and c in emb_df.columns]
        if not join_cols:
            return None
        emb_arr = _ensure_embedding_arrays(emb_df, embed_col_use)
        emb_df["_emb"] = list(emb_arr)
        merged = df[["topic"] + join_cols].merge(
            emb_df[join_cols + ["_emb"]], on=join_cols, how="inner"
        )
        topic_means = merged.groupby("topic")["_emb"].apply(lambda x: np.mean(np.vstack(x), axis=0)).reset_index()
    else:
        return None
    # One column: topic_embedding as "[0.1, -0.2, ...]" so CSV is parseable with literal_eval
    topic_means["topic_embedding"] = topic_means["_emb"].apply(lambda v: str([float(x) for x in v]))
    topic_means = topic_means.rename(columns={"topic": "topic_id"}).drop(columns=["_emb"])
    return topic_means


def sample_topic_texts(group: pd.DataFrame, n: int = 15, seed: int = 42) -> list[str]:
    return (
        group["chunk_text"]
        .dropna()
        .astype(str)
        .sample(n=min(n, len(group)), random_state=seed)
        .tolist()
    )


SYSTEM = "You are an expert annotator analyzing topics in conversations."


def _local_chat_completion(messages: list, max_tokens: int = 100, temperature: float = 0.0, top_p: float = 1.0) -> str:
    """Run chat completion locally with loaded model and tokenizer."""
    device = getattr(model, "device", next(model.parameters()).device)
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
    ).to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the new tokens
    reply_ids = out[0][input_ids.shape[1] :]
    return tokenizer.decode(reply_ids, skip_special_tokens=True).strip()


def label_topic(topic_id: int, chunk_examples: list[str], max_new_tokens: int = 100) -> str:
    chunk_examples_block = "\n".join(
        f"- {i+1}. {text}" for i, text in enumerate(chunk_examples)
    )

    prompt = PROMPT.format(
        topic_id=topic_id,
        chunk_examples=chunk_examples_block,
    )

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ]

    return _local_chat_completion(
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
    )


def parse_topic_table(gpt_output: str) -> dict:
    """
    Parse model output that should contain a Markdown table row:
    | topic_id | short_label | summary | keywords |

    Robust to:
    - keywords spilling onto the next row
    - extra pipes / empty columns
    - topic_id like "3 " or "-1"
    - summary/keywords containing stray pipes
    - minor formatting deviations
    """
    # Keep only non-empty lines
    lines = [ln.rstrip() for ln in gpt_output.strip().splitlines() if ln.strip()]

    # Extract "table-ish" rows (start with |)
    rows = []
    for ln in lines:
        s = ln.strip()
        if not s.startswith("|"):
            continue
        low = s.lower()

        # Skip header and separator rows
        if "topic_id" in low and "short_label" in low:
            continue
        if set(s.replace("|", "").strip()) <= {"-", " "}:
            continue

        # Split into cells; keep empties so we can detect "spill" rows
        cells = [c.strip() for c in s.strip().strip("|").split("|")]
        rows.append(cells)

    if not rows:
        raise ValueError(f"No Markdown table rows found. Model output was:\n{gpt_output}")

    def is_int_like(x: str) -> bool:
        return bool(re.fullmatch(r"-?\d+", x.strip()))

    # Find the first row that contains a topic_id somewhere
    start_idx = None
    topic_id = None

    for i, cells in enumerate(rows):
        # Sometimes there are extra columns; topic_id is usually the first non-empty cell
        for j, cell in enumerate(cells):
            if is_int_like(cell):
                start_idx = i
                topic_id = int(cell.strip())
                # We'll assume this row is the "main" row and topic_id is intended as first column.
                # If it's not in position 0, we will realign below.
                break
        if start_idx is not None:
            break

    if start_idx is None:
        raise ValueError(f"Could not find a topic_id integer in table rows. Output was:\n{gpt_output}")

    # Build a stream of cells starting from the row with the topic_id,
    # merging subsequent "spill" rows if needed.
    stream = []

    # Realign: if topic_id isn't in col0, pad empties to shift it into place
    main = rows[start_idx]
    # locate topic_id position
    tid_pos = next((k for k, c in enumerate(main) if is_int_like(c)), 0)
    if tid_pos > 0:
        main = [""] * tid_pos + main  # shift so tid is later; we will pick first int anyway

    # Add main row cells
    stream.extend(main)

    # If we don't yet have enough info, append following rows
    for r in rows[start_idx + 1:]:
        stream.extend(r)
        # stop early if we've accumulated plenty
        if len(stream) >= 12:
            break

    # Now, extract the 4 target fields.
    # We’ll:
    # 1) find the first int as topic_id
    # 2) take next non-empty cell as short_label
    # 3) then next non-empty as summary
    # 4) then the rest (joined) as keywords
    # This handles keywords spilling and extra columns.
    # Also handles cases where summary/keywords had extra split due to pipes: we re-join tail.
    def next_nonempty(idx: int) -> tuple[str, int]:
        while idx < len(stream) and stream[idx].strip() == "":
            idx += 1
        return (stream[idx].strip() if idx < len(stream) else ""), idx

    # 1) topic_id
    idx = 0
    while idx < len(stream) and not is_int_like(stream[idx]):
        idx += 1
    if idx >= len(stream):
        raise ValueError(f"Could not locate topic_id in merged cell stream. Output was:\n{gpt_output}")
    topic_id = int(stream[idx].strip())
    idx += 1

    # 2) short_label
    short_label, idx = next_nonempty(idx)
    if not short_label:
        raise ValueError(f"Missing short_label. Output was:\n{gpt_output}")
    idx += 1

    # 3) summary
    summary, idx = next_nonempty(idx)
    if not summary:
        raise ValueError(f"Missing summary. Output was:\n{gpt_output}")
    idx += 1

    # 4) keywords = everything remaining that's non-empty, joined back with " | " if needed
    keywords_parts = [c.strip() for c in stream[idx:] if c.strip() != ""]
    keywords = " ".join(keywords_parts).strip()

    # Clean common trailing punctuation artifacts
    keywords = keywords.strip().strip("|").strip()

    return {
        "topic_id": topic_id,
        "short_label": short_label,
        "summary": summary,
        "keywords": keywords,
    }

# ----------------------------- Main -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Sample chunks per topic and label topics with an LLM.")
    parser.add_argument(
        "--in_parquet",
        type=str,
        default="/project/ycleong/datasets/CANDOR/chunk_topic.parquet",
        help="Path to chunk_topic.parquet",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="/home/xpan02/topic_recurrence/data/topic-label_all.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--n_per_topic",
        type=int,
        default=8,
        help="Number of chunks to sample per topic",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--embedding_parquet",
        type=str,
        default=None,
        metavar="PATH",
        help="Parquet with chunk embeddings (conversation_id, chunk_id, chunk_vector) to merge and average per topic",
    )
    parser.add_argument(
        "--embedding_col",
        type=str,
        default="chunk_vector",
        help="Embedding column name in --in_parquet or --embedding_parquet (default: chunk_vector)",
    )
    parser.add_argument(
        "--min_topic_id",
        type=int,
        default=None,
        help="Optional: only label topics >= this id",
    )
    parser.add_argument(
        "--max_topic_id",
        type=int,
        default=None,
        help="Optional: only label topics <= this id",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Hugging Face model id (default: %s). Ignored if --model_path set." % DEFAULT_MODEL_ID,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        metavar="DIR",
        help="Load model from local dir (no Hugging Face). Overrides --model_id.",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit (saves VRAM; use for 8B+ on 24GB GPU to avoid OOM; requires bitsandbytes)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        metavar="PATH",
        help="Direct path to .env file (e.g. /home/user/topic_recurrence/.env)",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HF token for gated model download (overrides .env)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print step-by-step debug info to stderr",
    )
    args = parser.parse_args()

    load_path = args.model_path or args.model_id or DEFAULT_MODEL_ID
    from_hf = args.model_path is None

    if args.verbose:
        print("[verbose] Step 1: ARGS parsed. in_parquet=%s out_csv=%s load_path=%s" % (args.in_parquet, args.out_csv, load_path), file=sys.stderr)

    token = get_hf_token_optional(env_path=args.env, hf_token_arg=args.hf_token) if from_hf else None
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    global model, tokenizer
    if args.verbose:
        print("[verbose] Step 2: Loading model and tokenizer from %s (local GPU, no API cost)" % load_path, file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(load_path, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_kw = {"device_map": "auto", "torch_dtype": torch.bfloat16}
    if from_hf:
        load_kw["token"] = token
    if args.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            load_kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        except Exception as e:
            sys.exit("--load_in_4bit requires bitsandbytes: pip install bitsandbytes ; %s" % e)
    model = AutoModelForCausalLM.from_pretrained(load_path, **load_kw)
    if args.verbose:
        print("[verbose] Model loaded on device: %s" % str(next(model.parameters()).device), file=sys.stderr)

    # Step 3: Read input parquet
    if args.verbose:
        print("[verbose] Step 3: Reading parquet: %s" % args.in_parquet, file=sys.stderr)
    df = read_parquet_any(args.in_parquet)

    required = {"topic", "chunk_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input parquet: {missing}")

    # Average embedding per topic (if embeddings available in df or via --embedding_parquet)
    topic_mean_embeddings = compute_topic_mean_embeddings(
        df,
        embedding_parquet=args.embedding_parquet,
        embed_col=args.embedding_col,
    )

    topics = sorted(df["topic"].dropna().unique().tolist())
    if args.min_topic_id is not None:
        topics = [t for t in topics if t >= args.min_topic_id]
    if args.max_topic_id is not None:
        topics = [t for t in topics if t <= args.max_topic_id]

    if args.verbose:
        print("[verbose] Step 4: Topics to label: %d (ids %s ... %s)" % (len(topics), topics[0] if topics else "n/a", topics[-1] if topics else "n/a"), file=sys.stderr)

    rows = []
    for t in topics:
        group = df[df["topic"] == t]
        examples = sample_topic_texts(group, n=args.n_per_topic, seed=args.seed)

        # If a topic has no usable text, skip it
        if len(examples) == 0:
            continue

        model_output = label_topic(int(t), examples, max_new_tokens=args.max_new_tokens)
        row = parse_topic_table(model_output)
        rows.append(row)
    
        print(f"Labeled topic {t}: {row['short_label']}")

    out_df = pd.DataFrame(rows)
    if len(out_df) == 0:
        out_df = pd.DataFrame(columns=["topic_id", "short_label", "summary", "keywords"])
    else:
        out_df = out_df.sort_values("topic_id").reset_index(drop=True)
    if topic_mean_embeddings is not None:
        out_df = out_df.merge(topic_mean_embeddings, on="topic_id", how="left")
    out_df.to_csv(args.out_csv, index=False)
    if args.verbose:
        print("[verbose] Step 5: Wrote %d rows to %s" % (len(out_df), args.out_csv), file=sys.stderr)
    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
