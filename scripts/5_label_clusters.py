#!/usr/bin/env python3
# 5_label_clusters.py
# Author: Yolanda Pan (xpan02@uchicago.edu)
# Last Edited: 2026/2/17
# Label clusters using LLM. Uses topic labels (short_label, summary, keywords) from
# topic-label_llama_all.csv to generate cluster-level labels.
# Runs locally on Midway GPU (no paid API, no HF inference). Default model is non-gated.
# Input: topic_cluster_map.csv (topic_id -> cluster_id) + topic-label_llama_all.csv (topic labels)
# Output: cluster_labels.csv with cluster_id, cluster_label, cluster_summary, cluster_keywords

import os
import sys
import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------- Env & API Key -----------------------------
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


# Default: non-gated model, fits on 1 GPU, no HF token needed.
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
model = None
tokenizer = None

# ----------------------------- Prompt -----------------------------

PROMPT = """
You are an expert annotator analyzing a group of related conversation topics that form a cluster.
All the topics below belong to the same cluster.

### Cluster ID: {cluster_id}

### Topics in this cluster
{topic_summaries}

### Task
Based on these topics, infer the overarching theme or category that unifies them.
Produce only a one-row Markdown table with:

- cluster_id: {cluster_id}
- cluster_label: a concise 2–5 word name for this cluster
- cluster_summary: one sentence describing the common theme or category
- cluster_keywords: 3–8 key words or phrases (comma separated)

### Output format (very important)
| cluster_id | cluster_label | cluster_summary | cluster_keywords |
|------------|---------------|-----------------|------------------|
| {cluster_id} | ... | ... | ... |

Do not add extra commentary.
""".strip()

SYSTEM = "You are an expert annotator analyzing groups of related conversation topics."


def _local_chat_completion(messages: list, max_tokens: int = 200, temperature: float = 0.0, top_p: float = 1.0) -> str:
    """Run chat completion locally with loaded model and tokenizer."""
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    reply_ids = out[0][input_ids.shape[1] :]
    return tokenizer.decode(reply_ids, skip_special_tokens=True).strip()


def format_topic_summaries(df_topics: pd.DataFrame) -> str:
    """Format topic labels into a readable summary block for the prompt."""
    lines = []
    for idx, row in df_topics.iterrows():
        topic_id = row.get("topic_id", idx)
        short_label = row.get("short_label", "N/A")
        summary = row.get("summary", "")
        keywords = row.get("keywords", "")
        
        parts = [f"Topic {topic_id}: {short_label}"]
        if summary:
            parts.append(f"Summary: {summary}")
        if keywords:
            parts.append(f"Keywords: {keywords}")
        lines.append("- " + " | ".join(parts))
    return "\n".join(lines)


def label_cluster(cluster_id: int, df_topics: pd.DataFrame, max_new_tokens: int = 200) -> str:
    """Generate cluster label using LLM based on topic labels."""
    topic_summaries = format_topic_summaries(df_topics)
    
    prompt = PROMPT.format(
        cluster_id=cluster_id,
        topic_summaries=topic_summaries,
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


def parse_cluster_table(llm_output: str) -> dict:
    """Parse LLM output table to extract cluster_id, cluster_label, cluster_summary, cluster_keywords."""
    # Look for markdown table
    table_match = re.search(
        r"\|.*cluster_id.*\|.*cluster_label.*\|.*cluster_summary.*\|.*cluster_keywords.*\|\s*\n"
        r"\|[-\s|]+\|\s*\n"
        r"\|[^\|]+\|[^\|]+\|[^\|]+\|[^\|]+\|",
        llm_output,
        re.MULTILINE | re.IGNORECASE,
    )
    
    if table_match:
        row = table_match.group(0).split("\n")[-1]  # Last line is data row
        parts = [p.strip() for p in row.split("|")[1:-1]]  # Skip empty first/last
        if len(parts) >= 4:
            return {
                "cluster_id": int(parts[0]),
                "cluster_label": parts[1],
                "cluster_summary": parts[2],
                "cluster_keywords": parts[3],
            }
    
    # Fallback: try to extract fields individually
    result = {}
    cluster_id_match = re.search(r"cluster_id[:\s]+(\d+)", llm_output, re.IGNORECASE)
    if cluster_id_match:
        result["cluster_id"] = int(cluster_id_match.group(1))
    
    label_match = re.search(r"cluster_label[:\s]+([^\n|]+)", llm_output, re.IGNORECASE)
    if label_match:
        result["cluster_label"] = label_match.group(1).strip()
    
    summary_match = re.search(r"cluster_summary[:\s]+([^\n|]+)", llm_output, re.IGNORECASE)
    if summary_match:
        result["cluster_summary"] = summary_match.group(1).strip()
    
    keywords_match = re.search(r"cluster_keywords[:\s]+([^\n|]+)", llm_output, re.IGNORECASE)
    if keywords_match:
        result["cluster_keywords"] = keywords_match.group(1).strip()
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Label clusters using LLM based on topic labels from topic-label_llama_all.csv"
    )
    parser.add_argument(
        "--cluster_map",
        type=Path,
        default=Path("data/topic_cluster_map.csv"),
        help="Input CSV: topic_id, cluster_id (and optional short_label).",
    )
    parser.add_argument(
        "--topic_labels",
        type=Path,
        default=Path("data/topic-label_llama_all.csv"),
        help="Input CSV: topic_id, short_label, summary, keywords.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/cluster_labels.csv"),
        help="Output CSV: cluster_id, cluster_label, cluster_summary, cluster_keywords.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"HF model ID (default: {DEFAULT_MODEL_ID}).",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=None,
        help="Local model path (bypasses HF, overrides --model_id).",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HF token (for gated models).",
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=None,
        help=f"Path to .env file (default: {_default_env_path}).",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit quantization (saves VRAM).",
    )
    parser.add_argument(
        "--min_cluster_id",
        type=int,
        default=None,
        help="Only label clusters >= this ID.",
    )
    parser.add_argument(
        "--max_cluster_id",
        type=int,
        default=None,
        help="Only label clusters <= this ID.",
    )
    parser.add_argument(
        "--max_topics_per_cluster",
        type=int,
        default=10,
        help="Max number of topics to show per cluster in prompt (default: 10).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose progress.",
    )
    args = parser.parse_args()

    # Step 1: Load model
    token = get_hf_token_optional(env_path=args.env, hf_token_arg=args.hf_token)
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    load_path = args.model_path if args.model_path else args.model_id
    from_hf = args.model_path is None

    if args.verbose:
        print(f"[verbose] Loading model: {load_path} (from HF: {from_hf})", file=sys.stderr)

    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(load_path, token=token if from_hf else None)
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
            sys.exit(f"--load_in_4bit requires bitsandbytes: pip install bitsandbytes ; {e}")

    model = AutoModelForCausalLM.from_pretrained(load_path, **load_kw)
    if args.verbose:
        print(f"[verbose] Model loaded on device: {next(model.parameters()).device}", file=sys.stderr)

    # Step 2: Load data
    cluster_map = pd.read_csv(args.cluster_map)
    topic_labels = pd.read_csv(args.topic_labels)

    if "cluster_id" not in cluster_map.columns or "topic_id" not in cluster_map.columns:
        raise ValueError("cluster_map must have columns: topic_id, cluster_id")
    
    required_cols = {"topic_id", "short_label", "summary", "keywords"}
    missing = required_cols - set(topic_labels.columns)
    if missing:
        raise ValueError(f"topic_labels missing required columns: {missing}")

    # Merge to get topic labels per cluster
    merged = cluster_map.merge(topic_labels, on="topic_id", how="inner")
    
    # Filter clusters
    clusters = sorted(merged["cluster_id"].dropna().unique().tolist())
    if args.min_cluster_id is not None:
        clusters = [c for c in clusters if c >= args.min_cluster_id]
    if args.max_cluster_id is not None:
        clusters = [c for c in clusters if c <= args.max_cluster_id]
    
    # Exclude cluster_id -1 (noise)
    clusters = [c for c in clusters if c != -1]

    if args.verbose:
        print(f"[verbose] Clusters to label: {len(clusters)}", file=sys.stderr)

    rows = []
    for cluster_id in clusters:
        cluster_topics = merged[merged["cluster_id"] == cluster_id]
        
        # Limit topics shown in prompt
        if len(cluster_topics) > args.max_topics_per_cluster:
            cluster_topics = cluster_topics.sample(n=args.max_topics_per_cluster, random_state=42)
        
        if args.verbose:
            print(f"[verbose] Labeling cluster {cluster_id} ({len(cluster_topics)} topics)...", file=sys.stderr)
        
        try:
            llm_output = label_cluster(cluster_id, cluster_topics)
            parsed = parse_cluster_table(llm_output)
            
            if "cluster_id" not in parsed:
                parsed["cluster_id"] = cluster_id
            
            rows.append(parsed)
            if args.verbose:
                print(f"[verbose] Cluster {cluster_id}: {parsed.get('cluster_label', 'N/A')}", file=sys.stderr)
        except Exception as e:
            print(f"[ERROR] Failed to label cluster {cluster_id}: {e}", file=sys.stderr)
            rows.append({
                "cluster_id": cluster_id,
                "cluster_label": "ERROR",
                "cluster_summary": f"Error: {str(e)}",
                "cluster_keywords": "",
            })

    # Output
    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values("cluster_id").reset_index(drop=True)
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(out_df)} cluster labels to {args.output}")


if __name__ == "__main__":
    main()
