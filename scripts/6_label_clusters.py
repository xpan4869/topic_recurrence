# 6_label_clusters.py
# Author: Yolanda Pan (xpan02@uchicago.edu)
# Last Edited: 2026/2/17
# Label clusters using LLM. Uses topic labels (short_label, summary, keywords) from
# topic-label_all.csv to generate cluster-level labels.
# Runs locally on Midway GPU (no paid API, no HF inference). Default model is non-gated.
# Input: topic_cluster_map.csv (topic_id -> cluster_id) + topic-label_all.csv (topic labels)
# Output: cluster_labels.csv with cluster_id, cluster_label, cluster_summary, cluster_keywords

import os
import sys
import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd

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


def label_cluster(cluster_id: int, df_topics: pd.DataFrame, max_new_tokens: int = 512) -> str:
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


def placeholder_cluster_label(cluster_id: int, df_topics: pd.DataFrame) -> dict:
    """Quick test: no LLM. Build cluster_label/summary/keywords from topic short_labels and keywords."""
    labels = df_topics["short_label"].dropna().astype(str).tolist()
    kw = df_topics["keywords"].dropna().astype(str).tolist()
    # First 5 labels as a "name", first 3 summaries or labels for summary, first 8 keyword phrases
    cluster_label = ", ".join(labels[:3]) if labels else f"Cluster {cluster_id}"
    if len(cluster_label) > 60:
        cluster_label = cluster_label[:57] + "..."
    parts = labels[:5]
    cluster_summary = ". ".join(parts) if parts else f"Placeholder summary for cluster {cluster_id}."
    if len(cluster_summary) > 200:
        cluster_summary = cluster_summary[:197] + "..."
    all_kw = []
    for s in kw[:5]:
        all_kw.extend([x.strip() for x in str(s).split(",") if x.strip()][:3])
    cluster_keywords = ", ".join(all_kw[:10]) if all_kw else "placeholder"
    return {
        "cluster_id": cluster_id,
        "cluster_label": cluster_label,
        "cluster_summary": cluster_summary,
        "cluster_keywords": cluster_keywords,
    }


def parse_cluster_table(llm_output: str) -> dict:
    """Parse LLM output table to extract cluster_id, cluster_label, cluster_summary, cluster_keywords."""
    # Normalize: single newlines, strip
    text = ("\n".join(line.strip() for line in llm_output.strip().splitlines())).strip()

    # 1) Look for markdown table (flexible: optional header/separator, any line with 4 pipe cells)
    # Data row: | id | label | summary | keywords |
    data_row_match = re.search(
        r"\|?\s*(\d+)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if data_row_match:
        label = data_row_match.group(2).strip()
        summary = data_row_match.group(3).strip()
        keywords = data_row_match.group(4).strip()
        # Only accept row if at least label is non-empty (avoid treating "| 5 | | | |" as valid)
        if label or summary or keywords:
            return {
                "cluster_id": int(data_row_match.group(1)),
                "cluster_label": label,
                "cluster_summary": summary,
                "cluster_keywords": keywords,
            }

    # 2) Fallback: key-style (cluster_label: ..., etc.)
    result = {}
    cluster_id_match = re.search(r"cluster_id[:\s]+(\d+)", text, re.IGNORECASE)
    if cluster_id_match:
        result["cluster_id"] = int(cluster_id_match.group(1))

    for key, name in [
        ("cluster_label", "cluster_label"),
        ("cluster_summary", "cluster_summary"),
        ("cluster_keywords", "cluster_keywords"),
    ]:
        # Match "cluster_label: value" or "cluster_label: value" across line break
        m = re.search(
            rf"{name}[:\s]+(.+?)(?=\n\s*(?:cluster_\w+|$))",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if m:
            result[key] = m.group(1).strip()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Label clusters using LLM based on topic labels from topic-label_all.csv"
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
        default=Path("data/topic-label_all.csv"),
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
        default=None,
        help="Deprecated: all topics are now included (no sampling).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Max new tokens per cluster (default 512; increase if output is truncated).",
    )
    parser.add_argument(
        "--placeholder",
        action="store_true",
        help="Quick test: no LLM, build labels from topic short_label/keywords (pipeline only).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose progress.",
    )
    args = parser.parse_args()

    # Step 1: Load model (skip if placeholder)
    if not args.placeholder:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        globals()["torch"] = torch  # so _local_chat_completion can use it

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
    else:
        if args.verbose:
            print("[verbose] Placeholder mode: no model loaded", file=sys.stderr)

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
        
        # Use all topics (no sampling)
        if args.verbose:
            print(f"[verbose] Labeling cluster {cluster_id} ({len(cluster_topics)} topics)...", file=sys.stderr)
        
        if args.placeholder:
            parsed = placeholder_cluster_label(cluster_id, cluster_topics)
            rows.append(parsed)
            if args.verbose:
                print(f"[verbose] Cluster {cluster_id}: {parsed.get('cluster_label', 'N/A')}", file=sys.stderr)
        else:
            try:
                llm_output = label_cluster(cluster_id, cluster_topics, max_new_tokens=args.max_new_tokens)
                parsed = parse_cluster_table(llm_output)

                if "cluster_id" not in parsed:
                    parsed["cluster_id"] = cluster_id

                # Ensure all columns present; treat empty parsed fields as parse failure
                has_label = (
                    parsed.get("cluster_label") and str(parsed.get("cluster_label", "")).strip()
                )
                if not has_label:
                    if args.verbose:
                        print(
                            f"[verbose] Cluster {cluster_id}: LLM output empty or unparseable; using placeholder. raw snippet: {llm_output[:200]!r}...",
                            file=sys.stderr,
                        )
                    parsed = placeholder_cluster_label(cluster_id, cluster_topics)
                else:
                    for col in ("cluster_label", "cluster_summary", "cluster_keywords"):
                        if col not in parsed or parsed[col] is None:
                            parsed[col] = ""
                        else:
                            parsed[col] = str(parsed[col]).strip()

                rows.append(parsed)
                if args.verbose:
                    print(f"[verbose] Cluster {cluster_id}: {parsed.get('cluster_label', 'N/A')}", file=sys.stderr)
            except Exception as e:
                print(f"[ERROR] Failed to label cluster {cluster_id}: {e}", file=sys.stderr)
                try:
                    parsed = placeholder_cluster_label(cluster_id, cluster_topics)
                    rows.append(parsed)
                except Exception:
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
