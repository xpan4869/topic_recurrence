# 3_label_topics_llama.py
# Author: Yolanda Pan (xpan02@uchicago.edu)
# Last Edited: 2026/1/14

# Random-sample chunks within each BERTopic topic and use an LLM to generate:
    # - short_label (2–5 words)
    # - one-sentence summary
    # - keywords (3–8 comma-separated)
# Input: chunk_topic-num.parquet  (must contain: topic, chunk_text)
# Output: topic-label_all.csv

import os
import sys
import argparse
from pathlib import Path

import pandas as pd
import openai

from parquet_helper import read_parquet_any

# ----------------------------- Env & API Key -----------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    sys.exit("[ERROR] OPENAI_API_KEY not found. Put it in .env or export it before running.")

client = openai.OpenAI(api_key=api_key)


# ----------------------------- Prompt -----------------------------

PROMPT = """
You are an expert annotator analyzing a latent conversation topic.
All the text chunks below come from the same topic.

### Topic ID: {topic_id}

### Example text chunks
{chunk_examples}

### Task
Based on these examples, infer the underlying topic.
Produce only a one-row Markdown table with:

- topic_id: {topic_id}
- short_label: a concise 2–5 word name
- summary: one sentence describing what people are doing or discussing in this topic
- keywords: 3–8 key words or phrases (comma separated)

### Output format (very important)
| topic_id | short_label | summary | keywords |
|----------|-------------|---------|----------|
| {topic_id} | ... | ... | ... |

Do not add extra commentary.
""".strip()


# ----------------------------- Helpers -----------------------------
def sample_topic_texts(group: pd.DataFrame, n: int = 15, seed: int = 42) -> list[str]:
    return (
        group["chunk_text"]
        .dropna()
        .astype(str)
        .sample(n=min(n, len(group)), random_state=seed)
        .tolist()
    )


def label_topic(topic_id: int, chunk_examples: list[str], model: str = "gpt-4o") -> str:
    chunk_examples_block = "\n".join(
        f"- {i+1}. {text}" for i, text in enumerate(chunk_examples)
    )

    prompt = PROMPT.format(
        topic_id=topic_id,
        chunk_examples=chunk_examples_block,
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return resp.choices[0].message.content


def parse_topic_table(gpt_output: str) -> dict:
    """
    Parse the one-row Markdown table output into a dict:
    {topic_id:int, short_label:str, summary:str, keywords:str}
    """
    lines = [ln.strip() for ln in gpt_output.strip().splitlines() if ln.strip()]

    # Find the first data row that starts with "|" and is not header/separator
    for ln in lines:
        if not ln.startswith("|"):
            continue
        if ln.lower().startswith("| topic_id"):
            continue
        if set(ln.replace("|", "").strip()) <= {"-", " "}:
            continue

        parts = [p.strip() for p in ln.strip("|").split("|")]
        if len(parts) != 4:
            continue

        topic_id, short_label, summary, keywords = parts
        return {
            "topic_id": int(topic_id),
            "short_label": short_label,
            "summary": summary,
            "keywords": keywords,
        }

    raise ValueError("Could not parse a valid one-row topic table from model output.")


# ----------------------------- Main -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Sample chunks per topic and label topics with an LLM.")
    parser.add_argument(
        "--in_parquet",
        type=str,
        default="/project/macs40123/yolanda/candor_parquet/chunk_topic-num.parquet",
        help="Path to chunk_topic-num.parquet",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="topic-label_all.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--n_per_topic",
        type=int,
        default=15,
        help="Number of chunks to sample per topic",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model name for labeling",
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
    args = parser.parse_args()

    df = read_parquet_any(args.in_parquet)

    required = {"topic", "chunk_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input parquet: {missing}")

    topics = sorted(df["topic"].dropna().unique().tolist())

    if args.min_topic_id is not None:
        topics = [t for t in topics if t >= args.min_topic_id]
    if args.max_topic_id is not None:
        topics = [t for t in topics if t <= args.max_topic_id]

    rows = []
    for t in topics:
        group = df[df["topic"] == t]
        examples = sample_topic_texts(group, n=args.n_per_topic, seed=args.seed)

        # If a topic has no usable text, skip it
        if len(examples) == 0:
            continue

        gpt_output = label_topic(int(t), examples, model=args.model)
        row = parse_topic_table(gpt_output)
        rows.append(row)
    
        print(f"Labeled topic {t}: {row['short_label']}")

    out_df = pd.DataFrame(rows).sort_values("topic_id").reset_index(drop=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
