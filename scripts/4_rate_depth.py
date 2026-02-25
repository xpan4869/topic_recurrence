# 4_rate_depth.py
# Author: Yolanda Pan
# After 2_topic_modeling.py: add depth_score per chunk.
# Reads chunk_topic.parquet from CANDOR_DIR, computes depth_score per row (LLM or placeholder),
# writes to CANDOR_DIR/chunk_topic_depth.parquet (same schema + depth column).

import os
import re
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from parquet_helper import read_parquet_any, write_parquet_any


CANDOR_DIR = Path("/project/ycleong/datasets/CANDOR")
CHUNK_TOPIC_FILE = "chunk_topic.parquet"
CHUNK_TOPIC_DEPTH_FILE = "chunk_topic_depth.parquet"
# Default: non-gated model, fits on 1 GPU, no HF token needed.
DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

# Same prompt as in demo1.ipynb (Test depth prompt section)
DEPTH_PROMPT = """You are rating the depth of a conversation.

### Conversation Chunk
{chunk_text}

### Definition of Depth
We define deep conversations as those in which people engage in self-disclosure by revealing personally intimate information about their thoughts, feelings, or experiences.

### Rating Scale (1–7)
1 = completely shallow, no self-disclosure  
7 = highly intimate, vulnerable, and personally revealing  

Higher numbers indicate greater personal self-disclosure and reflection.

### Task
Select one integer from 1 to 7 that best reflects the overall depth of this chunk.

Reply with exactly:
depth_score: <integer>
Do not add any explanation.
"""

def build_depth_prompt(chunk_text: str, max_chars: int = 1500) -> str:
    return DEPTH_PROMPT.format(chunk_text=(chunk_text or "").strip()[:max_chars])


def compute_depth_placeholder(df: pd.DataFrame, text_col: str = "chunk_text") -> pd.Series:
    """Placeholder: return a simple depth proxy (0–1) when not using LLM."""
    if text_col not in df.columns:
        return pd.Series(0.0, index=df.index)
    lengths = df[text_col].fillna("").astype(str).str.len()
    return (lengths / (lengths.max() or 1)).clip(0, 1).astype("float32")


def parse_depth_output(llm_output: str):
    """
    Parse 'depth_score: <1-7>' from model output.
    Return integer 1–7, or None if parsing fails.
    """
    if not llm_output or not isinstance(llm_output, str):
        return None

    # Look specifically for depth_score pattern first
    m = re.search(r"depth_score\s*:\s*([1-7])\b", llm_output, re.IGNORECASE)
    if m:
        return int(m.group(1))

    # Fallback: if model outputs just a single digit like "5"
    m2 = re.fullmatch(r"\s*([1-7])\s*", llm_output.strip())
    if m2:
        return int(m2.group(1))

    return None


# ----------------------------- LLM (optional) -----------------------------
model = None
tokenizer = None


def _load_env(path: Path) -> None:
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


def _local_chat_completion(messages: list, max_tokens: int = 80, temperature: float = 0.0) -> str:
    import torch
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
            pad_token_id=tokenizer.eos_token_id,
        )
    reply_ids = out[0][input_ids.shape[1] :]
    return tokenizer.decode(reply_ids, skip_special_tokens=True).strip()


def rate_chunk_depth_llm(chunk_text: str, max_new_tokens: int = 80) -> Optional[float]:
    """Call LLM and return depth in [0, 1] or None on failure."""
    prompt = build_depth_prompt(chunk_text)
    messages = [
        {"role": "system", "content": "You are an expert annotator rating conversation depth."},
        {"role": "user", "content": prompt},
    ]
    out = _local_chat_completion(messages, max_tokens=max_new_tokens)
    return parse_depth_output(out)


# ----------------------------- Main -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Add depth_score to chunk_topic; writes CANDOR_DIR/chunk_topic_depth.parquet.")
    parser.add_argument(
        "--parquet",
        type=Path,
        default=CANDOR_DIR / CHUNK_TOPIC_FILE,
        help="Path to chunk_topic.parquet (input)",
    )
    parser.add_argument(
        "--out_parquet",
        type=Path,
        default=CANDOR_DIR / CHUNK_TOPIC_DEPTH_FILE,
        help="Output parquet path (default: CANDOR_DIR/chunk_topic_depth.parquet)",
    )
    parser.add_argument(
        "--no_llm",
        action="store_true",
        help="Use length-based placeholder instead of LLM (no GPU needed)",
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
        help="Load model from local dir. Overrides --model_id.",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit (saves VRAM; requires bitsandbytes)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to .env for HF_TOKEN",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HF token for gated model (overrides .env)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=80,
        help="Max new tokens per chunk for depth reply",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress",
    )
    args = parser.parse_args()

    # Input: read from topic file
    in_path = args.parquet
    if not in_path.exists():
        raise FileNotFoundError(f"Missing: {in_path}")

    df = read_parquet_any(str(in_path))
    required = {"chunk_id", "conversation_id", "chunk_text", "topic"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Output: CANDOR_DIR/chunk_topic_depth.parquet by default
    out_path = args.out_parquet
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.no_llm:
        df["depth_score"] = compute_depth_placeholder(df, "chunk_text")
        if args.verbose:
            print("Using placeholder depth (--no_llm)")
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from tqdm import tqdm

        load_path = args.model_path or args.model_id or DEFAULT_MODEL_ID
        from_hf = args.model_path is None
        token = None
        if from_hf:
            _load_env(Path(args.env) if args.env else Path("/home/xpan02/topic_recurrence/.env"))
            token = args.hf_token or os.getenv("HF_TOKEN")
            if token:
                os.environ["HF_TOKEN"] = token
        global model, tokenizer
        if args.verbose:
            print("Loading model:", load_path)
        tokenizer = AutoTokenizer.from_pretrained(load_path, token=token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        load_kw = {"device_map": "auto", "torch_dtype": torch.bfloat16}
        if from_hf:
            load_kw["token"] = token
        if args.load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        model = AutoModelForCausalLM.from_pretrained(load_path, **load_kw)

        texts = df["chunk_text"].fillna("").astype(str).tolist()
        scores = []
        for text in tqdm(texts, desc="depth", disable=not args.verbose):
            s = rate_chunk_depth_llm(text, max_new_tokens=args.max_new_tokens)
            if s is None:
                s = 0.0  # fallback on parse failure
            scores.append(s)
        df["depth_score"] = pd.Series(scores, dtype="float32")

    write_parquet_any(df, str(out_path))
    print("Done. Read from:", in_path)
    print("      Saved to:", out_path)


if __name__ == "__main__":
    main()
