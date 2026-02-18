# 2b_rate_depth.py
# Author: Yolanda Pan
# After 2_topic_modeling.py: add depth_score per chunk to chunk_topic.parquet (in place).
# Reads chunk_topic.parquet (chunk_id, conversation_id, chunk_text, topic, embedding),
# computes depth_score per row (LLM or placeholder), writes back to the same file.
# No new file in CANDOR.
# Prompt matches demo1.ipynb "Test depth prompt (for 2b_rate_depth.py)" section.

import os
import re
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from parquet_helper import read_parquet_any, write_parquet_any


CANDOR_DIR = Path("/project/ycleong/datasets/CANDOR")
CHUNK_TOPIC_FILE = "chunk_topic.parquet"
# Default: non-gated model, fits on 1 GPU, no HF token needed.
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# Same prompt as in demo1.ipynb (Test depth prompt section)
DEPTH_PROMPT = """You are an expert annotator rating conversation depth.

### Chunk to rate
{chunk_text}

### Task
Rate the depth of this conversation chunk on a scale from 1 to 7.

Depth refers to the extent to which speakers engage in:
- self-disclosure about internal states (thoughts, feelings, experiences)
- reflective processing (explaining meaning, causes, changes, identity)
- substantive intellectual or emotional engagement

Use the following anchors:

1 = Purely surface-level (greetings, logistics, weather, coordination, filler)
2 = Mostly factual or transactional; no internal states expressed
3 = Mild personal opinions or preferences; limited elaboration
4 = Some personal experience mentioned but little reflection
5 = Clear self-disclosure of thoughts or feelings
6 = Reflective processing (explaining why something matters or what it means)
7 = Deep reflection or vulnerability involving identity, life meaning, or emotional insight

Important:
- Emotional intensity alone does not equal depth.
- Topic seriousness alone does not equal depth.
- Rate based on reflective or self-referential engagement.

Reply with exactly two lines:
depth_score: <integer from 1 to 7>
reason: <one short phrase describing why>
"""


def build_depth_prompt(chunk_text: str, max_chars: int = 1500) -> str:
    return DEPTH_PROMPT.format(chunk_text=(chunk_text or "").strip()[:max_chars])


def compute_depth_placeholder(df: pd.DataFrame, text_col: str = "chunk_text") -> pd.Series:
    """Placeholder: return a simple depth proxy (0–1) when not using LLM."""
    if text_col not in df.columns:
        return pd.Series(0.0, index=df.index)
    lengths = df[text_col].fillna("").astype(str).str.len()
    return (lengths / (lengths.max() or 1)).clip(0, 1).astype("float32")


def parse_depth_output(llm_output: str) -> Optional[float]:
    """
    Parse "depth_score: <1-7>" from model output. Return value in [0, 1] (normalized from 1–7),
    or None if parse fails.
    """
    if not llm_output or not isinstance(llm_output, str):
        return None
    m = re.search(r"depth_score\s*:\s*(\d+)", llm_output.strip(), re.IGNORECASE)
    if not m:
        return None
    raw = int(m.group(1))
    if raw < 1 or raw > 7:
        return None
    return (raw - 1) / 6.0  # 1->0, 7->1


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
    parser = argparse.ArgumentParser(description="Add depth_score to chunk_topic.parquet (in place).")
    parser.add_argument(
        "--parquet",
        type=Path,
        default=CANDOR_DIR / CHUNK_TOPIC_FILE,
        help="Path to chunk_topic.parquet (read and overwrite)",
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

    path = args.parquet
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")

    df = read_parquet_any(str(path))
    required = {"chunk_id", "conversation_id", "chunk_text", "topic"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

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

    write_parquet_any(df, str(path))
    print("Done. Updated (in place):", path)


if __name__ == "__main__":
    main()
