# 1_generate_embeddings.py (friends corpus)
# Author: Yolanda Pan
# Date: 2026/01/20
#
# About:
# Convert Friends season JSONs into chunk-level parquet with MPNet embeddings.
#
# Chunking:
# - bounded within each scene (scene_id = conversation)
# - word bounded: 40–120 words per chunk (approx)
#
# Embeddings:
# - embed PURE text only (no speaker names, no notes)

import os
import re
import json
import glob
from typing import Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

from parquet_helper import write_parquet_any


# ============================================================
# Config
# ============================================================

FRIENDS_JSON_DIR = "/home/xpan02/CASNL/topic_recurrence/data/friends_corpus"
OUT_DIR = "/project/ycleong/datasets/Friends"

MIN_WORDS = 40
MAX_WORDS = 120

# For embeddings: recommended False
INCLUDE_NOTES_IN_CHUNK_TEXT = False
INCLUDE_SPEAKERS_IN_CHUNK_TEXT = False   # recommended False for topic modeling

# Embedding
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBED_CHUNK_BATCH_SIZE = 8
EMBED_STRIDE_TOKENS = 64
EMBED_FP16 = True          # only applies on GPU
EMBED_NORMALIZE = True


# ============================================================
# Regex helpers
# ============================================================

U_RE = re.compile(r"_u(\d+)$")
S_RE = re.compile(r"friends_season_(\d+)\.json$")


# ============================================================
# Embedding Model Wrapper (MPNet)
# ============================================================

class MPNetEmbedder:
    def __init__(
        self,
        model_name: str = EMBED_MODEL_NAME,
        device: Optional[str] = None,
        fp16: bool = False,
        normalize: bool = True,
        max_len: int = 512,
    ):
        self.device = device if device in {"cpu", "cuda"} else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.normalize = normalize
        self.max_len = max_len

        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.mdl = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.hidden = self.mdl.config.hidden_size

        self.fp16 = bool(fp16 and self.device == "cuda")
        if self.fp16:
            self.mdl.half()

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    @staticmethod
    def _l2(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v) + 1e-12
        return v / n

    @torch.inference_mode()
    def embed_text(
        self,
        text: str,
        stride_tokens: int = 64,
        normalize_out: Optional[bool] = None,
        chunk_batch_size: int = 16,
    ) -> np.ndarray:
        if normalize_out is None:
            normalize_out = self.normalize

        if text is None or not str(text).strip():
            return np.zeros(self.hidden, dtype=np.float32)

        enc = self.tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_len,
            stride=stride_tokens,
            return_overflowing_tokens=True,
            padding="max_length",
            return_attention_mask=True,
        )
        input_ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)

        chunk_vecs, chunk_weights = [], []
        for i in range(0, input_ids.shape[0], chunk_batch_size):
            ids_b = input_ids[i:i + chunk_batch_size]
            attn_b = attn[i:i + chunk_batch_size]

            out = self.mdl(input_ids=ids_b, attention_mask=attn_b)
            pooled = self._mean_pool(out.last_hidden_state, attn_b)

            chunk_vecs.append(pooled.float().cpu().numpy())
            chunk_weights.append(attn_b.sum(dim=1).float().cpu().numpy())

        V = np.vstack(chunk_vecs).astype(np.float32)
        W = np.concatenate(chunk_weights).astype(np.float32)
        W = W / (W.sum() + 1e-9)

        vec = (V * W[:, None]).sum(axis=0).astype(np.float32)
        if normalize_out:
            vec = self._l2(vec)
        return vec

    def embed_series(
        self,
        texts: pd.Series,
        stride_tokens: int = 64,
        normalize_out: Optional[bool] = None,
        chunk_batch_size: int = 16,
        show_progress: bool = True,
    ) -> list[np.ndarray]:
        texts = texts.fillna("").astype(str)
        out, n = [], len(texts)
        tick = max(1, n // 20)

        for i, t in enumerate(texts):
            out.append(
                self.embed_text(
                    t,
                    stride_tokens=stride_tokens,
                    normalize_out=normalize_out,
                    chunk_batch_size=chunk_batch_size,
                )
            )
            if show_progress and ((i + 1) % tick == 0 or i + 1 == n):
                print(f"Embedded {i+1}/{n}")

        return out


# ============================================================
# Friends JSON -> utterance-level dataframe (minimal columns)
# ============================================================

def _safe_get_text(u: dict) -> str:
    for k in ["text", "transcript", "transcript_with_note"]:
        v = u.get(k, None)
        if v is not None and str(v).strip():
            return str(v)
    return ""

def _safe_get_speaker(u: dict) -> str:
    if "speaker" in u and u["speaker"] is not None:
        return str(u["speaker"])
    sp = u.get("speakers", None)
    if isinstance(sp, list) and len(sp) > 0:
        return str(sp[0])
    return "UNKNOWN"

def load_friends_json_season(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    season_id = data.get("season_id", None)
    if not season_id:
        m = S_RE.search(path)
        if m:
            season_id = f"s{int(m.group(1)):02d}"

    rows = []
    for ep in data.get("episodes", []):
        episode_id = ep.get("episode_id", None)

        for sc in ep.get("scenes", []):
            scene_id = sc.get("scene_id", None)
            if scene_id is None and season_id and episode_id:
                scene_id = f"{season_id}_{episode_id}_c??"

            for u in sc.get("utterances", []):
                utterance_id = u.get("utterance_id", u.get("id", None))
                speaker = _safe_get_speaker(u)
                text = _safe_get_text(u)

                turn_id = None
                if utterance_id:
                    mm = U_RE.search(str(utterance_id))
                    if mm:
                        turn_id = int(mm.group(1))

                is_note = (speaker == "TRANSCRIPT_NOTE") or (
                    speaker == "UNKNOWN" and text.strip().startswith("[")
                )

                rows.append({
                    "season_id": season_id,
                    "episode_id": episode_id,
                    "scene_id": scene_id,
                    "turn_id": turn_id,
                    "speaker": speaker,
                    "utterance": text,
                    "transcript_note": bool(is_note),
                })

    df = pd.DataFrame(rows)

    if len(df) > 0 and df["turn_id"].isna().any():
        df = df.sort_values(["season_id", "episode_id", "scene_id"]).reset_index(drop=True)
        df["turn_id"] = df.groupby("scene_id").cumcount() + 1

    if len(df) > 0:
        df["n_words"] = df["utterance"].fillna("").astype(str).str.split().str.len().astype(int)

    return df


# ============================================================
# Chunking (within each scene)
# ============================================================

def assign_chunks_by_words_scene(
    df: pd.DataFrame,
    min_words: int = 40,
    max_words: int = 120,
) -> pd.DataFrame:
    df = df.sort_values(["season_id", "episode_id", "scene_id", "turn_id"]).reset_index(drop=True).copy()

    chunk_ids = []
    cur_chunk = -1
    cur_words = 0
    prev_scene = None

    for _, row in df.iterrows():
        scene = row["scene_id"]
        w = int(row["n_words"])

        if scene != prev_scene:
            cur_chunk += 1
            cur_words = 0
            prev_scene = scene
        else:
            if cur_words >= min_words and (cur_words + w > max_words):
                cur_chunk += 1
                cur_words = 0

        chunk_ids.append(cur_chunk)
        cur_words += w

    df["chunk_id"] = chunk_ids
    return df


def make_chunk_df(df: pd.DataFrame) -> pd.DataFrame:
    # sanity: chunk_id should not cross scenes
    mapping = df.groupby("chunk_id")["scene_id"].nunique()
    bad = mapping[mapping > 1]
    if len(bad) > 0:
        raise ValueError(f"Some chunk_id map to multiple scenes: {bad.index.tolist()[:10]}")

    df = df.sort_values(["chunk_id", "turn_id"]).copy()

    def _join(g: pd.DataFrame) -> str:
        parts = []
        for spk, utt, is_note in zip(
            g["speaker"].tolist(),
            g["utterance"].fillna("").astype(str).tolist(),
            g["transcript_note"].tolist(),
        ):
            if not utt.strip():
                continue
            if is_note and not INCLUDE_NOTES_IN_CHUNK_TEXT:
                continue

            if INCLUDE_SPEAKERS_IN_CHUNK_TEXT and not is_note:
                parts.append(f"{spk}: {utt}")
            elif is_note:
                parts.append(utt)  # note raw text (rarely used)
            else:
                parts.append(utt)  # pure dialogue text

        return " ".join(parts)

    chunk_df = (
        df.groupby("chunk_id")
          .apply(lambda g: pd.Series({
              "season_id": g["season_id"].iloc[0],
              "episode_id": g["episode_id"].iloc[0],
              "scene_id": g["scene_id"].iloc[0],
              "start_turn_id": int(g["turn_id"].min()),
              "end_turn_id": int(g["turn_id"].max()),
              "n_utterances": int(len(g)),
              "n_words": int(g["n_words"].sum()),
              "chunk_text": _join(g),
          }))
          .reset_index()
    )

    # stable ID you can always trace back (even without turn-level parquet)
    chunk_df["chunk_uid"] = chunk_df.apply(
        lambda r: f"{r['scene_id']}_t{int(r['start_turn_id']):03d}-{int(r['end_turn_id']):03d}",
        axis=1
    )

    return chunk_df


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(FRIENDS_JSON_DIR, "friends_season_*.json")))
    if len(paths) == 0:
        raise FileNotFoundError(f"No friends_season_*.json found in: {FRIENDS_JSON_DIR}")

    all_df = []
    for p in paths:
        print(f"Loading {os.path.basename(p)}")
        all_df.append(load_friends_json_season(p))

    full = pd.concat(all_df, ignore_index=True)

    print(f"Loaded utterances: {len(full):,}")
    print(f"Unique scenes: {full['scene_id'].nunique():,}")

    full = assign_chunks_by_words_scene(full, min_words=MIN_WORDS, max_words=MAX_WORDS)
    chunk_df = make_chunk_df(full)

    print("Chunk word count summary:")
    print(chunk_df["n_words"].describe())

    embedder = MPNetEmbedder(
        model_name=EMBED_MODEL_NAME,
        fp16=EMBED_FP16,
        normalize=EMBED_NORMALIZE,
    )

    chunk_df["chunk_vector"] = embedder.embed_series(
        chunk_df["chunk_text"],
        stride_tokens=EMBED_STRIDE_TOKENS,
        chunk_batch_size=EMBED_CHUNK_BATCH_SIZE,
        show_progress=True,
    )

    chunk_out = os.path.join(OUT_DIR, "friends_chunk_embed.parquet")
    write_parquet_any(chunk_df, chunk_out)
    print(f"Saved chunk-level: {chunk_out}")
    print("Done.")


if __name__ == "__main__":
    main()
