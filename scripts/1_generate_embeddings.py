# Author: Yolanda Pan
# Date: 12/11/2025
# About: Chunk CANDOR turn-level conversations into word-bounded chunks, 
# embed each chunk with MPNet, and save both turn-level (with chunk_id)
# and chunk-level (with embeddings) Parquet outputs.


import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from parquet_helper import read_parquet_any, write_parquet_any


# Change working directory to the script location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ============================================
# Embedding Model Wrapper (MPNet)
# ============================================

class MPNetEmbedder:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str | None = None,
        fp16: bool = False,
        normalize: bool = True,
    ):
        # Auto-select device unless explicitly defined
        self.device = device if device in {"cpu", "cuda"} else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.normalize = normalize

        # Load tokenizer + model
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.mdl = AutoModel.from_pretrained(model_name).to(self.device).eval()

        self.hidden = self.mdl.config.hidden_size

        # FP16 optimization (GPU only)
        self.fp16 = bool(fp16 and self.device == "cuda")
        if self.fp16:
            self.mdl.half()

        self.max_len = 512  # MPNet max sequence length

    def _l2(self, v: np.ndarray) -> np.ndarray:
        """L2 normalization for embeddings."""
        n = np.linalg.norm(v) + 1e-12
        return v / n

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling but only over unmasked tokens."""
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    @torch.inference_mode()
    def embed_text(
        self,
        text: str,
        stride_tokens: int = 64,
        normalize_out: bool | None = None,
        chunk_batch_size: int = 16,
    ) -> np.ndarray:
        """Embed a (possibly long) text with sliding window + weighted pooling."""
        if normalize_out is None:
            normalize_out = self.normalize

        # Empty text → zero vector
        if text is None or not str(text).strip():
            return np.zeros(self.hidden, dtype=np.float32)

        # Tokenize with sliding window
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

        # Collect per-window embeddings
        chunk_vecs, chunk_weights = [], []
        for i in range(0, input_ids.shape[0], chunk_batch_size):
            ids_b = input_ids[i:i + chunk_batch_size]
            attn_b = attn[i:i + chunk_batch_size]

            out = self.mdl(input_ids=ids_b, attention_mask=attn_b)
            pooled = self._mean_pool(out.last_hidden_state, attn_b)

            chunk_vecs.append(pooled.float().cpu().numpy())
            chunk_weights.append(attn_b.sum(dim=1).float().cpu().numpy())

        # Weighted average across sliding windows
        V = np.vstack(chunk_vecs).astype(np.float32)
        W = np.concatenate(chunk_weights).astype(np.float32)
        W = W / (W.sum() + 1e-9)

        vec = (V * W[:, None]).sum(axis=0).astype(np.float32)

        # Optional normalization
        if normalize_out:
            vec = self._l2(vec)

        return vec

    def embed_series(
        self,
        texts: pd.Series,
        stride_tokens: int = 64,
        normalize_out: bool | None = None,
        chunk_batch_size: int = 16,
        show_progress: bool = True,
    ) -> list[np.ndarray]:
        """Embed a pandas Series of texts."""
        texts = texts.fillna("").astype(str)
        out, n = [], len(texts)
        tick = max(1, n // 20)  # Print progress ~20 times

        for i, t in enumerate(texts):
            out.append(self.embed_text(
                t,
                stride_tokens=stride_tokens,
                normalize_out=normalize_out,
                chunk_batch_size=chunk_batch_size,
            ))
            if show_progress and ((i + 1) % tick == 0 or i + 1 == n):
                print(f"Embedded {i+1}/{n}")
        return out

    @staticmethod
    def stack(vectors) -> np.ndarray:
        """Safely stack a list/Series of 1D vectors into a matrix."""
        if isinstance(vectors, (pd.Series, pd.Index)):
            vectors = vectors.to_list()

        arrs = []
        for v in vectors:
            if v is None:
                continue
            a = np.asarray(v, dtype=np.float32)
            if a.ndim > 1:
                a = a.reshape(-1)
            if a.ndim == 1:
                arrs.append(a)

        if len(arrs) == 0:
            return np.zeros((0, 0), dtype=np.float32)

        # Validate consistent dimensionality
        dim = arrs[0].shape[0]
        for a in arrs:
            if a.shape[0] != dim:
                raise ValueError("Inconsistent embedding dimensions.")

        return np.vstack(arrs)


# ============================================
# Chunking Logic (word-based, per conversation)
# ============================================

def assign_chunks_by_words(
    df: pd.DataFrame,
    min_words: int = 40,
    max_words: int = 120,
) -> pd.DataFrame:
    """
    Assign a *global* chunk_id but respect conversation boundaries.

    - Sort by (conversation_id, turn_id)
    - Within each conversation, we accumulate word counts
    - A new chunk starts when:
        (a) we are at the first utterance of a conversation, OR
        (b) current words >= min_words AND adding the next utterance would exceed max_words
    - chunk_id is unique across the whole dataset (no sharing across conversations)
    """
    if "conversation_id" not in df.columns:
        raise ValueError("Expected a 'conversation_id' column in the input dataframe.")
    if "turn_id" not in df.columns:
        raise ValueError("Expected a 'turn_id' column in the input dataframe.")

    df = df.sort_values(["conversation_id", "turn_id"]).reset_index(drop=True).copy()

    # Compute n_words if missing
    if "n_words" not in df.columns:
        df["n_words"] = (
            df["utterance"].astype(str).str.split().str.len().fillna(0).astype(int)
        )

    chunk_ids = []
    cur_chunk = -1        # start at -1 so first real chunk becomes 0
    cur_words = 0
    prev_convo = None

    for _, row in df.iterrows():
        convo = row["conversation_id"]
        w = int(row["n_words"])

        # If we move to a new conversation, force a new chunk
        if convo != prev_convo:
            cur_chunk += 1          # new conversation → new chunk id
            cur_words = 0
            prev_convo = convo

        else:
            # Within the same conversation, decide if we start a new chunk
            if cur_words >= min_words and (cur_words + w > max_words):
                cur_chunk += 1
                cur_words = 0

        chunk_ids.append(cur_chunk)
        cur_words += w

    df["chunk_id"] = chunk_ids
    return df


def make_chunk_texts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a turn-level dataframe (with chunk_id) into a *unique* chunk-level dataframe:

        chunk_id, conversation_id, chunk_text

    - We assume each chunk_id belongs to exactly one conversation_id
    - conversation_id is taken as the first value inside each chunk_id group
    """
    # sanity check: each chunk_id should map to exactly one conversation_id
    mapping = df.groupby("chunk_id")["conversation_id"].nunique()
    bad_chunks = mapping[mapping > 1]
    if len(bad_chunks) > 0:
        # This should NOT happen now; if it does, it's a bug in chunking.
        raise ValueError(
            f"Some chunk_id values map to multiple conversation_id values: {bad_chunks.index.tolist()[:10]}"
        )

    chunks = (
        df.groupby("chunk_id")
          .agg(
              conversation_id=("conversation_id", "first"),
              chunk_text=("utterance", lambda s: " ".join(s.astype(str))),
          )
          .reset_index()
    )
    return chunks


# ============================================
# Main Script: Chunk → Embed
# ============================================

if __name__ == '__main__':
    BASE = "/project/macs40123/yolanda/candor_parquet"

    # 1. Load conversation (turn-level)
    full_convo = read_parquet_any(f"{BASE}/transcript_backbiter.ALL.parquet")

    # 2. Assign chunk_id based on word counts (per conversation)
    full_convo = assign_chunks_by_words(
        full_convo,
        min_words=40,
        max_words=120,
    )

    # 3. Aggregate each chunk into one long text + conversation_id
    chunk_df = make_chunk_texts(full_convo)

    # 4. Embed each chunk_text  (optional: you can comment this out
    #    if you don't want embeddings right now)
    embedder = MPNetEmbedder(normalize=True)
    chunk_df["chunk_vector"] = embedder.embed_series(
        chunk_df["chunk_text"],
        stride_tokens=64,
        chunk_batch_size=8,
    )

    # 5. Save turn-level with chunk_id
    write_parquet_any(
        full_convo,
        f"{BASE}/backbiter_turns_with_chunkid.parquet",
    )

    # 6. Save chunk-level with embeddings
    write_parquet_any(
        chunk_df,
        f"{BASE}/backbiter_chunk_embed.parquet",
    )
    

    print("Done. Saved:")
    print("  - backbiter_turns_with_chunkid.parquet")
    print("  - backbiter_chunk_embed.parquet      (with embeddings)")

 