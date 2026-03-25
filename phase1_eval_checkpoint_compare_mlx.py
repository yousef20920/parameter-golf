from __future__ import annotations

import argparse
import glob
import importlib.util
import io
import math
import pickle
import time
import zlib
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import sentencepiece as spm
from mlx.utils import tree_unflatten

REPO_ROOT = Path(__file__).resolve().parent
TARGET = REPO_ROOT / "train_gpt_mlx.py"
DEFAULT_DATASET = REPO_ROOT / "data/datasets/fineweb10B_sp1024"
DEFAULT_TOKENIZER = REPO_ROOT / "data/tokenizers/fineweb_1024_bpe.model"
DEFAULT_CHECKPOINT = REPO_ROOT / "logs/mlx_full_m2_20260322_003835_mlx_model.int8.ptz"


def load_target_module():
    spec = importlib.util.spec_from_file_location("phase1_target_mlx", TARGET)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load target module from {TARGET}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare flat vs doc-aware sliding eval on an MLX checkpoint")
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Path to MLX checkpoint (.ptz or .npz)")
    p.add_argument("--checkpoint-kind", choices=("auto", "artifact", "raw"), default="auto")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Dataset directory containing fineweb_val_*.bin")
    p.add_argument("--tokenizer-path", default=str(DEFAULT_TOKENIZER), help="SentencePiece tokenizer model")
    p.add_argument("--num-docs", type=int, default=128, help="Evaluate only the first N docs; 0 means full validation")
    p.add_argument("--seq-len", type=int, default=1024, help="Evaluation sequence length")
    p.add_argument("--stride", type=int, default=64, help="Sliding window stride")
    p.add_argument("--batch-seqs", type=int, default=4, help="Fixed eval batch size in windows")
    p.add_argument("--include-next-bos", type=int, choices=(0, 1), default=1, help="Keep the next doc BOS in each doc span")
    return p.parse_args()


def load_checkpoint(path: Path, checkpoint_kind: str):
    if checkpoint_kind == "raw":
        return mx.load(str(path)), "raw"
    if checkpoint_kind == "artifact":
        return load_artifact(path), "artifact"
    if path.suffix == ".ptz":
        return load_artifact(path), "artifact"
    return mx.load(str(path)), "raw"


def load_artifact(path: Path):
    blob = path.read_bytes()
    raw = zlib.decompress(blob)
    return pickle.loads(raw)


def infer_model_config(flat_state: dict[str, object]) -> dict[str, int | float]:
    vocab_size, dim = map(int, flat_state["tok_emb.weight"].shape)
    block_ids = sorted(
        {
            int(name.split(".")[1])
            for name in flat_state
            if name.startswith("blocks.")
        }
    )
    num_layers = len(block_ids)
    num_heads = int(flat_state["blocks.0.attn.q_gain"].shape[0])
    head_dim = dim // num_heads
    num_kv_heads = int(flat_state["blocks.0.attn.c_k.weight"].shape[0]) // head_dim
    mlp_mult = int(flat_state["blocks.0.mlp.fc.weight"].shape[0]) // dim
    return {
        "vocab_size": vocab_size,
        "dim": dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "mlp_mult": mlp_mult,
    }


def build_model(target, cfg: dict[str, int | float]):
    return target.GPT(
        vocab_size=int(cfg["vocab_size"]),
        num_layers=int(cfg["num_layers"]),
        dim=int(cfg["dim"]),
        num_heads=int(cfg["num_heads"]),
        num_kv_heads=int(cfg["num_kv_heads"]),
        mlp_mult=int(cfg["mlp_mult"]),
        logit_chunk_tokens=0,
        logit_softcap=30.0,
        rope_base=10000.0,
        tied_embed_init_std=0.005,
        qk_gain_init=1.5,
    )


def load_model(target, checkpoint_path: Path, checkpoint_kind: str):
    payload, source = load_checkpoint(checkpoint_path, checkpoint_kind)
    if source == "artifact":
        flat_state = target.dequantize_state_dict_int8(payload)
    else:
        flat_state = payload
    cfg = infer_model_config(flat_state)
    model = build_model(target, cfg)
    model.update(tree_unflatten(list(flat_state.items())))
    return model, cfg, source


def find_document_spans(all_tokens: np.ndarray, bos_id: int, include_next_bos: bool) -> list[tuple[int, int]]:
    bos_positions = np.flatnonzero(all_tokens == bos_id)
    if bos_positions.size == 0:
        raise ValueError(f"Could not find BOS token id {bos_id} in validation tokens")
    if int(bos_positions[0]) != 0:
        raise ValueError(f"Expected validation stream to start with BOS at offset 0, got {int(bos_positions[0])}")
    docs: list[tuple[int, int]] = []
    for i, start in enumerate(bos_positions.tolist()):
        end = int(bos_positions[i + 1]) if i + 1 < bos_positions.size else int(all_tokens.size)
        if include_next_bos and i + 1 < bos_positions.size:
            end += 1
        length = end - start
        if length < 2:
            raise ValueError(f"Document starting at offset {start} is too short for evaluation")
        docs.append((start, length))
    return docs


def load_data_shard_cpu(path: Path) -> np.ndarray:
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    tokens = np.fromfile(path, dtype="<u2", offset=256 * np.dtype("<i4").itemsize)
    if tokens.size != num_tokens:
        raise ValueError(f"Expected {num_tokens} tokens in {path}, found {tokens.size}")
    return np.asarray(tokens, dtype=np.int32)


def load_val_prefix(dataset_dir: Path, bos_id: int, num_docs: int | None, include_next_bos: bool) -> tuple[np.ndarray, int]:
    val_files = sorted(glob.glob(str(dataset_dir / "fineweb_val_*.bin")))
    if not val_files:
        raise FileNotFoundError(f"No validation shards found under {dataset_dir}")
    all_tokens = np.ascontiguousarray(np.concatenate([load_data_shard_cpu(Path(p)) for p in val_files], axis=0))
    docs = find_document_spans(all_tokens, bos_id, include_next_bos=include_next_bos)
    if num_docs is None or num_docs <= 0:
        return all_tokens, len(docs)
    num_docs = min(num_docs, len(docs))
    last_start, last_len = docs[num_docs - 1]
    prefix = np.ascontiguousarray(all_tokens[: last_start + last_len])
    if prefix.size > 0 and int(prefix[-1]) == bos_id:
        prefix = np.ascontiguousarray(prefix[:-1])
    return prefix, num_docs


def token_losses(model, input_ids: mx.array, target_ids: mx.array) -> mx.array:
    hidden = model(input_ids)
    logits_proj = hidden @ model.tok_emb.weight.astype(hidden.dtype).T
    logits = model.softcap(logits_proj)
    return nn.losses.cross_entropy(logits.astype(mx.float32), target_ids, reduction="none")


def build_flat_windows(total_tokens: int, seq_len: int, stride: int) -> list[int]:
    return [
        ws for ws in range(0, total_tokens, stride)
        if min(ws + seq_len, total_tokens) - ws >= stride or ws == 0
    ]


def build_doc_windows(docs: list[tuple[int, int]], seq_len: int, stride: int) -> list[tuple[int, int, int, int, int]]:
    windows: list[tuple[int, int, int, int, int]] = []
    for doc_start, doc_len in docs:
        pred_len = doc_len - 1
        score_start = 0
        score_end = min(seq_len, pred_len)
        while score_start < pred_len:
            ws = max(0, score_end - seq_len)
            wlen = score_end - ws
            score_offset = score_start - ws
            score_len = score_end - score_start
            windows.append((doc_start, ws, wlen, score_offset, score_len))
            score_start = score_end
            score_end = min(score_end + stride, pred_len)
    return windows


def eval_flat_sliding(
    model,
    compiled_token_losses,
    val_tokens: np.ndarray,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
    seq_len: int,
    stride: int,
    batch_seqs: int,
) -> tuple[float, float]:
    total_tokens = val_tokens.size - 1
    windows = build_flat_windows(total_tokens, seq_len, stride)
    loss_sum = 0.0
    token_count = 0.0
    byte_count = 0.0
    for bi in range(0, len(windows), batch_seqs):
        batch_ws = windows[bi:bi + batch_seqs]
        x_batch = np.zeros((batch_seqs, seq_len), dtype=np.int32)
        y_batch = np.zeros((batch_seqs, seq_len), dtype=np.int32)
        scored: list[tuple[int, int]] = []
        for i, ws in enumerate(batch_ws):
            end = min(ws + seq_len, total_tokens)
            wlen = end - ws
            chunk = val_tokens[ws : end + 1]
            x_batch[i, :wlen] = chunk[:-1]
            y_batch[i, :wlen] = chunk[1:]
            scored.append((0 if ws == 0 else max(wlen - stride, 0), wlen))
        losses = compiled_token_losses(mx.array(x_batch, dtype=mx.int32), mx.array(y_batch, dtype=mx.int32))
        mx.eval(losses)
        losses_np = np.asarray(losses, dtype=np.float32)
        for i, (s, e) in enumerate(scored):
            loss_sum += float(losses_np[i, s:e].astype(np.float64).sum())
            token_count += float(e - s)
            prev = x_batch[i, s:e]
            tgt = y_batch[i, s:e]
            tb = base_bytes_lut[tgt].astype(np.float64, copy=True)
            tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).astype(np.float64, copy=False)
            byte_count += float(tb.sum())
        if (bi // batch_seqs) % 25 == 0:
            done = min(bi + batch_seqs, len(windows))
            running_bpb = (loss_sum / token_count) / math.log(2.0) * (token_count / byte_count)
            print(f"flat_eval_progress:{done}/{len(windows)} running_bpb:{running_bpb:.6f}")
    val_loss = loss_sum / token_count
    val_bpb = (val_loss / math.log(2.0)) * (token_count / byte_count)
    return val_loss, val_bpb


def eval_doc_aware_sliding(
    model,
    compiled_token_losses,
    val_tokens: np.ndarray,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
    bos_id: int,
    seq_len: int,
    stride: int,
    batch_seqs: int,
    include_next_bos: bool,
) -> tuple[float, float]:
    docs = find_document_spans(val_tokens, bos_id, include_next_bos=include_next_bos)
    windows = build_doc_windows(docs, seq_len, stride)
    loss_sum = 0.0
    token_count = 0.0
    byte_count = 0.0
    for bi in range(0, len(windows), batch_seqs):
        batch_windows = windows[bi:bi + batch_seqs]
        x_batch = np.zeros((batch_seqs, seq_len), dtype=np.int32)
        y_batch = np.zeros((batch_seqs, seq_len), dtype=np.int32)
        scored: list[tuple[int, int]] = []
        for i, (doc_start, ws, wlen, score_offset, score_len) in enumerate(batch_windows):
            end = ws + wlen
            chunk = val_tokens[doc_start + ws : doc_start + end + 1]
            x_batch[i, :wlen] = chunk[:-1]
            y_batch[i, :wlen] = chunk[1:]
            scored.append((score_offset, score_offset + score_len))
        losses = compiled_token_losses(mx.array(x_batch, dtype=mx.int32), mx.array(y_batch, dtype=mx.int32))
        mx.eval(losses)
        losses_np = np.asarray(losses, dtype=np.float32)
        for i, (s, e) in enumerate(scored):
            loss_sum += float(losses_np[i, s:e].astype(np.float64).sum())
            token_count += float(e - s)
            prev = x_batch[i, s:e]
            tgt = y_batch[i, s:e]
            tb = base_bytes_lut[tgt].astype(np.float64, copy=True)
            tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).astype(np.float64, copy=False)
            byte_count += float(tb.sum())
        if (bi // batch_seqs) % 25 == 0:
            done = min(bi + batch_seqs, len(windows))
            running_bpb = (loss_sum / token_count) / math.log(2.0) * (token_count / byte_count)
            print(f"doc_eval_progress:{done}/{len(windows)} running_bpb:{running_bpb:.6f}")
    expected_tokens = float((val_tokens.size - 1) if include_next_bos else (val_tokens.size - len(docs)))
    if abs(token_count - expected_tokens) > 0.5:
        raise RuntimeError(f"Doc-aware eval scored {token_count:.0f} tokens, expected {expected_tokens:.0f}")
    val_loss = loss_sum / token_count
    val_bpb = (val_loss / math.log(2.0)) * (token_count / byte_count)
    return val_loss, val_bpb


def main() -> None:
    args = parse_args()
    target = load_target_module()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    tokenizer_path = Path(args.tokenizer_path).expanduser().resolve()
    dataset_dir = Path(args.dataset).expanduser().resolve()
    include_next_bos = bool(args.include_next_bos)

    model, cfg, checkpoint_source = load_model(target, checkpoint_path, args.checkpoint_kind)
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    bos_id = int(sp.bos_id())
    if bos_id < 0:
        raise RuntimeError("Tokenizer does not define a BOS token")

    val_tokens, docs_loaded = load_val_prefix(dataset_dir, bos_id, args.num_docs if args.num_docs > 0 else None, include_next_bos)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = target.build_sentencepiece_luts(sp, int(cfg["vocab_size"]))
    compiled_token_losses = mx.compile(lambda x, y: token_losses(model, x, y), inputs=model.state, outputs=model.state)

    print(f"target_file: {TARGET}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"checkpoint_source: {checkpoint_source}")
    print(f"docs_loaded: {docs_loaded}")
    print(f"tokens_loaded: {val_tokens.size}")
    print(
        f"model_cfg:vocab={cfg['vocab_size']} layers={cfg['num_layers']} dim={cfg['dim']} "
        f"heads={cfg['num_heads']} kv_heads={cfg['num_kv_heads']} mlp_mult={cfg['mlp_mult']}"
    )
    print(f"seq_len:{args.seq_len} stride:{args.stride} batch_seqs:{args.batch_seqs} include_next_bos:{include_next_bos}")

    t0 = time.perf_counter()
    flat_loss, flat_bpb = eval_flat_sliding(
        model,
        compiled_token_losses,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        seq_len=args.seq_len,
        stride=args.stride,
        batch_seqs=args.batch_seqs,
    )
    flat_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    doc_loss, doc_bpb = eval_doc_aware_sliding(
        model,
        compiled_token_losses,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        bos_id=bos_id,
        seq_len=args.seq_len,
        stride=args.stride,
        batch_seqs=args.batch_seqs,
        include_next_bos=include_next_bos,
    )
    doc_time = time.perf_counter() - t1

    print(f"flat_sliding loss={flat_loss:.8f} bpb={flat_bpb:.8f} time_s={flat_time:.2f}")
    print(f"doc_aware    loss={doc_loss:.8f} bpb={doc_bpb:.8f} time_s={doc_time:.2f}")
    print(f"delta_bpb    {flat_bpb - doc_bpb:+.8f}")


if __name__ == "__main__":
    main()
