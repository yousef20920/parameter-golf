from __future__ import annotations

import argparse
import contextlib
import glob
import importlib.util
import numpy as np
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent
TARGET = REPO_ROOT / "records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py"
DEFAULT_DATASET = REPO_ROOT / "data/datasets/fineweb10B_sp1024"


def load_target_module():
    if "sentencepiece" not in sys.modules:
        sys.modules["sentencepiece"] = types.ModuleType("sentencepiece")
    spec = importlib.util.spec_from_file_location("phase1_target_real", TARGET)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load target module from {TARGET}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.torch.autocast = lambda *args, **kwargs: contextlib.nullcontext()
    return module


class WindowMajorityModel(nn.Module):
    def __init__(self, vocab_size: int, bos_id: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.bos_id = bos_id

    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = input_ids.shape
        logits = torch.full((bsz, seqlen, self.vocab_size), -6.0, dtype=torch.float32, device=input_ids.device)
        for b in range(bsz):
            row = input_ids[b]
            valid = row[(row != 0) & (row != self.bos_id)]
            pred = self.bos_id
            if valid.numel() > 0:
                counts = torch.bincount(valid, minlength=self.vocab_size)
                pred = int(counts.argmax())
            logits[b, :, pred] = 6.0
        return logits


def load_val_prefix(module, dataset_dir: Path, bos_id: int, num_docs: int) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    val_files = sorted(glob.glob(str(dataset_dir / "fineweb_val_*.bin")))
    if not val_files:
        raise FileNotFoundError(f"No validation shards found under {dataset_dir}")
    token_tensors = [load_data_shard_cpu(Path(p)) for p in val_files]
    all_tokens = torch.cat(token_tensors).to(torch.int64)
    docs = module.find_document_spans(all_tokens, bos_id, include_next_bos=True)
    if num_docs <= 0 or num_docs > len(docs):
        num_docs = len(docs)
    last_start, last_len = docs[num_docs - 1]
    prefix = all_tokens[: last_start + last_len].contiguous()
    if prefix.numel() > 0 and int(prefix[-1].item()) == bos_id:
        prefix = prefix[:-1].contiguous()
    prefix_docs = module.find_document_spans(prefix, bos_id, include_next_bos=True)
    return prefix, prefix_docs


def load_data_shard_cpu(path: Path) -> torch.Tensor:
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    tokens = np.fromfile(path, dtype="<u2", offset=256 * np.dtype("<i4").itemsize)
    if tokens.size != num_tokens:
        raise ValueError(f"Expected {num_tokens} tokens in {path}, found {tokens.size}")
    return torch.from_numpy(tokens.astype(np.int64, copy=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU smoke test for doc-aware eval on real validation tokens")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Dataset directory containing fineweb_val_*.bin")
    parser.add_argument("--num-docs", type=int, default=64, help="Number of validation docs to include")
    parser.add_argument("--seq-len", type=int, default=256, help="Evaluation window length")
    parser.add_argument("--stride", type=int, default=64, help="Sliding stride")
    parser.add_argument("--batch-seqs", type=int, default=8, help="Batch size in windows for CPU smoke test")
    parser.add_argument("--bos-id", type=int, default=1, help="BOS token id used in the cached dataset")
    parser.add_argument("--vocab-size", type=int, default=1024, help="Vocabulary size for the dummy logits")
    args = parser.parse_args()

    module = load_target_module()
    dataset_dir = Path(args.dataset).expanduser().resolve()
    val_tokens, docs = load_val_prefix(module, dataset_dir, args.bos_id, args.num_docs)

    eval_args = SimpleNamespace(train_seq_len=args.seq_len)
    model = WindowMajorityModel(vocab_size=args.vocab_size, bos_id=args.bos_id)
    base_bytes_lut = torch.ones(args.vocab_size, dtype=torch.float64)
    has_leading_space_lut = torch.zeros(args.vocab_size, dtype=torch.bool)
    is_boundary_token_lut = torch.zeros(args.vocab_size, dtype=torch.bool)
    is_boundary_token_lut[args.bos_id] = True

    flat_loss, flat_bpb = module.eval_val_sliding(
        args=eval_args,
        base_model=model,
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        val_tokens=val_tokens,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
        stride=args.stride,
        batch_seqs=args.batch_seqs,
    )
    doc_loss, doc_bpb = module.eval_val_sliding_doc_aware(
        args=eval_args,
        base_model=model,
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        val_tokens=val_tokens,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
        bos_token_id=args.bos_id,
        stride=args.stride,
        batch_seqs=args.batch_seqs,
        include_next_bos=True,
    )

    print("target_file:", TARGET)
    print("dataset_dir:", dataset_dir)
    print("docs_in_prefix:", len(docs))
    print("tokens_in_prefix:", val_tokens.numel())
    print("first_doc_spans:", docs[:5])
    print(f"flat_sliding loss={flat_loss:.6f} bpb={flat_bpb:.6f}")
    print(f"doc_aware    loss={doc_loss:.6f} bpb={doc_bpb:.6f}")
    print(f"delta_bpb    {flat_bpb - doc_bpb:+.6f}")


if __name__ == "__main__":
    main()
