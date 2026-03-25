from __future__ import annotations

import contextlib
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent
TARGET = REPO_ROOT / "records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py"


def load_target_module():
    # The smoke test only exercises eval helpers, so a stub is enough on machines
    # that do not have sentencepiece installed in the same Python as torch.
    if "sentencepiece" not in sys.modules:
        sys.modules["sentencepiece"] = types.ModuleType("sentencepiece")
    spec = importlib.util.spec_from_file_location("phase1_target", TARGET)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load target module from {TARGET}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.torch.autocast = lambda *args, **kwargs: contextlib.nullcontext()
    return module


class MajorityDocModel(nn.Module):
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
            if valid.numel() == 0:
                pred = self.bos_id
            else:
                counts = torch.bincount(valid, minlength=self.vocab_size)
                pred = int(counts.argmax())
            logits[b, :, pred] = 6.0
        return logits


def build_synthetic_val_tokens(bos_id: int = 1) -> torch.Tensor:
    docs = [
        [bos_id, 5, 5, 5, 5, 5, 5],
        [bos_id, 7, 7, 7, 7, 7, 7],
        [bos_id, 9, 9, 9, 9, 9, 9],
    ]
    flat = [tok for doc in docs for tok in doc]
    return torch.tensor(flat, dtype=torch.int64)


def main() -> None:
    mod = load_target_module()
    bos_id = 1
    vocab_size = 16
    val_tokens = build_synthetic_val_tokens(bos_id=bos_id)
    args = SimpleNamespace(train_seq_len=6)
    model = MajorityDocModel(vocab_size=vocab_size, bos_id=bos_id)
    base_bytes_lut = torch.ones(vocab_size, dtype=torch.float64)
    has_leading_space_lut = torch.zeros(vocab_size, dtype=torch.bool)
    is_boundary_token_lut = torch.zeros(vocab_size, dtype=torch.bool)
    is_boundary_token_lut[bos_id] = True

    docs_keep_next = mod.find_document_spans(val_tokens, bos_token_id=bos_id, include_next_bos=True)
    docs_strict = mod.find_document_spans(val_tokens, bos_token_id=bos_id, include_next_bos=False)

    flat_loss, flat_bpb = mod.eval_val_sliding(
        args=args,
        base_model=model,
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        val_tokens=val_tokens,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
        stride=2,
        batch_seqs=4,
    )
    doc_loss, doc_bpb = mod.eval_val_sliding_doc_aware(
        args=args,
        base_model=model,
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        val_tokens=val_tokens,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
        bos_token_id=bos_id,
        stride=2,
        batch_seqs=4,
        include_next_bos=True,
    )

    print("target_file:", TARGET)
    print("val_tokens:", val_tokens.tolist())
    print("docs_keep_next_bos:", docs_keep_next)
    print("docs_strict:", docs_strict)
    print(f"flat_sliding     loss={flat_loss:.6f} bpb={flat_bpb:.6f}")
    print(f"doc_aware        loss={doc_loss:.6f} bpb={doc_bpb:.6f}")
    print(f"delta_bpb        {flat_bpb - doc_bpb:+.6f}")
    if doc_bpb > flat_bpb:
        raise SystemExit("Expected doc-aware sliding to be no worse on this synthetic sample")


if __name__ == "__main__":
    main()
