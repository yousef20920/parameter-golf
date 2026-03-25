from __future__ import annotations

import argparse
import contextlib
import glob
import importlib.util
import io
import sys
import time
import types
import zlib
from pathlib import Path

import numpy as np
import torch

try:
    import zstandard
except ImportError:
    zstandard = None

REPO_ROOT = Path(__file__).resolve().parent
TARGET = REPO_ROOT / "records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py"
DEFAULT_DATASET = REPO_ROOT / "data/datasets/fineweb10B_sp1024"
DEFAULT_TOKENIZER = REPO_ROOT / "data/tokenizers/fineweb_1024_bpe.model"


def load_target_module():
    if "sentencepiece" not in sys.modules:
        sys.modules["sentencepiece"] = types.ModuleType("sentencepiece")
    spec = importlib.util.spec_from_file_location("phase1_target_eval", TARGET)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load target module from {TARGET}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_data_shard_cpu(path: Path) -> torch.Tensor:
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    tokens = np.fromfile(path, dtype="<u2", offset=256 * np.dtype("<i4").itemsize)
    if tokens.size != num_tokens:
        raise ValueError(f"Expected {num_tokens} tokens in {path}, found {tokens.size}")
    return torch.from_numpy(tokens.astype(np.int64, copy=False))


def load_val_prefix(target, dataset_dir: Path, bos_id: int, num_docs: int | None) -> tuple[torch.Tensor, int]:
    val_files = sorted(glob.glob(str(dataset_dir / "fineweb_val_*.bin")))
    if not val_files:
        raise FileNotFoundError(f"No validation shards found under {dataset_dir}")
    all_tokens = torch.cat([load_data_shard_cpu(Path(p)) for p in val_files]).contiguous()
    if num_docs is None or num_docs <= 0:
        return all_tokens, len(target.find_document_spans(all_tokens, bos_id, include_next_bos=True))
    docs = target.find_document_spans(all_tokens, bos_id, include_next_bos=True)
    num_docs = min(num_docs, len(docs))
    last_start, last_len = docs[num_docs - 1]
    prefix = all_tokens[: last_start + last_len].contiguous()
    if prefix.numel() > 0 and int(prefix[-1].item()) == bos_id:
        prefix = prefix[:-1].contiguous()
    return prefix, num_docs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare flat sliding eval vs doc-aware sliding eval on a SOTA checkpoint")
    p.add_argument("--checkpoint", required=True, help="Path to a checkpoint from the SOTA record script (.pt or .ptz)")
    p.add_argument("--checkpoint-kind", choices=("auto", "state_dict", "artifact"), default="auto")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Dataset directory containing fineweb_val_*.bin")
    p.add_argument("--tokenizer-path", default=str(DEFAULT_TOKENIZER), help="SentencePiece tokenizer model")
    p.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    p.add_argument("--num-docs", type=int, default=0, help="Evaluate only the first N docs; 0 means full validation")
    p.add_argument("--seq-len", type=int, default=2048, help="Evaluation sequence length")
    p.add_argument("--stride", type=int, default=64, help="Sliding window stride")
    p.add_argument("--batch-seqs", type=int, default=8, help="Number of windows per eval batch")
    p.add_argument("--include-next-bos", type=int, choices=(0, 1), default=1, help="Keep the next doc BOS in each doc span")
    return p.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but torch.cuda.is_available() is false")
    return dev


def build_model(target, device: torch.device):
    model = target.GPT(
        vocab_size=1024,
        num_layers=10,
        model_dim=512,
        num_heads=8,
        num_kv_heads=4,
        mlp_mult=3.0,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
        bigram_vocab_size=10240,
        bigram_dim=128,
    )
    if device.type == "cuda":
        model = model.to(device).bfloat16()
        for module in model.modules():
            if isinstance(module, target.CastedLinear):
                module.float()
        target.restore_low_dim_params_to_fp32(model)
    else:
        model = model.to(device)
    return model


def load_checkpoint_payload(path: Path, checkpoint_kind: str):
    if checkpoint_kind == "state_dict":
        return torch.load(path, map_location="cpu")
    if checkpoint_kind == "artifact":
        return load_quantized_payload(path)
    if path.suffix == ".ptz":
        return load_quantized_payload(path)
    return torch.load(path, map_location="cpu")


def load_quantized_payload(path: Path):
    blob = path.read_bytes()
    if blob[:4] == b"\x28\xb5\x2f\xfd":
        if zstandard is None:
            raise RuntimeError("Checkpoint looks like zstd-compressed .ptz but zstandard is not installed")
        raw = zstandard.ZstdDecompressor().decompress(blob)
    else:
        raw = zlib.decompress(blob)
    return torch.load(io.BytesIO(raw), map_location="cpu")


def unwrap_state_dict(payload):
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    return payload


def load_model_weights(target, model, checkpoint_path: Path, checkpoint_kind: str) -> str:
    payload = load_checkpoint_payload(checkpoint_path, checkpoint_kind)
    template_sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if isinstance(payload, dict) and "w" in payload and "m" in payload:
        state_dict = target.dequantize_mixed_int6(payload["w"], payload["m"], template_sd)
        source = "artifact"
    else:
        state_dict = unwrap_state_dict(payload)
        source = "state_dict"
    model.load_state_dict(state_dict, strict=True)
    return source


def maybe_patch_autocast(target, device: torch.device) -> None:
    if device.type != "cuda":
        target.torch.autocast = lambda *args, **kwargs: contextlib.nullcontext()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    target = load_target_module()
    device = choose_device(args.device)
    maybe_patch_autocast(target, device)

    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError("sentencepiece is required for tokenizer-aware checkpoint evaluation") from exc

    sp = spm.SentencePieceProcessor(model_file=str(Path(args.tokenizer_path).expanduser().resolve()))
    bos_id = int(sp.bos_id())
    if bos_id < 0:
        raise RuntimeError("Tokenizer does not define a BOS token")

    dataset_dir = Path(args.dataset).expanduser().resolve()
    num_docs = args.num_docs if args.num_docs > 0 else None
    val_tokens, docs_loaded = load_val_prefix(target, dataset_dir, bos_id, num_docs)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = target.build_sentencepiece_luts(sp, 1024, device)

    eval_args = argparse.Namespace(train_seq_len=args.seq_len)
    model = build_model(target, device)
    checkpoint_source = load_model_weights(target, model, checkpoint_path, args.checkpoint_kind)
    model.eval()

    print(f"target_file: {TARGET}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"checkpoint_source: {checkpoint_source}")
    print(f"device: {device}")
    print(f"dataset_dir: {dataset_dir}")
    print(f"docs_loaded: {docs_loaded}")
    print(f"tokens_loaded: {val_tokens.numel()}")
    print(f"seq_len: {args.seq_len} stride: {args.stride} batch_seqs: {args.batch_seqs}")

    t0 = time.perf_counter()
    flat_loss, flat_bpb = target.eval_val_sliding(
        args=eval_args,
        base_model=model,
        rank=0,
        world_size=1,
        device=device,
        val_tokens=val_tokens,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
        stride=args.stride,
        batch_seqs=args.batch_seqs,
    )
    flat_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    doc_loss, doc_bpb = target.eval_val_sliding_doc_aware(
        args=eval_args,
        base_model=model,
        rank=0,
        world_size=1,
        device=device,
        val_tokens=val_tokens,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
        bos_token_id=bos_id,
        stride=args.stride,
        batch_seqs=args.batch_seqs,
        include_next_bos=bool(args.include_next_bos),
    )
    doc_time = time.perf_counter() - t1

    print(f"flat_sliding loss={flat_loss:.8f} bpb={flat_bpb:.8f} time_s={flat_time:.2f}")
    print(f"doc_aware    loss={doc_loss:.8f} bpb={doc_bpb:.8f} time_s={doc_time:.2f}")
    print(f"delta_bpb    {flat_bpb - doc_bpb:+.8f}")


if __name__ == "__main__":
    main()
