from __future__ import annotations

import argparse
import pickle
import time
import zlib
from pathlib import Path

import mlx.core as mx
import sentencepiece as spm
from mlx.utils import tree_unflatten

import phase1_eval_checkpoint_compare_mlx as compare

DEFAULT_RAW_CHECKPOINT = compare.REPO_ROOT / "logs/mlx_full_m2_20260322_003835_mlx_model.npz"

PRESET_OVERRIDES: dict[str, dict[str, float]] = {
    "baseline": {},
    "mlp_9999": {"mlp": 99.99},
    "mlp_999": {"mlp": 99.9},
    "attn_safe_mlp_9999": {"attn": 99.9999, "mlp": 99.99},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep MLX int8 quantization clip presets on a raw checkpoint")
    p.add_argument("--checkpoint", default=str(DEFAULT_RAW_CHECKPOINT), help="Path to raw MLX checkpoint (.npz preferred)")
    p.add_argument("--checkpoint-kind", choices=("auto", "artifact", "raw"), default="raw")
    p.add_argument("--dataset", default=str(compare.DEFAULT_DATASET), help="Dataset directory containing fineweb_val_*.bin")
    p.add_argument("--tokenizer-path", default=str(compare.DEFAULT_TOKENIZER), help="SentencePiece tokenizer model")
    p.add_argument("--num-docs", type=int, default=128, help="Evaluate only the first N docs; 0 means full validation")
    p.add_argument("--seq-len", type=int, default=1024, help="Evaluation sequence length")
    p.add_argument("--stride", type=int, default=96, help="Sliding window stride")
    p.add_argument("--batch-seqs", type=int, default=2, help="Fixed eval batch size in windows")
    p.add_argument("--include-next-bos", type=int, choices=(0, 1), default=1, help="Keep the next doc BOS in each doc span")
    p.add_argument("--eval-mode", choices=("doc", "flat", "both"), default="doc")
    p.add_argument("--presets", default="baseline,mlp_9999,mlp_999,attn_safe_mlp_9999", help="Comma-separated preset names")
    return p.parse_args()


def selected_presets(raw: str) -> list[tuple[str, dict[str, float]]]:
    names = [part.strip() for part in raw.split(",") if part.strip()]
    if not names:
        raise ValueError("Expected at least one preset")
    missing = [name for name in names if name not in PRESET_OVERRIDES]
    if missing:
        raise ValueError(f"Unknown preset(s): {', '.join(missing)}")
    return [(name, PRESET_OVERRIDES[name]) for name in names]


def merged_clip_percentiles(target, overrides: dict[str, float]) -> dict[str, float]:
    merged = dict(target.INT8_DEFAULT_CLIP_PERCENTILES)
    merged.update(overrides)
    return merged


def evaluate_model(
    args: argparse.Namespace,
    model,
    compiled_token_losses,
    target,
    val_tokens,
    base_bytes_lut,
    has_leading_space_lut,
    is_boundary_token_lut,
    bos_id: int,
) -> tuple[float | None, float | None]:
    flat_bpb = None
    doc_bpb = None
    if args.eval_mode in ("flat", "both"):
        _, flat_bpb = compare.eval_flat_sliding(
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
    if args.eval_mode in ("doc", "both"):
        _, doc_bpb = compare.eval_doc_aware_sliding(
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
            include_next_bos=bool(args.include_next_bos),
        )
    return flat_bpb, doc_bpb


def fmt_metric(value: float | None) -> str:
    return "" if value is None else f"{value:.8f}"


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    tokenizer_path = Path(args.tokenizer_path).expanduser().resolve()
    dataset_dir = Path(args.dataset).expanduser().resolve()
    include_next_bos = bool(args.include_next_bos)

    target = compare.load_target_module()
    payload, checkpoint_source = compare.load_checkpoint(checkpoint_path, args.checkpoint_kind)
    if checkpoint_source == "artifact":
        flat_state = target.dequantize_state_dict_int8(payload)
    else:
        flat_state = payload
    cfg = compare.infer_model_config(flat_state)
    model = compare.build_model(target, cfg)
    model.update(tree_unflatten(list(flat_state.items())))

    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    bos_id = int(sp.bos_id())
    if bos_id < 0:
        raise RuntimeError("Tokenizer does not define a BOS token")
    val_tokens, docs_loaded = compare.load_val_prefix(
        dataset_dir,
        bos_id,
        args.num_docs if args.num_docs > 0 else None,
        include_next_bos=include_next_bos,
    )
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = target.build_sentencepiece_luts(
        sp,
        int(cfg["vocab_size"]),
    )
    compiled_token_losses = mx.compile(lambda x, y: compare.token_losses(model, x, y), inputs=model.state, outputs=model.state)

    print(f"target_file: {compare.TARGET}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"checkpoint_source: {checkpoint_source}")
    print(f"docs_loaded: {docs_loaded}")
    print(f"tokens_loaded: {val_tokens.size}")
    print(
        f"model_cfg:vocab={cfg['vocab_size']} layers={cfg['num_layers']} dim={cfg['dim']} "
        f"heads={cfg['num_heads']} kv_heads={cfg['num_kv_heads']} mlp_mult={cfg['mlp_mult']}"
    )
    print(
        f"eval_mode:{args.eval_mode} seq_len:{args.seq_len} stride:{args.stride} "
        f"batch_seqs:{args.batch_seqs} include_next_bos:{include_next_bos}"
    )

    print("raw_eval_start")
    t0 = time.perf_counter()
    raw_flat_bpb, raw_doc_bpb = evaluate_model(
        args,
        model,
        compiled_token_losses,
        target,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        bos_id,
    )
    raw_eval_s = time.perf_counter() - t0
    print(
        f"raw_eval_done flat_bpb:{fmt_metric(raw_flat_bpb)} doc_bpb:{fmt_metric(raw_doc_bpb)} "
        f"time_s:{raw_eval_s:.2f}"
    )

    results: list[dict[str, str | float | int]] = []
    for preset_name, overrides in selected_presets(args.presets):
        clip_percentiles = merged_clip_percentiles(target, overrides)
        print(f"preset_start name:{preset_name} clip_percentiles:{clip_percentiles}")
        quant_obj, quant_stats = target.quantize_state_dict_int8(flat_state, clip_percentiles=clip_percentiles)
        quant_raw = pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
        quant_blob = zlib.compress(quant_raw, level=9)
        artifact_bytes = len(quant_blob)
        quant_flat = target.dequantize_state_dict_int8(quant_obj)
        model.update(tree_unflatten(list(quant_flat.items())))
        t1 = time.perf_counter()
        flat_bpb, doc_bpb = evaluate_model(
            args,
            model,
            compiled_token_losses,
            target,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            bos_id,
        )
        eval_s = time.perf_counter() - t1
        result = {
            "preset": preset_name,
            "artifact_bytes": artifact_bytes,
            "payload_bytes": int(quant_stats["int8_payload_bytes"]),
            "flat_bpb": flat_bpb if flat_bpb is not None else "",
            "doc_bpb": doc_bpb if doc_bpb is not None else "",
            "delta_flat_vs_raw": (flat_bpb - raw_flat_bpb) if flat_bpb is not None and raw_flat_bpb is not None else "",
            "delta_doc_vs_raw": (doc_bpb - raw_doc_bpb) if doc_bpb is not None and raw_doc_bpb is not None else "",
            "eval_time_s": eval_s,
        }
        results.append(result)
        print(
            f"preset_done name:{preset_name} artifact_bytes:{artifact_bytes} payload_bytes:{quant_stats['int8_payload_bytes']} "
            f"flat_bpb:{fmt_metric(flat_bpb)} doc_bpb:{fmt_metric(doc_bpb)} "
            f"delta_flat_vs_raw:{fmt_metric(result['delta_flat_vs_raw'] if isinstance(result['delta_flat_vs_raw'], float) else None)} "
            f"delta_doc_vs_raw:{fmt_metric(result['delta_doc_vs_raw'] if isinstance(result['delta_doc_vs_raw'], float) else None)} "
            f"eval_time_s:{eval_s:.2f}"
        )

    print("")
    print("summary_table")
    print("preset,artifact_bytes,payload_bytes,flat_bpb,doc_bpb,delta_flat_vs_raw,delta_doc_vs_raw,eval_time_s")
    for row in results:
        print(
            f"{row['preset']},{row['artifact_bytes']},{row['payload_bytes']},"
            f"{fmt_metric(row['flat_bpb'] if isinstance(row['flat_bpb'], float) else None)},"
            f"{fmt_metric(row['doc_bpb'] if isinstance(row['doc_bpb'], float) else None)},"
            f"{fmt_metric(row['delta_flat_vs_raw'] if isinstance(row['delta_flat_vs_raw'], float) else None)},"
            f"{fmt_metric(row['delta_doc_vs_raw'] if isinstance(row['delta_doc_vs_raw'], float) else None)},"
            f"{row['eval_time_s']:.2f}"
        )

    metric_key = "doc_bpb" if args.eval_mode in ("doc", "both") else "flat_bpb"
    best = min(results, key=lambda row: float(row[metric_key]))
    print("")
    print(
        f"best_preset name:{best['preset']} artifact_bytes:{best['artifact_bytes']} "
        f"{metric_key}:{float(best[metric_key]):.8f}"
    )


if __name__ == "__main__":
    main()
