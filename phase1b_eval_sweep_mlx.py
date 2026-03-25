from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlx.core as mx
import sentencepiece as spm

import phase1_eval_checkpoint_compare_mlx as compare


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer")
    return values


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Phase 1B stride/context sweeps on an MLX checkpoint")
    p.add_argument("--checkpoint", default=str(compare.DEFAULT_CHECKPOINT), help="Path to MLX checkpoint (.ptz or .npz)")
    p.add_argument("--checkpoint-kind", choices=("auto", "artifact", "raw"), default="auto")
    p.add_argument("--dataset", default=str(compare.DEFAULT_DATASET), help="Dataset directory containing fineweb_val_*.bin")
    p.add_argument("--tokenizer-path", default=str(compare.DEFAULT_TOKENIZER), help="SentencePiece tokenizer model")
    p.add_argument("--num-docs", type=int, default=512, help="Evaluate only the first N docs; 0 means full validation")
    p.add_argument("--seq-lens", type=parse_int_list, default=[1024], help="Comma-separated sequence lengths, e.g. 1024,1536")
    p.add_argument("--strides", type=parse_int_list, default=[32, 48, 64, 96], help="Comma-separated strides, e.g. 32,48,64,96")
    p.add_argument("--batch-seqs", type=int, default=2, help="Fixed eval batch size in windows")
    p.add_argument("--include-next-bos", type=int, choices=(0, 1), default=1, help="Keep the next doc BOS in each doc span")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    tokenizer_path = Path(args.tokenizer_path).expanduser().resolve()
    dataset_dir = Path(args.dataset).expanduser().resolve()
    include_next_bos = bool(args.include_next_bos)

    target = compare.load_target_module()
    model, cfg, checkpoint_source = compare.load_model(target, checkpoint_path, args.checkpoint_kind)
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

    combos = [(seq_len, stride) for seq_len in args.seq_lens for stride in args.strides if stride <= seq_len]
    if not combos:
        raise RuntimeError("No valid (seq_len, stride) combinations remain after filtering stride <= seq_len")

    print(f"target_file: {compare.TARGET}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"checkpoint_source: {checkpoint_source}")
    print(f"docs_loaded: {docs_loaded}")
    print(f"tokens_loaded: {val_tokens.size}")
    print(
        f"model_cfg:vocab={cfg['vocab_size']} layers={cfg['num_layers']} dim={cfg['dim']} "
        f"heads={cfg['num_heads']} kv_heads={cfg['num_kv_heads']} mlp_mult={cfg['mlp_mult']}"
    )
    print(f"include_next_bos:{include_next_bos} batch_seqs:{args.batch_seqs}")
    print("combos:" + ",".join(f"{seq_len}x{stride}" for seq_len, stride in combos))

    results: list[dict[str, float | int]] = []
    for seq_len, stride in combos:
        print(f"run_start seq_len:{seq_len} stride:{stride}")
        t0 = time.perf_counter()
        flat_loss, flat_bpb = compare.eval_flat_sliding(
            model,
            compiled_token_losses,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            seq_len=seq_len,
            stride=stride,
            batch_seqs=args.batch_seqs,
        )
        flat_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        doc_loss, doc_bpb = compare.eval_doc_aware_sliding(
            model,
            compiled_token_losses,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            bos_id=bos_id,
            seq_len=seq_len,
            stride=stride,
            batch_seqs=args.batch_seqs,
            include_next_bos=include_next_bos,
        )
        doc_time = time.perf_counter() - t1
        delta_bpb = flat_bpb - doc_bpb
        result = {
            "seq_len": seq_len,
            "stride": stride,
            "flat_bpb": flat_bpb,
            "doc_bpb": doc_bpb,
            "delta_bpb": delta_bpb,
            "flat_time_s": flat_time,
            "doc_time_s": doc_time,
            "total_time_s": flat_time + doc_time,
        }
        results.append(result)
        print(
            "run_done "
            f"seq_len:{seq_len} stride:{stride} "
            f"flat_bpb:{flat_bpb:.8f} doc_bpb:{doc_bpb:.8f} delta_bpb:{delta_bpb:+.8f} "
            f"flat_time_s:{flat_time:.2f} doc_time_s:{doc_time:.2f}"
        )

    print("")
    print("summary_table")
    print("seq_len,stride,flat_bpb,doc_bpb,delta_bpb,flat_time_s,doc_time_s,total_time_s")
    for row in results:
        print(
            f"{row['seq_len']},{row['stride']},"
            f"{row['flat_bpb']:.8f},{row['doc_bpb']:.8f},{row['delta_bpb']:+.8f},"
            f"{row['flat_time_s']:.2f},{row['doc_time_s']:.2f},{row['total_time_s']:.2f}"
        )

    best_doc = min(results, key=lambda row: row["doc_bpb"])
    best_delta = max(results, key=lambda row: row["delta_bpb"])
    print("")
    print(
        "best_doc "
        f"seq_len:{best_doc['seq_len']} stride:{best_doc['stride']} "
        f"doc_bpb:{best_doc['doc_bpb']:.8f} delta_bpb:{best_doc['delta_bpb']:+.8f}"
    )
    print(
        "best_delta "
        f"seq_len:{best_delta['seq_len']} stride:{best_delta['stride']} "
        f"doc_bpb:{best_delta['doc_bpb']:.8f} delta_bpb:{best_delta['delta_bpb']:+.8f}"
    )


if __name__ == "__main__":
    main()
