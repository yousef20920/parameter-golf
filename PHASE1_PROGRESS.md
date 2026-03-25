# Phase 1 Progress

## Goal

Phase 1 is testing whether document-aware validation improves measured `val_bpb` compared with the repo's existing flat sliding-window evaluation.

The core idea is:

- `flat sliding`: score one long concatenated validation token stream
- `doc-aware sliding`: split the stream at BOS tokens and score each document separately

This matters because flat sliding lets the model see context from the previous document, which is not a clean approximation of true per-document evaluation.

## What Changed

### 1. Added document-aware eval to the current SOTA PyTorch record script

File:

- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py`

Changes:

- added `EVAL_DOC_AWARE`
- added `EVAL_INCLUDE_NEXT_BOS`
- added BOS-based document span parsing
- added a document-aware sliding evaluator
- wired final eval to use the doc-aware path when enabled

Important bug fix:

- the first version over-scored tail tokens
- that was fixed by making the doc-aware windows partition each document's scored targets exactly once

Verification:

- `python3 -m py_compile` passed
- `git diff --check` passed

### 2. Added local smoke tests

Files:

- `phase1_docaware_smoke_test.py`
- `phase1_real_val_smoke_test.py`

Purpose:

- validate the evaluator logic on CPU
- validate on both synthetic data and a small real validation prefix

### 3. Added eval-only compare runners

Files:

- `phase1_eval_checkpoint_compare.py`
- `phase1_eval_checkpoint_compare_mlx.py`

Purpose:

- compare `flat sliding` vs `doc-aware sliding` without retraining
- use a real saved checkpoint or artifact
- support local testing on the Mac via the MLX artifact

Important MLX-script fix:

- `phase1_eval_checkpoint_compare_mlx.py` initially assumed doc-aware eval always scored `tokens - 1`
- that is only true when `--include-next-bos 1`
- for `--include-next-bos 0`, the correct expected token count is `tokens - docs`
- this was fixed on 2026-03-23

## Results So Far

### A. Synthetic smoke test

Script:

- `python3 phase1_docaware_smoke_test.py`

Result:

- `flat_sliding     loss=3.500092 bpb=5.049566`
- `doc_aware        loss=1.200092 bpb=1.731367`
- `delta_bpb        +3.318199`

Interpretation:

- this is only a logic sanity check
- it confirms the evaluator behaves correctly when cross-document context should clearly hurt

### B. Real validation-prefix smoke test with a dummy model

Script:

- `python3 phase1_real_val_smoke_test.py --num-docs N --seq-len 128 --stride 32 --batch-seqs 4`

Results:

| Docs | Flat BPB | Doc-Aware BPB | Delta |
| --- | ---: | ---: | ---: |
| 16 | 16.621780 | 16.598610 | +0.023170 |
| 64 | 16.605664 | 16.579942 | +0.025722 |
| 128 | 16.577294 | 16.559834 | +0.017460 |

Interpretation:

- the doc-aware path works on real `fineweb_val_*` data
- the direction is consistently favorable
- these are not leaderboard numbers because the model is a dummy model

### C. Real checkpoint compare on the Mac using the MLX artifact

Checkpoint used:

- `logs/mlx_full_m2_20260322_003835_mlx_model.int8.ptz`

Inferred model config:

- `layers=9`
- `dim=512`
- `heads=8`
- `kv_heads=4`
- `mlp_mult=2`

#### 128-doc compare

Command:

```bash
.venv/bin/python phase1_eval_checkpoint_compare_mlx.py \
  --checkpoint logs/mlx_full_m2_20260322_003835_mlx_model.int8.ptz \
  --num-docs 128 \
  --seq-len 1024 \
  --stride 64 \
  --batch-seqs 2
```

Result:

- `flat_sliding loss=3.86925848 bpb=2.31786331 time_s=70.96`
- `doc_aware    loss=3.86670273 bpb=2.31364133 time_s=35.94`
- `delta_bpb    +0.00422198`

#### 512-doc compare

Command:

```bash
.venv/bin/python phase1_eval_checkpoint_compare_mlx.py \
  --checkpoint logs/mlx_full_m2_20260322_003835_mlx_model.int8.ptz \
  --num-docs 512 \
  --seq-len 1024 \
  --stride 64 \
  --batch-seqs 2
```

Result:

- `flat_sliding loss=3.85341165 bpb=2.31567246 time_s=269.10`
- `doc_aware    loss=3.85208930 bpb=2.31474688 time_s=145.43`
- `delta_bpb    +0.00092558`

Interpretation:

- document-aware evaluation still helps on a real trained checkpoint
- the effect shrank significantly as the sample size increased
- Phase 1A looks real, but probably not large enough by itself to beat the best submission

## Full Validation Estimate on the Mac

Based on the 128-doc MLX checkpoint timing and the full validation size of about `62,021,846` tokens:

- flat full estimate: about `29,116s` (`8.1h`)
- doc-aware full estimate: about `14,747s` (`4.1h`)
- total flat+doc compare: about `43,863s` (`12.2h`)

Conclusion:

- full validation on the Mac is possible
- it is expensive enough that it should only be done if the expected gain looks meaningful

## What We Learned

1. The Phase 1 doc-aware evaluator works.
2. The implementation bug in the first version is fixed.
3. The effect is directionally positive on:
   - synthetic data
   - real validation prefixes
   - a real MLX checkpoint
4. The gain appears to shrink as the evaluation sample gets larger.
5. On this checkpoint, smaller stride did not improve the best doc-aware score.
6. On this checkpoint, longer eval context hurt absolute doc-aware performance.
7. As of current evidence, doc-aware eval is worth keeping, but it does not yet look like a standalone winning move.

## Phase 1B Results

A 512-doc MLX compare was run with:

```bash
.venv/bin/python phase1_eval_checkpoint_compare_mlx.py \
  --checkpoint logs/mlx_full_m2_20260322_003835_mlx_model.int8.ptz \
  --num-docs 512 \
  --seq-len 1024 \
  --stride 64 \
  --batch-seqs 2 \
  --include-next-bos 0
```

Result:

- `flat_sliding loss=3.85341165 bpb=2.31567246 time_s=286.87`
- `doc_aware    loss=3.85066380 bpb=2.31182887 time_s=154.39`
- `delta_bpb    +0.00384359`

Important interpretation note:

- `--include-next-bos 1` keeps the BOS token of the next document as a scored target
- `--include-next-bos 0` drops those BOS targets
- this means `0` can look slightly better partly because it is scoring an easier subset of targets
- for apples-to-apples comparison with the original token stream, `1` is the cleaner setting

Direct comparison on the same 512-doc prefix:

| Setting | Doc-Aware BPB | Gain vs Flat |
| --- | ---: | ---: |
| `--include-next-bos 1` | 2.31474688 | +0.00092558 |
| `--include-next-bos 0` | 2.31182887 | +0.00384359 |

Interpretation:

- `0` looks better numerically
- but `1` is the more defensible evaluation because it preserves all boundary targets instead of dropping them
- if the goal is a clean replacement for flat evaluation, `1` is still the safer default

### Stride sweep on the 512-doc MLX prefix

Command:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python phase1b_eval_sweep_mlx.py \
  --checkpoint logs/mlx_full_m2_20260322_003835_mlx_model.int8.ptz \
  --num-docs 512 \
  --seq-lens 1024 \
  --strides 32,48,64,96 \
  --batch-seqs 2 \
  --include-next-bos 1
```

Results:

| Seq Len | Stride | Flat BPB | Doc-Aware BPB | Delta |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 32 | 2.31603628 | 2.31488652 | +0.00114976 |
| 1024 | 48 | 2.31571549 | 2.31480659 | +0.00090890 |
| 1024 | 64 | 2.31567246 | 2.31474688 | +0.00092558 |
| 1024 | 96 | 2.31547036 | 2.31466081 | +0.00080955 |

Interpretation:

- `1024x96` produced the best absolute doc-aware score on this 512-doc slice
- the improvement over `1024x64` was only about `0.000086 bpb`
- that is too small to treat as decisive without a larger sample or full validation
- smaller strides did not help this checkpoint

### Longer-context probe on a 256-doc MLX prefix

Command:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python phase1b_eval_sweep_mlx.py \
  --checkpoint logs/mlx_full_m2_20260322_003835_mlx_model.int8.ptz \
  --num-docs 256 \
  --seq-lens 1024,1536,2048 \
  --strides 96 \
  --batch-seqs 1 \
  --include-next-bos 1
```

Results:

| Seq Len | Stride | Flat BPB | Doc-Aware BPB | Delta |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 96 | 2.27305386 | 2.27182227 | +0.00123160 |
| 1536 | 96 | 2.28597980 | 2.27623570 | +0.00974410 |
| 2048 | 96 | 2.29375977 | 2.27822244 | +0.01553733 |

Interpretation:

- longer context was clearly worse on this checkpoint and prefix
- the large `delta_bpb` values at longer context are misleading
- what matters is the absolute doc-aware score, and that got worse as `seq_len` increased
- on this model, `1024` still looks better than `1536` or `2048`

Phase 1B conclusion:

- keep `include_next_bos=1`
- keep `seq_len=1024`
- `stride=96` is the best numerical result we measured, but only barely better than `64`
- if we want the safest default today, `1024x64` remains defensible
- if we want the best measured number on the current prefix, `1024x96` is the tentative winner

## Recommended Next Steps

1. Keep `--include-next-bos 1` as the default unless there is a strong reason to change the target set.
2. Use `seq_len=1024` for now; the longer-context probe was negative.
3. Treat `stride=96` as the tentative best measured setting, but only by a very small margin over `64`.
4. Move to Phase 2, because Phase 1 now looks close to exhausted on this local checkpoint.
