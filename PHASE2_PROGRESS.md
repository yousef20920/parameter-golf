# Phase 2 Progress

## Goal

Phase 2 is attacking the quantization and export path.

The local prototype here uses the MLX checkpoint because it is runnable on the Mac and lets us test export ideas quickly before touching the more complex PyTorch record script.

## What Changed

### 1. Added category-specific clip percentiles to the MLX int8 exporter

File:

- `train_gpt_mlx.py`

Changes:

- added tensor classification for export categories:
  - `embed`
  - `mlp`
  - `attn`
  - `bigram`
  - `other`
- added configurable clip percentiles per category
- changed `quantize_float_array(...)` to accept an explicit quantile
- changed `quantize_state_dict_int8(...)` to accept a clip-percentile config
- recorded the applied clip quantile in quantization metadata

Default behavior remains the same when no override config is provided.

### 2. Added a local quantization sweep runner

File:

- `phase2_quant_sweep_mlx.py`

Purpose:

- load the raw `.npz` checkpoint
- quantize it with multiple clip presets
- serialize each artifact with `pickle + zlib`
- dequantize the result back into the model
- measure roundtrip validation quality on a real validation prefix

### 3. Ported the clipping idea into the PyTorch mixed int5/int6 exporter

File:

- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py`

Changes:

- added low-bit row clip percentiles by category:
  - `LOWBIT_ROW_CLIP_PERCENTILE`
  - `LOWBIT_ROW_CLIP_PERCENTILE_EMBED`
  - `LOWBIT_ROW_CLIP_PERCENTILE_MLP`
  - `LOWBIT_ROW_CLIP_PERCENTILE_ATTN`
  - `LOWBIT_ROW_CLIP_PERCENTILE_BIGRAM`
  - `LOWBIT_ROW_CLIP_PERCENTILE_OTHER`
- changed the mixed int5/int6 row quantizer to support percentile clipping instead of only max-abs scaling
- kept the default behavior backward-compatible:
  - `100.0` means old max-abs behavior
- added export-time logging so the active low-bit clip settings are visible in the run log

Suggested first H100 try:

```bash
LOWBIT_ROW_CLIP_PERCENTILE_MLP=99.99 \
LOWBIT_ROW_CLIP_PERCENTILE_ATTN=99.9999
```

Everything else can stay at the default `100.0` initially.

## Phase 2 Presets Tested

The first sweep used these presets:

- `baseline`
  - all categories at `99.99984`
- `mlp_9999`
  - `mlp=99.99`
- `mlp_999`
  - `mlp=99.9`
- `attn_safe_mlp_9999`
  - `attn=99.9999`
  - `mlp=99.99`
  - others unchanged

## Results

### A. 128-doc doc-aware triage

Command:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python phase2_quant_sweep_mlx.py \
  --checkpoint logs/mlx_full_m2_20260322_003835_mlx_model.npz \
  --checkpoint-kind raw \
  --num-docs 128 \
  --seq-len 1024 \
  --stride 96 \
  --batch-seqs 2 \
  --include-next-bos 1 \
  --eval-mode doc \
  --presets baseline,mlp_9999,mlp_999,attn_safe_mlp_9999
```

Raw model reference:

- `doc_bpb = 2.31020174`

Quantized results:

| Preset | Artifact Bytes | Doc-Aware BPB | Quantized Gap vs Raw |
| --- | ---: | ---: | ---: |
| `baseline` | 8,996,130 | 2.31359530 | +0.00339356 |
| `mlp_9999` | 8,995,988 | 2.31355062 | +0.00334888 |
| `mlp_999` | 8,994,417 | 2.31360806 | +0.00340633 |
| `attn_safe_mlp_9999` | 8,995,995 | 2.31351123 | +0.00330949 |

Result:

- best preset on this sweep was `attn_safe_mlp_9999`

### B. 256-doc confirmation

Command:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python phase2_quant_sweep_mlx.py \
  --checkpoint logs/mlx_full_m2_20260322_003835_mlx_model.npz \
  --checkpoint-kind raw \
  --num-docs 256 \
  --seq-len 1024 \
  --stride 96 \
  --batch-seqs 2 \
  --include-next-bos 1 \
  --eval-mode doc \
  --presets baseline,attn_safe_mlp_9999
```

Raw model reference:

- `doc_bpb = 2.26873723`

Quantized results:

| Preset | Artifact Bytes | Doc-Aware BPB | Quantized Gap vs Raw |
| --- | ---: | ---: | ---: |
| `baseline` | 8,996,130 | 2.27182227 | +0.00308503 |
| `attn_safe_mlp_9999` | 8,995,995 | 2.27174059 | +0.00300336 |

Result:

- `attn_safe_mlp_9999` was still better on the larger sample
- improvement over baseline was about `0.00008168 bpb`
- it also made the compressed artifact `135` bytes smaller

## Interpretation

What this means:

- category-specific clipping is a real lever
- the effect is small, but it was consistent across `128` and `256` docs
- making MLP clipping slightly more aggressive while keeping attention a little safer helped

What this does not mean:

- this is not yet a leaderboard-scale win
- this does not prove the same setting will be best in the PyTorch mixed int5/int6 exporter
- this was not full validation

One useful detail:

- payload bytes were unchanged across presets
- the on-disk artifact changed slightly because the quantized values compressed differently under `zlib`

## Current Best Local Phase 2 Preset

For the current MLX checkpoint and the tested doc-aware eval setting:

- `attn_safe_mlp_9999`

That means:

- `attn` clip percentile: `99.9999`
- `mlp` clip percentile: `99.99`
- everything else: `99.99984`

## Recommended Next Phase 2 Steps

1. Test a few nearby presets around the current winner:
   - slightly safer attention
   - slightly more aggressive MLP
   - maybe a non-default `other`
2. Add one genuinely new export idea after clipping:
   - outlier-row fp16 keep path
   - layer-specific clipping
   - layer-specific keep-float rules
3. After the MLX path is stable, port the best idea into the PyTorch mixed int5/int6 exporter.

Status update:

- that port is now done
- the next step is to run the PyTorch record script with the new env vars and compare roundtrip score and bytes
