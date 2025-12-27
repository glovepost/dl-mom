# Deep-Latent Mixture of Models (DL-MoM)

This repository accompanies the paper **Deep-Latent Mixture of Models (DL-MoM): A Training-Free Architecture for System 2 Reasoning via Latent-Space Collaboration**. DL-MoM enables multiple LLMs to collaborate primarily in latent space by exchanging probabilistic beliefs (Soft Belief Packets) and merging expert preferences with a contrastive TIES-style consensus. The system uses trend-based entropy control to switch between latent exploration and explicit output, and it supports training-free KV-cache compression for efficient per-expert context retention.

## Summary of the Paper

Key contributions:
- **Soft Belief Packet protocol** for cross-model latent communication without learned alignment matrices.
- **Trend-based entropy controller** with switch-count controls for stable adaptive computation.
- **Contrastive consensus engine** (TIES-style) for merging expert preferences in logit space.
- **Training-free blueprint** with reproducible experiment registry and runner specification.

Results:
- Ablation-suite figures are included in `runs/ablations/plots/`.
- The project page (`index.html`) embeds these plots and summarizes the protocol.

## Running Ablations and Experiments (AMD Halo Strix, gfx1151)

The ablation runner is specified in Section 6 of the paper (`docs/dl-mom-paper.md`). On AMD Halo Strix (gfx1151),
use ROCm/HIP and run the suites with the `dlmom` CLI.

### Prerequisites

1. ROCm/HIP installed and visible (sanity check with `rocminfo` and `rocm-smi`).
2. `dlmom` CLI available in your environment (see the runner spec in the paper).

### Run a Suite (Example)

```bash
dlmom run \
  --suite A1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --bench gsm8k \
  --out runs/ablations \
  --device hip \
  --dtype bf16 \
  --seeds 0 1 2 \
  --max-steps 256 \
  --max-handoffs 2 \
  --max-switches 10 \
  --batch-size auto \
  --resume
```

### Aggregate Results

```bash
dlmom aggregate \
  --in runs/ablations \
  --out runs/ablations/aggregate.csv
```

### Generate Plots

```bash
dlmom plot \
  --in runs/ablations \
  --out runs/ablations/plots \
  --format png
```

### Notes

- Use `--device hip` and `--dtype bf16` for ROCm/HIP.
- Keep `--max-steps`, `--max-handoffs`, and `--max-switches` fixed across suites for comparability.
- The plotting output is expected at `runs/ablations/plots` and is embedded in the project page.
