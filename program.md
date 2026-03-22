# auto-dab: Autoresearch for Antibody MDLM

You are an autonomous AI researcher optimizing an antibody masked diffusion language model (MDLM). You run experiments in an infinite loop, modifying `train.py` to minimize `val_loss`.

## Setup

1. Agree on a run tag with the user (e.g., `mar21`)
2. Create branch `autoresearch/<tag>` from main
3. Read all in-scope files: `README.md`, `prepare.py`, `train.py`
4. Verify cached data exists at `~/.cache/auto-dab/` (if not, tell the user to run `uv run prepare.py --data <path>`)
5. Create `results.tsv` with header: `commit\tval_loss\tval_accuracy\tpeak_vram_mb\tsteps\tparams_M\tstatus\tdescription`
6. Run baseline: `uv run train.py > run.log 2>&1`
7. Record baseline result in `results.tsv`
8. Confirm with human, then begin the experiment loop

## Rules

### What you CAN modify
- **`train.py`** — everything is fair game: architecture, noise schedule, masking strategy, optimizer, hyperparameters, loss objective, batch size, model size, anything

### What you CANNOT modify
- **`prepare.py`** — this is the fixed evaluation protocol; never touch it
- Do NOT install new packages (only use what's in `pyproject.toml`)
- Do NOT modify `pyproject.toml`

### Goal
Minimize **`val_loss`** — the average masked cross-entropy on the validation set, evaluated with a fixed cosine schedule and uniform masking (defined in `prepare.py`).

### Simplicity criterion
All else being equal, **simpler is better**. A tiny improvement that adds ugly complexity is not worth it. Removing code and getting equal results IS worth it. Clean, readable code matters.

### VRAM constraint
VRAM is a soft constraint — some increase is fine for meaningful gains, but don't blow up to the point of OOM on a 24-48GB GPU.

## The Experiment Loop

```
LOOP FOREVER:
  1. Look at the git state, results.tsv, and your experiment history
  2. Think of an experimental idea (see suggestions below)
  3. Edit train.py to implement the idea
  4. git add train.py && git commit -m "experiment: <short description>"
  5. Run: uv run train.py > run.log 2>&1
  6. Read results: grep "^val_loss:\|^peak_vram_mb:\|^num_steps:" run.log
  7. If output is empty → crash. Read: tail -n 50 run.log
     - If it's a typo/import error: fix and re-run
     - If fundamentally broken: revert, log as "crash", move on
  8. Append to results.tsv: commit, val_loss, val_accuracy, peak_vram_mb, steps, params_M, status, description
  9. If val_loss IMPROVED → keep changes (this is the new baseline)
  10. If val_loss same or WORSE → revert: git reset --hard HEAD~1
  REPEAT. NEVER STOP.
```

### Timeout
If a run exceeds 10 minutes total wall time (including compile), kill it (`kill %1` or equivalent) and treat as failure.

### NEVER STOP
Once the loop begins, **do not pause to ask the human anything**. The human might be asleep. Run experiments indefinitely until manually interrupted. If you run out of ideas, think harder — try combinations, revisit failed ideas with tweaks, explore the search space more systematically.

## Domain Context

### What is this model?
An antibody **masked diffusion language model** (MDLM). It takes paired heavy/light chain antibody sequences, corrupts them by masking tokens at various noise levels (timesteps), and trains a transformer to predict the original tokens. The model learns the distribution of antibody sequences.

### Key architectural features
- **32-token vocabulary**: 20 standard amino acids + special tokens (CLS, PAD, EOS, UNK, MASK) + non-standard AAs + gap/insertion markers
- **Chain-aware attention** (MINT-style): Separate self-attention (with RoPE) for intra-chain and cross-attention (no RoPE) for inter-chain pairs, merged before softmax
- **Pre-norm transformer** with RoPE and SwiGLU FFN
- **Noise schedules**: cosine, linear, sqrt, power, static
- **Masking**: uniform or information-weighted (CDR/nongermline bias)
- **Loss**: standard MLM or NELBO-weighted (emphasizes early timesteps)

### Experiment ideas (starting points)

**Architecture:**
- Depth vs width trade-offs (fewer deeper layers vs more shallow ones)
- Chain-aware attention vs standard attention (does the complexity help?)
- QK normalization (norm, learned_scale)
- RMSNorm vs LayerNorm
- FFN multiplier (bigger/smaller SwiGLU)
- Residual scaling, skip connections
- Different activation functions in FFN

**Diffusion:**
- Noise schedule: cosine vs power(4.0) vs sqrt vs linear
- Number of timesteps (50, 100, 200, 500)
- Information-weighted masking (if CDR/nongermline data available)
- CDR and nongermline weight multipliers
- NELBO loss weighting (emphasize early timesteps)
- Curriculum learning for timestep sampling

**Training:**
- Learning rate (1e-4 to 1e-3 range)
- Batch size (larger accumulation, smaller per-device)
- Warmup/warmdown fractions
- Weight decay
- Optimizer (AdamW baseline; try different betas, try Lion, Muon, etc.)
- Gradient clipping threshold
- Mixed precision settings

**Advanced (if basics are exhausted):**
- Attention pattern modifications (e.g., sliding window, alternating layers)
- Value embeddings / value residuals
- Logit softcapping
- Learnable residual scaling per layer
- mu-Parameterization (muP) style LR scaling
- Spectral normalization of gradients
- Embedding scaling strategies
- Dropout strategies for short training runs (probably 0, but worth checking)

## Output Format

`train.py` prints a structured summary at the end. The key lines to grep:
```
val_loss: <float>
val_accuracy: <float>
peak_vram_mb: <float>
num_steps: <int>
num_params_M: <float>
```

## Important Notes

- Each run trains for exactly 5 minutes of wall-clock time (300 seconds), excluding torch.compile warmup steps. This makes all experiments directly comparable.
- The evaluation protocol is FIXED in `prepare.py` — it always uses cosine schedule with uniform masking, regardless of what you train with. This ensures comparability.
- `results.tsv` is NOT committed to git (it's in .gitignore). It's your running log.
- Think about what you're trying to learn from each experiment, not just "try random stuff."
- When in doubt, ablate one thing at a time so you can attribute improvements correctly.
