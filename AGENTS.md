# Repository Guidelines

## Project Structure & Modules
- `src/` — core code: training (`train_constellation.py`, `training_constellation.py`), data loaders (`loaders/`), models (`models/`), utilities (`utils/`), viz/analysis.
- Datasets: `constellation_diagrams/` (SNR‑preserving images), `data/` (HDF5 inputs, not tracked).
- Outputs: `perturbed_constellations/`, `checkpoints/`, `confusion_matrices/`, `f1_scores/`.
- Docs/config: `README.md`, `PAPER.md` (research notes), `CLAUDE.md` (latest findings), `pyproject.toml`, `CYCLIC_LR_GUIDE.md`, `sweep_*.yml`.
- Papers: latest iteration under `papers/ELSP_Paper/` (see `results/README.md`); older drafts in `papers/` root.

## Build, Test, and Dev Commands
- Env: `uv sync` (Python ≥3.9; dev uses Python 3.11 + PyTorch 2.4.x per CLAUDE.md).
- Generate constellations (SNR‑preserving): `uv run python src/generate_snr_preserving_constellations.py --h5_dir data/split_hdf5 --output_dir constellation_diagrams`.
- Train (default CE for modulation+SNR, cycle‑aware LR): `uv run python src/train_constellation.py --model_type swin_tiny --use_pretrained true --batch_size 256 --epochs 100`.
- Recommended stable setup from sweeps: `--model_type resnet50 --snr_layer_config bottleneck_64 --base_lr 1e-4 --batch_size 128`.
- Resume: `uv run python src/train_constellation.py --checkpoint checkpoints/best_model_...pth`.
- Evaluate: `uv run python src/test_constellation.py --model_checkpoint checkpoints/best.pth --data_dir constellation_diagrams --perturbation_dir perturbed_constellations`.
- Perturbations: `uv run python src/perturb_constellations.py --percents 1 5 10`.
- Quick logic check: `uv run python test_cycle_aware_patience.py`.

## Coding Style & Naming
- Python: 4‑space indents; max line length 120 (`.flake8`); explicit imports.
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`; CLI flags `--kebab-case`.
- Mirror file purpose (e.g., `validate_constellation.py`, `image_utils.py`).

## Testing Guidelines
- Script‑based tests (no pytest configured). Use `test_*.py` and run via `uv run python <file>.py`.
- Keep tests deterministic and data‑light; document any path assumptions.

## Architecture & Preprocessing
- Task: joint modulation+SNR; 272‑class grid when using discrete SNR.
- Current code: SNR classification with CrossEntropyLoss; CLAUDE.md notes SNR regression (SmoothL1) as a new option under exploration.
- Architecture insights (sweeps): ResNet50 with `bottleneck_64/128` SNR head performs best; pretrained weights are critical; high LR (1e‑3) can be unstable.
- Constellation generation must preserve SNR (power normalization). Avoid per‑image histogram normalization.

## Commit & Pull Request Guidelines
- Commits: imperative, concise (e.g., `Fix checkpoint loading for direct state dict format`, `Implement cycle-aware patience`).
- PRs: purpose, linked issues, repro commands, and sample outputs (plots under `confusion_matrices/`, `f1_scores/`); note data requirements and compatibility.

## Security & Config
- Keep secrets (e.g., `WANDB_API_KEY`) out of VCS; use `.env`/env vars.
- GPU usage must be gated by `torch.cuda.is_available()`; provide CPU fallbacks.
- All paths configurable via CLI; avoid user‑specific hard‑coding.
