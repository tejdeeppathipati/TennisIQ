# Workflow

1. Place raw video in `data/raw/`.
2. Train/prepare court and ball models from `tennisiq/cv/*`.
3. Run `python -m tennisiq.pipeline.run_all`.
4. Inspect outputs in `outputs/runs/<timestamp>/`.
