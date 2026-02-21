# HooperAI 🏀

Basketball vision intelligence for college coaches.
Paste a YouTube link. Get a trained detector, annotated footage, highlight clips, and heatmaps — in under 30 minutes.


What It Does
HooperAI takes raw college basketball footage and runs a fully automated, agent-driven computer vision pipeline that produces:

A fine-tuned YOLOv8n detection model adapted to your specific footage
Side-by-side sync video — raw footage vs. HooperAI-analyzed footage, perfectly synced
Auto-extracted highlight clips — shot attempts and defensive breakdowns with timestamps
Per-quarter player movement heatmaps
Eval report — per-class mAP, FP rate, generalization score, full agent decision log

No ML expertise required from the coach.

How It Works
HooperAI runs a two-stage training pipeline with an adaptive agent loop:
YouTube URL
    │
    ▼
Frame Extraction (OpenCV + yt-dlp)
    │  fps=2 start (intentionally conservative)
    │  75/25 auto-split: train / generalization test
    ▼
pHash Deduplication (imagehash + Actian VectorAI DB)
    │  Near-duplicate frames dropped
    │  Embeddings indexed for diversity sampling
    ▼
Pseudo-Labeling (Pretrained YOLOv8n on SportsMOT)
    │  First-pass YOLO labels generated
    ▼
Label Refinement (Parallel Codex Subagents)
    │  Basketball labeling policy enforced:
    │  on-court players only, consistent rim boxes
    │  Implausible shards auto re-queued
    ▼
Coach Checkpoint Review
    │  24 maximally diverse frames (Actian VectorAI)
    │  Approve / flag / note — feedback saved to SQLite
    ▼
Fine-Tuning on Modal A100 (YOLOv8n)
    ▼
Eval → Decision Tree
    │  mAP below floor → increase fps, mine hard frames
    │  FP rate high → tighten bench/crowd rules
    │  All criteria met → exit loop
    │  Max 3 iterations
    ▼
Output Generation (FFmpeg + matplotlib)
    │  Annotated overlay video
    │  Per-quarter heatmaps
    │  Shot attempt + defensive breakdown clips
    ▼
Results Dashboard
    Side-by-side sync player · Highlight clips · Heatmaps · Eval report
