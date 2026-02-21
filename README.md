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

Tech Stack
LayerTechnologyFrontendNext.js + Tailwind CSSBackend APIFastAPI (Python)GPU ComputeModal (A100 serverless)Job StateSQLiteFrame ExtractionOpenCV + yt-dlpDeduplicationimagehash + Actian VectorAI DBDetection + TrainingUltralytics YOLOv8nLabeling SubagentsOpenAI Codex APIVideo OutputOpenCV + FFmpegHeatmapsmatplotlibArtifact StorageModal Volume

Project Structure
hooperai/
  frontend/                        # Next.js app
    pages/
      index.tsx                    # Upload page
      results/[jobId].tsx          # Results dashboard
    components/
      UploadForm.tsx
      ProgressTracker.tsx          # Live stage + decision log
      CheckpointReview.tsx         # 24-frame approve/flag UI
      SideBySidePlayer.tsx         # Synced dual video — centerpiece
      HeatmapViewer.tsx            # Per-quarter heatmap tabs
      HighlightClips.tsx           # Scrollable clip reel
      EvalReport.tsx               # Metrics + pass/fail card

  backend/
    main.py                        # FastAPI routes
    db.py                          # SQLite helpers
    modal_runner.py                # Modal function invocation

  pipeline/                        # All stages — runs inside Modal
    stage_00_load.py               # Load SportsMOT pretrained model
    stage_01_collect.py            # yt-dlp + OpenCV + 75/25 split
    stage_02_dedup.py              # pHash + Actian VectorAI
    stage_03_pseudolabel.py        # YOLOv8n inference
    stage_04_label.py              # Codex subagents + QC
    stage_05_train.py              # YOLOv8n fine-tuning
    stage_06_eval.py               # mAP, FP rate, gen score
    stage_07_decide.py             # Decision tree
    stage_08_output.py             # FFmpeg overlays + clips

  modal_app.py                     # Modal function + image spec
  PRD.md                           # Full product requirements document
  .env                             # Environment variables (never commit)
  data/
    sportsmot_baseline/            # Pretrained checkpoint

Prerequisites

Python 3.10+
Node.js 18+
FFmpeg installed and on PATH
Docker (for Actian VectorAI DB)
Modal account (modal.com — $30 free credits)
OpenAI API key (console.OpenAI.com)


Setup
1. Clone the repo
bashgit clone https://github.com/yourusername/hooperai.git
cd hooperai
2. Install Python dependencies
bashpip install ultralytics opencv-python yt-dlp imagehash OpenAI \
            matplotlib fastapi uvicorn python-multipart requests \
            aiosqlite python-dotenv modal
3. Install Node dependencies
bashcd frontend
npm install
cd ..
4. Set up environment variables
bashcp .env.example .env
Fill in .env:
OpenAI_API_KEY=your_key_here
NEXT_PUBLIC_API_URL=http://localhost:8000
5. Authenticate Modal
bashmodal token new
6. Create Modal Volume for artifacts
bashmodal volume create hooperai-artifacts
7. Start Actian VectorAI DB
bashdocker pull williamimoh/actian-vectorai-db:1.0b
docker run -p 5432:5432 williamimoh/actian-vectorai-db:1.0b
8. Verify Modal GPU access
bashmodal run test_modal.py
# Should print: GPU available: True

Running Locally
bash# Terminal 1 — FastAPI backend
cd backend
uvicorn main:app --reload --port 8000

# Terminal 2 — Next.js frontend
cd frontend
npm run dev

# Open http://localhost:3000

Environment Variables
VariableDescriptionOpenAI_API_KEYOpenAI API key for Codex labeling subagentsNEXT_PUBLIC_API_URLFastAPI backend URL (localhost:8000 for local dev)
Modal credentials are stored locally via modal token new and do not need to be in .env.

Agent Decision Loop
HooperAI starts with a deliberately conservative config (fps=2, ~150 frames) to guarantee the decision tree fires on the first iteration. A typical run looks like:
Iteration 1  mAP 0.51  →  "Below floor. Increasing fps to 4, mining paint frames."
Iteration 2  mAP 0.67  →  "FP rate elevated. Tightening bench/crowd rules."
Iteration 3  mAP 0.73  →  "All exit criteria met. Exiting loop."
Exit criteria: player mAP@50 ≥ 0.70, rim mAP@50 ≥ 0.65, FP rate below threshold, minimum 500 labeled frames.

Basketball Labeling Policy
Enforced by Codex subagents across all shards:

Players — on-court only, ignore bench and crowd
Rim — tight consistent box around rim + net region
Ball — confidence tiers: certain → label, probable → label with flag, ambiguous → skip


Outputs
All artifacts written to Modal Volume under runs/<job_id>/:
runs/<job_id>/
  weights/best.pt              # Fine-tuned model
  outputs/
    annotated_video.mp4        # Overlay video for sync player
    heatmap_q1.png             # Per-quarter heatmaps
    heatmap_q2.png
    heatmap_q3.png
    heatmap_q4.png
    clips/                     # Auto-extracted highlight clips
  eval_results.json            # Per-class metrics + gen score
  session.json                 # Saved for next game run

Data Science Complexity
HooperAI spans multiple data science domains:

Computer Vision — object detection, domain adaptation, transfer learning
Semi-supervised Learning — pseudo-label generation from pretrained model
Data Engineering — frame sampling strategy, pHash deduplication, embedding-based diversity sampling
Generative AI — LLM-powered parallel labeling subagents with domain-specific policy enforcement
ML Evaluation — per-class mAP@50, FP rate, generalization score across train/test domains
Data Analytics — spatial heatmap generation, event detection for highlight clip extraction

