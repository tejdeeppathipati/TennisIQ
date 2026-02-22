# HooperAI

Basketball vision intelligence for college coaches.
Paste a YouTube link. Get a trained detector, annotated footage, highlight clips, and heatmaps — in under 30 minutes.

## What It Does

HooperAI takes raw college basketball footage and runs a fully automated, agent-driven computer vision pipeline:

- Fine-tuned YOLOv8n detection model adapted to your specific footage
- Side-by-side sync video — raw footage vs. HooperAI-analyzed footage, perfectly synced
- Auto-extracted highlight clips — shot attempts and defensive breakdowns with timestamps
- Per-quarter player movement heatmaps
- Eval report — per-class mAP, FP rate, generalization score, full agent decision log

No ML expertise required from the coach.

## Project Structure

```
hooperai/
  frontend/          # Next.js + Tailwind (deploys to Vercel)
  backend/           # FastAPI + SQLite (deploys to Railway)
  pipeline/          # Pipeline stages 00-08 (runs inside Modal)
  modal_app.py       # Modal function definition (A100 GPU)
```

## Prerequisites

- Python 3.11+
- Node.js 18+
- [Modal account](https://modal.com) with API token
- [Anthropic API key](https://console.anthropic.com)
- Actian VectorAI DB connection string (optional — fallback to random sampling if unavailable)

## Setup

### Backend

```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
# Edit .env with your Modal token, Anthropic key, etc.
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local
# Edit .env.local with your backend URL
npm run dev
```

### Modal Pipeline

```bash
# From repo root
pip install modal
modal token set --token-id YOUR_TOKEN_ID --token-secret YOUR_TOKEN_SECRET
modal deploy modal_app.py
```

## Environment Variables

### Backend (`backend/.env`)

| Variable | Description |
|---|---|
| `MODAL_TOKEN_ID` | Modal API token ID |
| `MODAL_TOKEN_SECRET` | Modal API token secret |
| `ANTHROPIC_API_KEY` | Anthropic API key for Codex subagents |
| `ACTIAN_VECTORAI_URL` | Actian VectorAI DB connection string (optional) |
| `FRONTEND_URL` | Frontend origin for CORS (e.g. https://hooperai.vercel.app) |

### Frontend (`frontend/.env.local`)

| Variable | Description |
|---|---|
| `NEXT_PUBLIC_API_URL` | FastAPI backend URL (e.g. http://localhost:8000) |

## Pipeline Stages

| Stage | Description |
|---|---|
| 00 | Load pretrained YOLOv8n (SportsMOT baseline) |
| 01 | Frame extraction via yt-dlp + OpenCV (fps=2 conservative start) |
| 02 | pHash deduplication + Actian VectorAI DB indexing |
| 03 | Pseudo-label inference with pretrained model |
| 04 | Codex subagent label refinement with basketball policy |
| CHECKPOINT | Coach reviews 24 diverse frames, approves/flags |
| 05 | YOLOv8n fine-tuning on A100 |
| 06 | Eval — mAP@50, FP rate, generalization score |
| 07 | Decision tree — loop or exit (max 3 iterations) |
| 08 | FFmpeg overlay video + heatmaps + highlight clips |

## Cost

Approximately $1.10 per full pipeline run on Modal A100. Modal offers $30 free credits on signup.

## Demo Flow

1. Coach pastes YouTube URL and clicks "Run HooperAI"
2. Live progress tracker shows each pipeline stage + agent decisions in plain English
3. Coach reviews 24 sample frames at checkpoint (approve/flag/note)
4. Pipeline fine-tunes and iterates (up to 3 times) until accuracy targets are met
5. Results page loads with side-by-side sync player, heatmaps, and highlight clips
