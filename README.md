# TennisIQ

Tennis vision intelligence for college coaches.
Upload an MP4 or paste a YouTube link — get court detection, ball tracking, player analysis, point segmentation, and coaching insights powered by Modal T4 GPU.

## What It Does

TennisIQ takes tennis match footage and runs a fully automated inference pipeline:

- **Court detection** — ResNet50 keypoint regression (14 court keypoints per frame)
- **Ball tracking** — YOLOv5 detection with outlier removal and interpolation
- **Player detection** — YOLOv8n + ByteTrack with HOG fallback and carry-forward
- **Homography** — pixel-to-court coordinate mapping with confidence scoring
- **Event detection** — heuristic bounce/hit classification with in/out calls
- **Point segmentation** — groups events into tennis points with serve zone + fault detection
- **Visual outputs** — annotated overlay video, serve placement chart, error heatmap, player movement heatmaps, per-point clips, coaching cards

No ML expertise required from the coach.

## Project Structure

```
TennisIQ/
  frontend/          # Next.js + Tailwind (results dashboard)
  backend/           # FastAPI + SQLite (API + job orchestration)
  tennisiq/          # CV pipeline (court, ball, player, events, points, output)
  checkpoints/       # Model weights (not tracked in git)
  outputs/           # Pipeline results per job (not tracked in git)
  modal_app.py       # Legacy Modal app (training pipeline)
```

## Prerequisites

- Python 3.11+
- Node.js 18+
- [Modal account](https://modal.com) with API token
- Model checkpoints in `checkpoints/` directory:
  - `court_resnet/keypoints_model.pth`
  - `ball_yolo5/yolo5_last.pt`

## Setup

### Backend

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your Modal token
python -m uvicorn main:app --host 127.0.0.1 --port 8002
```

### Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local
# Edit .env.local: NEXT_PUBLIC_API_URL=http://localhost:8002
npm run dev
```

### Deploy Modal Pipeline

```bash
# From repo root
pip install modal
modal token set --token-id YOUR_TOKEN_ID --token-secret YOUR_TOKEN_SECRET
modal deploy tennisiq/modal_court.py
```

## Environment Variables

### Backend (`backend/.env`)

| Variable | Description |
|---|---|
| `BACKEND_URL` | Self-referencing URL for status callbacks (default: `http://localhost:8002`) |
| `FRONTEND_URL` | Frontend origin for CORS (default: `http://localhost:3000`) |
| `OUTPUTS_DIR` | Path to pipeline output directory (default: `../outputs`) |

### Frontend (`frontend/.env.local`)

| Variable | Description |
|---|---|
| `NEXT_PUBLIC_API_URL` | FastAPI backend URL (e.g. `http://localhost:8002`) |

## Pipeline Stages (per 10-second segment)

| Phase | Description |
|---|---|
| 1 | Court keypoint detection (ResNet50, 14 keypoints) |
| 2 | Homography computation + confidence scoring |
| 3 | Ball detection (YOLOv5) + track cleaning |
| 4 | Ball physics (court-space speed/acceleration) |
| 5 | Player detection (YOLOv8n + ByteTrack, HOG fallback) |
| 6 | Event detection (bounces + hits, in/out classification) |
| 6b | Point segmentation (serve zones, fault types, rally metrics) |
| 7 | Structured output (JSON + coaching cards + visual data) |
| 8 | Overlay video rendering + per-point clip extraction |

## How It Works

1. Coach uploads MP4 or pastes YouTube URL
2. Backend splits video into 10-second segments
3. Each segment is sent to Modal T4 GPU for inference
4. Results are accumulated and merged across all segments
5. Results page shows side-by-side video player, point timeline, coaching cards, heatmaps, and downloads
