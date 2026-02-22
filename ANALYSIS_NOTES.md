# Analysis UI & Data Contract (Merge Guide)

This file documents the analysis stack so we avoid merge conflicts and keep a single source of truth.

## Components to keep (owned here)
- `frontend/components/SideBySidePlayer.tsx` — twin video player (raw + overlay).
- `frontend/components/PointDeck.tsx` — point cards with coaching summary, clip, and jump.
- `frontend/components/AnalysisDashboard.tsx` — quality/serve/rally/errors/movement/pace + atomic log.

## Data contract
- `frontend/lib/types.ts` defines `AnalysisData` (schema for `analysis.json`). Treat this as canonical.
- Backend writer: `backend/analysis.py` builds `analysis.json` (wired in `backend/main.py` and `backend/modal_runner.py`).

## Layout wiring (results page)
- File: `frontend/app/results/[jobId]/page.tsx`
  - Use `SideBySidePlayer` for videos.
  - Use `PointDeck` (replaces PointTimeline/CoachingCards/HighlightClips).
  - Render **one** `AnalysisDashboard` (no duplicates).
  - Keep right-column blocks: `ServePlacementChart`, `HeatmapViewer`, `DownloadPanel`, Match Stats, Insights.

## Merge rules
- Schema first: update `frontend/lib/types.ts` and `backend/analysis.py` together. Add fields; do not replace the file.
- If conflicts touch `SideBySidePlayer`, `PointDeck`, or `AnalysisDashboard`, keep these versions and layer styling tweaks on top.
- Run `npm run lint` before pushing.

## Data flow
- Backend writes `outputs/{job_id}/analysis.json` from `run.json`, `events.json`, `points.json`, `timeseries/*`, `visuals/*`.
- Frontend fetches `/results/{job_id}/data`; `analysis` must match `AnalysisData`.

## Quick checklist before merge
- Lint: `npm run lint`.
- Verify `analysis.json` exists for a sample run and loads without duplicate dashboards.
- Ensure only one `AnalysisDashboard` is rendered.
