"""
CourtAI Modal application.

Defines the GPU function that runs the full tennis detection pipeline on an A100.
Invoked non-blocking by the FastAPI backend via modal_runner.spawn_pipeline().

Usage:
    modal deploy modal_app.py
    modal run modal_app.py  (for local testing)
"""
import subprocess
import time
import logging
import json
import os
from pathlib import Path

import modal

logger = logging.getLogger(__name__)

app = modal.App("courtai")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxrender1", "libxext6")
    .pip_install(
        "ultralytics>=8.3.0",
        "opencv-python-headless>=4.10.0",
        "yt-dlp>=2024.11.4",
        "imagehash>=4.3.1",
        "Pillow>=10.4.0",
        "anthropic>=0.40.0",
        "matplotlib>=3.9.0",
        "requests>=2.32.3",
        "numpy>=1.26.0",
        "scipy>=1.14.0",
    )
)

artifacts_volume = modal.Volume.from_name("courtai-artifacts", create_if_missing=True)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def post_status(
    backend_url: str,
    job_id: str,
    stage: str,
    description: str,
    status: str = "running",
    iteration: int | None = None,
    eval_metrics: dict | None = None,
    decision: dict | None = None,
    artifacts: list[dict] | None = None,
    checkpoint_frames: list[dict] | None = None,
    error_message: str | None = None,
) -> None:
    """POST a status update back to the FastAPI backend."""
    import requests

    payload = {
        "job_id": job_id,
        "stage": stage,
        "description": description,
        "status": status,
    }
    if iteration is not None:
        payload["iteration"] = iteration
    if eval_metrics is not None:
        payload["eval_metrics"] = eval_metrics
    if decision is not None:
        payload["decision"] = decision
    if artifacts is not None:
        payload["artifacts"] = artifacts
    if checkpoint_frames is not None:
        payload["checkpoint_frames"] = checkpoint_frames
    if error_message is not None:
        payload["error_message"] = error_message

    try:
        resp = requests.post(f"{backend_url}/status/update", json=payload, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"Failed to post status update to backend: {e}")


def poll_for_checkpoint_approval(backend_url: str, job_id: str, timeout: int = 1800) -> list[dict]:
    """
    Poll the FastAPI backend until the coach submits checkpoint feedback.
    Returns the list of feedback items.
    Times out after `timeout` seconds (default: 30 minutes).
    """
    import requests

    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{backend_url}/status/{job_id}", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") not in ("awaiting_review",):
                resp2 = requests.get(f"{backend_url}/checkpoint/{job_id}/feedback", timeout=10)
                if resp2.status_code == 200:
                    return resp2.json().get("feedback", [])
                return []
        except Exception as e:
            logger.warning(f"Checkpoint poll error: {e}")
        time.sleep(5)

    logger.warning(f"Checkpoint approval timed out after {timeout}s — continuing without coach feedback")
    return []


@app.function(
    gpu="A100",
    timeout=3600,
    image=image,
    volumes={"/data": artifacts_volume},
    secrets=[
        modal.Secret.from_name("courtai-secrets"),
    ],
)
def run_pipeline(job_id: str, footage_url: str, config: dict, backend_url: str) -> dict:
    """
    Full CourtAI pipeline — runs on Modal A100.

    Stages:
        00: Load pretrained tennis model
        01: Frame extraction with dead frame skip
        02: pHash deduplication
        03: Pseudo-label generation (4 classes)
        04: Codex subagent label refinement (tennis policy)
        CHECKPOINT: Coach review
        05: YOLOv8n fine-tuning (loop up to 3x)
        06: Evaluation (per-class mAP for player, ball, court_lines, net)
        07: Decision tree
        08: Output generation (annotated video, heatmaps, serve chart, clips)
    """
    import sys
    sys.path.insert(0, "/root")

    from pipeline import (
        stage_00_load,
        stage_01_collect,
        stage_02_dedup,
        stage_03_pseudolabel,
        stage_04_label,
        stage_05_train,
        stage_06_eval,
        stage_07_decide,
        stage_08_output,
    )

    result = subprocess.run(["which", "ffmpeg"], capture_output=True)
    if result.returncode != 0:
        msg = "FFmpeg is not available in the Modal environment. Cannot generate video outputs."
        post_status(backend_url, job_id, "error", msg, status="error", error_message=msg)
        raise RuntimeError(msg)

    max_iterations = config.get("max_iterations", 3)
    all_decisions = []
    all_eval_results = []
    coach_feedback = []
    checkpoint_frames_meta = []

    try:
        # ── Stage 00: Load pretrained model ───────────────────────────────────
        post_status(backend_url, job_id, "loading_model",
                    "Loading pretrained tennis detection model (player, ball, court lines, net).")
        stage_00_result = stage_00_load.run(config)
        model_path = stage_00_result["model_path"]

        # ── Stage 01: Frame extraction ─────────────────────────────────────────
        post_status(backend_url, job_id, "extracting",
                    f"Extracting video frames at {config.get('fps', 30)} fps with dead frame skip.")
        try:
            stage_01_result = stage_01_collect.run(job_id, footage_url, config)
        except stage_01_collect.DownloadError as e:
            post_status(
                backend_url, job_id, "needs_upload",
                str(e), status="needs_upload",
                error_message=str(e)
            )
            return {"status": "needs_upload", "error": str(e)}

        train_frames = stage_01_result["train_frames"]
        test_frames = stage_01_result["test_frames"]
        video_path = stage_01_result["video_path"]
        total_frames = stage_01_result["total_frames"]
        dead_skipped = stage_01_result.get("dead_skipped", 0)

        post_status(backend_url, job_id, "extracting",
                    f"Extracted {total_frames} frames ({dead_skipped} dead frames skipped). "
                    f"{len(train_frames)} for training, {len(test_frames)} held out for testing.")

        # ── Stage 02: Deduplication ────────────────────────────────────────────
        post_status(backend_url, job_id, "deduplicating",
                    "Removing duplicate frames to keep only unique, high-information moments.")
        stage_02_result = stage_02_dedup.run(job_id, train_frames, config)
        unique_frames = stage_02_result["unique_frames"]
        checkpoint_candidates = stage_02_result["checkpoint_candidates"]

        post_status(backend_url, job_id, "deduplicating",
                    f"Kept {len(unique_frames)} unique frames after removing near-duplicates.")

        # ── Stage 03: Pseudo-labeling ──────────────────────────────────────────
        post_status(backend_url, job_id, "labeling",
                    "Running initial detection for players, ball, court lines, and net on all frames.")
        stage_03_result = stage_03_pseudolabel.run(job_id, unique_frames, model_path, config)
        pseudo_label_paths = stage_03_result["label_paths"]

        # ── Stage 04: Codex label refinement ──────────────────────────────────
        post_status(backend_url, job_id, "refining_labels",
                    "Expert review: applying tennis-specific labeling rules to all detections.")
        stage_04_result = stage_04_label.run(
            job_id=job_id,
            frame_paths=unique_frames,
            pseudo_label_paths=pseudo_label_paths,
            coach_feedback=[],
            checkpoint_frames=[],
            config=config,
        )
        refined_label_paths = stage_04_result["label_paths"]

        # ── Checkpoint: Coach review ───────────────────────────────────────────
        checkpoint_frames_meta = [
            {"frame_index": i, "frame_path": p, "overlay_path": None}
            for i, p in enumerate(checkpoint_candidates)
        ]
        post_status(
            backend_url, job_id, "awaiting_review",
            "Ready for your review. Please check the 24 sample frames and approve or flag any issues.",
            status="awaiting_review",
            checkpoint_frames=checkpoint_frames_meta,
        )

        coach_feedback = poll_for_checkpoint_approval(backend_url, job_id)

        if coach_feedback:
            post_status(backend_url, job_id, "processing_feedback",
                        "Applying your feedback to the training data.")
            stage_04_result = stage_04_label.run(
                job_id=job_id,
                frame_paths=unique_frames,
                pseudo_label_paths=pseudo_label_paths,
                coach_feedback=coach_feedback,
                checkpoint_frames=checkpoint_frames_meta,
                config=config,
            )
            refined_label_paths = stage_04_result["label_paths"]

        # ── Fine-tuning loop (max 3 iterations) ───────────────────────────────
        current_model_path = model_path
        current_config = dict(config)
        current_frames = unique_frames
        current_labels = refined_label_paths
        previous_eval = None
        dataset_yaml = None

        for iteration in range(1, max_iterations + 1):
            post_status(
                backend_url, job_id, "training",
                f"Training pass {iteration} of {max_iterations}: fine-tuning detection model on your footage.",
                iteration=iteration,
            )
            stage_05_result = stage_05_train.run(
                job_id=job_id,
                iteration=iteration,
                frame_paths=current_frames,
                label_paths=current_labels,
                model_path=current_model_path,
                config=current_config,
            )
            current_model_path = stage_05_result["best_model_path"]
            dataset_yaml = str(Path(stage_05_result["run_dir"]).parent.parent / job_id / "dataset.yaml")

            post_status(
                backend_url, job_id, "evaluating",
                f"Measuring detection accuracy after training pass {iteration}.",
                iteration=iteration,
            )
            eval_result = stage_06_eval.run(
                job_id=job_id,
                iteration=iteration,
                model_path=current_model_path,
                dataset_yaml=dataset_yaml,
                test_frames=test_frames,
                test_label_dir=str(stage_03_result["label_dir"]),
                frame_count=len(current_frames),
                config=current_config,
            )
            all_eval_results.append(eval_result)
            post_status(
                backend_url, job_id, "evaluating",
                f"Pass {iteration}: player {eval_result['player_map']:.0%}, "
                f"ball {eval_result['ball_map']:.0%}, "
                f"court lines {eval_result['court_lines_map']:.0%}, "
                f"net {eval_result['net_map']:.0%}, "
                f"FP rate {eval_result['fp_rate']:.0%}.",
                iteration=iteration,
                eval_metrics=eval_result,
            )

            decision = stage_07_decide.decide(
                iteration=iteration,
                eval_metrics=eval_result,
                previous_eval=previous_eval,
                current_config=current_config,
                max_iterations=max_iterations,
            )
            all_decisions.append(decision)
            post_status(
                backend_url, job_id, "deciding",
                decision["justification"],
                iteration=iteration,
                decision={"action": decision["action"], "justification": decision["justification"]},
            )

            if decision["action"] == "exit":
                break

            current_config.update(decision.get("config_updates", {}))

            if decision["action"] in ("mine_frames", "retrain") and current_config.get("fps", 30) > config.get("fps", 30):
                post_status(
                    backend_url, job_id, "extracting",
                    f"Collecting more frames at {current_config['fps']} fps.",
                    iteration=iteration,
                )
                stage_01_result_new = stage_01_collect.run(job_id, footage_url, current_config)
                new_frames = stage_01_result_new["train_frames"]

                stage_02_result_new = stage_02_dedup.run(job_id, new_frames, current_config)
                new_unique = stage_02_result_new["unique_frames"]

                stage_03_result_new = stage_03_pseudolabel.run(job_id, new_unique, current_model_path, current_config)
                new_pseudo = stage_03_result_new["label_paths"]

                stage_04_result_new = stage_04_label.run(
                    job_id=job_id,
                    frame_paths=new_unique,
                    pseudo_label_paths=new_pseudo,
                    coach_feedback=coach_feedback,
                    checkpoint_frames=checkpoint_frames_meta,
                    config=current_config,
                )
                current_frames = new_unique
                current_labels = stage_04_result_new["label_paths"]

            elif decision["action"] == "relabel":
                stage_04_result_new = stage_04_label.run(
                    job_id=job_id,
                    frame_paths=current_frames,
                    pseudo_label_paths=pseudo_label_paths,
                    coach_feedback=coach_feedback,
                    checkpoint_frames=checkpoint_frames_meta,
                    config=current_config,
                )
                current_labels = stage_04_result_new["label_paths"]

            previous_eval = eval_result

        # ── Stage 08: Output generation ────────────────────────────────────────
        post_status(backend_url, job_id, "generating_outputs",
                    "Rendering annotated video, heatmaps, serve placement chart, and highlight clips.")

        stage_08_result = stage_08_output.run(
            job_id=job_id,
            video_path=video_path,
            model_path=current_model_path,
            decisions=all_decisions,
            eval_results=all_eval_results,
            coach_feedback=coach_feedback,
            footage_url=footage_url,
            config=current_config,
        )

        artifacts = [
            {"type": "annotated_video", "path": stage_08_result["annotated_video"]},
            {"type": "best_model", "path": stage_08_result["best_model"]},
            {"type": "eval_report", "path": stage_08_result["eval_report"]},
            {"type": "session_json", "path": stage_08_result["session_json"]},
            {"type": "serve_chart", "path": stage_08_result["serve_chart"]},
        ]
        for hp in stage_08_result["heatmaps"]:
            artifacts.append({"type": "heatmap", "path": hp})
        for clip in stage_08_result["clips"]:
            artifacts.append({
                "type": "clip",
                "path": clip["path"],
                "metadata": {"label": clip["label"], "timestamp_sec": clip["timestamp_sec"], "clip_type": clip["type"]}
            })

        artifacts_volume.commit()

        post_status(
            backend_url, job_id, "complete",
            "Pipeline complete. Your results are ready.",
            status="complete",
            artifacts=artifacts,
        )

        return {"status": "complete", "artifact_paths": {a["type"]: a["path"] for a in artifacts}}

    except Exception as e:
        import traceback
        msg = f"Pipeline failed: {str(e)}"
        logger.error(f"{msg}\n{traceback.format_exc()}")
        post_status(backend_url, job_id, "error", msg, status="error", error_message=msg)
        raise


@app.local_entrypoint()
def main():
    """Local test entrypoint for development."""
    test_config = {
        "fps": 30,
        "max_iterations": 1,
        "train_split": 0.75,
        "epochs": 5,
        "batch_size": 4,
        "dead_frame_motion_threshold": 0.02,
    }
    run_pipeline.remote(
        job_id="test-job-001",
        footage_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        config=test_config,
        backend_url="http://localhost:8000",
    )
