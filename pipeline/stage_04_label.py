"""
Stage 04: Codex subagent label refinement with tennis labeling policy.

Splits frames into shards and spawns parallel Codex subagents to review
and refine pseudo-labels using the tennis labeling policy:
  - Players: on-court players only (exclude crowd, ball persons, umpires)
  - Ball: first-class citizen — certain -> label, probable -> label w/ flag, ambiguous -> skip
  - Court lines: baseline, service line, center service line, sidelines
  - Net: full net structure including posts
  - QC: implausible shards re-queued up to 3 times before error surface

Each subagent receives a shard of (frame, pseudo_label) pairs and returns
refined label files.
"""
import json
import logging
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

logger = logging.getLogger(__name__)

REFINED_LABELS_DIR = Path("/data/refined_labels")
MAX_RETRIES = 3

TENNIS_LABELING_POLICY = """
You are a tennis video annotation expert. Your task is to review and refine YOLO-format bounding box labels for tennis match footage.

CLASS DEFINITIONS:
- Class 0: player (on-court players only)
- Class 1: ball (tennis ball)
- Class 2: court_lines (baseline, service line, center service line, sidelines)
- Class 3: net (net structure including posts)

TENNIS LABELING POLICY:

1. PLAYERS (Class 0): Include only players who are actively on the court. Exclude:
   - Ball persons / ball kids
   - Chair umpire and line judges
   - Coaches and spectators in stands
   - Players warming up on adjacent courts
   - Partial players at frame edges with less than 30% of body visible

2. BALL (Class 1) — first-class detection target:
   - Certain (high contrast against court/sky, clear round shape): ALWAYS label
   - Probable (partial view, motion blur streak): label with confidence flag
   - Ambiguous (too small, occluded, or lost against background): skip
   - The ball is small (~3-8 px in many frames) — accept tight bounding boxes
   - During serve: ball toss should be labeled if clearly visible

3. COURT LINES (Class 2):
   - Label visible baseline, service lines, center service line, and sidelines
   - Use elongated bounding boxes that trace the line segments
   - Only label lines that are clearly visible (not hidden by shadows or players)
   - Skip partially visible lines at extreme frame edges

4. NET (Class 3):
   - Label the full net structure, including the net cord and posts when visible
   - Use a bounding box that covers the entire visible net span
   - In close-up shots where only part of the net is visible, label what is shown

5. QC CHECKS — reject the shard and flag for re-queue if:
   - More than 4 player boxes in a single frame (singles has 2, doubles has 4)
   - Any box covers more than 50% of the frame area
   - Average box area > 30% of frame area across the shard
   - No boxes on any frame (suspicious for match footage)
   - More than 2 ball detections per frame (max 1 ball in play)

For each frame, return the refined labels in YOLO format:
class_id cx cy w h
(all values normalized 0-1)

Return a JSON object with:
{
  "frames": [
    {
      "frame_path": "<path>",
      "labels": [{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.2}, ...],
      "notes": "<optional note>"
    }
  ],
  "shard_valid": true,
  "rejection_reason": null
}
"""


def read_label_file(label_path: str) -> list[dict]:
    """Read YOLO format label file into list of detection dicts."""
    path = Path(label_path)
    if not path.exists():
        return []
    detections = []
    for line in path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            detections.append({
                "class_id": int(parts[0]),
                "cx": float(parts[1]),
                "cy": float(parts[2]),
                "w": float(parts[3]),
                "h": float(parts[4]),
            })
    return detections


def write_refined_label(label_path: Path, labels: list[dict]) -> None:
    """Write refined labels to YOLO format file."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{l['class_id']} {l['cx']:.6f} {l['cy']:.6f} {l['w']:.6f} {l['h']:.6f}"
        for l in labels
    ]
    label_path.write_text("\n".join(lines))


def is_shard_valid(shard_result: dict) -> tuple[bool, Optional[str]]:
    """QC check on shard refinement output."""
    if not shard_result.get("shard_valid", True):
        return False, shard_result.get("rejection_reason", "Subagent flagged shard as invalid")

    for frame_data in shard_result.get("frames", []):
        labels = frame_data.get("labels", [])
        player_boxes = [l for l in labels if l["class_id"] == 0]
        ball_boxes = [l for l in labels if l["class_id"] == 1]

        if len(player_boxes) > 4:
            return False, f"Implausible detection: {len(player_boxes)} players in one frame (max 4 for doubles)"

        if len(ball_boxes) > 2:
            return False, f"Too many ball detections: {len(ball_boxes)} in one frame"

        for l in labels:
            area = l["w"] * l["h"]
            if area > 0.50:
                return False, f"Box covers {area:.0%} of frame (threshold 50%)"

    return True, None


def refine_shard_with_codex(
    shard_index: int,
    shard: list[tuple[str, str]],
    anthropic_key: str,
) -> dict:
    """
    Call Anthropic API to refine labels for one shard of frames.

    Args:
        shard_index: shard number for logging
        shard: list of (frame_path, label_path) tuples
        anthropic_key: Anthropic API key

    Returns:
        Parsed refinement result dict
    """
    import anthropic
    import base64

    client = anthropic.Anthropic(api_key=anthropic_key)

    shard_data = []
    for frame_path, label_path in shard:
        labels = read_label_file(label_path)
        with open(frame_path, "rb") as f:
            img_b64 = base64.standard_b64encode(f.read()).decode("utf-8")
        shard_data.append({
            "frame_path": frame_path,
            "current_labels": labels,
            "image_b64": img_b64,
        })

    messages_content = [
        {
            "type": "text",
            "text": f"Review and refine labels for shard {shard_index} ({len(shard)} frames).\n\n"
                    f"Current pseudo-labels per frame:\n{json.dumps([{'frame': d['frame_path'], 'labels': d['current_labels']} for d in shard_data], indent=2)}\n\n"
                    f"Return refined labels as specified in the tennis labeling policy."
        }
    ]

    for item in shard_data[:4]:
        messages_content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": item["image_b64"],
            }
        })

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=4096,
        system=TENNIS_LABELING_POLICY,
        messages=[{"role": "user", "content": messages_content}],
    )

    response_text = response.content[0].text

    try:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(response_text[start:end])
    except json.JSONDecodeError:
        pass

    logger.warning(f"Shard {shard_index}: Could not parse Codex response as JSON, returning pseudo-labels unchanged")
    return {
        "frames": [{"frame_path": fp, "labels": read_label_file(lp), "notes": "parse_failed"} for fp, lp in shard],
        "shard_valid": True,
        "rejection_reason": None,
    }


def process_shard(
    shard_index: int,
    shard: list[tuple[str, str]],
    job_id: str,
    output_dir: Path,
    anthropic_key: str,
    max_retries: int = MAX_RETRIES,
) -> tuple[bool, list[str]]:
    """
    Process one shard with retry logic.
    Returns (success, list of written label paths).
    """
    for attempt in range(max_retries):
        try:
            result = refine_shard_with_codex(shard_index, shard, anthropic_key)
            valid, reason = is_shard_valid(result)

            if not valid:
                logger.warning(
                    f"Shard {shard_index} attempt {attempt + 1}/{max_retries} failed QC: {reason}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                continue

            written_paths = []
            for frame_data in result.get("frames", []):
                frame_path = frame_data["frame_path"]
                labels = frame_data.get("labels", [])
                frame_stem = Path(frame_path).stem
                label_path = output_dir / f"{frame_stem}.txt"
                write_refined_label(label_path, labels)
                written_paths.append(str(label_path))

            logger.info(f"Shard {shard_index}: refined {len(written_paths)} frames")
            return True, written_paths

        except Exception as e:
            logger.error(f"Shard {shard_index} attempt {attempt + 1}/{max_retries} error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    logger.error(f"Shard {shard_index}: all {max_retries} attempts failed — using pseudo-labels as fallback")
    fallback_paths = []
    for frame_path, label_path in shard:
        frame_stem = Path(frame_path).stem
        out_path = output_dir / f"{frame_stem}.txt"
        labels = read_label_file(label_path)
        write_refined_label(out_path, labels)
        fallback_paths.append(str(out_path))
    return False, fallback_paths


def apply_coach_feedback(
    refined_label_dir: Path,
    coach_feedback: list[dict],
    checkpoint_frames: list[dict],
) -> None:
    """
    Incorporate coach feedback by removing rejected frames from training set.
    Rejected frames get their label files cleared.
    """
    rejected_indices = {
        f["frame_index"] for f in coach_feedback if f.get("action") == "reject"
    }
    for frame_data in checkpoint_frames:
        if frame_data.get("frame_index") in rejected_indices:
            frame_stem = Path(frame_data["frame_path"]).stem
            label_path = refined_label_dir / f"{frame_stem}.txt"
            if label_path.exists():
                label_path.write_text("")
                logger.info(f"Cleared labels for coach-rejected frame: {frame_stem}")


def run(
    job_id: str,
    frame_paths: list[str],
    pseudo_label_paths: list[str],
    coach_feedback: list[dict],
    checkpoint_frames: list[dict],
    config: dict,
) -> dict:
    """
    Refine pseudo-labels using parallel Codex subagents with tennis labeling policy.

    Returns:
        dict with 'refined_label_dir', 'label_paths', 'failed_shards'
    """
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    shard_size = config.get("shard_size", 25)
    max_workers = config.get("label_workers", 4)

    output_dir = REFINED_LABELS_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = list(zip(frame_paths, pseudo_label_paths))
    shards = [pairs[i:i + shard_size] for i in range(0, len(pairs), shard_size)]
    logger.info(f"Processing {len(pairs)} frames in {len(shards)} shards (size={shard_size}, workers={max_workers})")

    all_label_paths = []
    failed_shards = []

    if not anthropic_key:
        logger.warning("ANTHROPIC_API_KEY not set — copying pseudo-labels without refinement")
        for frame_path, label_path in pairs:
            frame_stem = Path(frame_path).stem
            out_path = output_dir / f"{frame_stem}.txt"
            labels = read_label_file(label_path)
            write_refined_label(out_path, labels)
            all_label_paths.append(str(out_path))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_shard, i, shard, job_id, output_dir, anthropic_key): i
                for i, shard in enumerate(shards)
            }
            for future in as_completed(futures):
                shard_idx = futures[future]
                success, paths = future.result()
                all_label_paths.extend(paths)
                if not success:
                    failed_shards.append(shard_idx)

    if coach_feedback and checkpoint_frames:
        apply_coach_feedback(output_dir, coach_feedback, checkpoint_frames)

    return {
        "refined_label_dir": str(output_dir),
        "label_paths": all_label_paths,
        "failed_shards": failed_shards,
        "total_refined": len(all_label_paths),
    }
