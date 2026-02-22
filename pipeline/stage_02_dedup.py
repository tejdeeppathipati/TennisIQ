"""
Stage 02: pHash deduplication + Actian VectorAI DB indexing.

- Computes perceptual hash (pHash) for every extracted frame.
- Drops frames above similarity threshold (configurable, default hamming distance <= 10).
- Stores frame embeddings in Actian VectorAI DB for semantic diversity sampling.
- Falls back to stratified random sampling if Actian is unavailable (NFR-R06).
"""
import os
import logging
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

ACTIAN_VECTORAI_URL = os.getenv("ACTIAN_VECTORAI_URL")


def compute_phash(image_path: str, hash_size: int = 16) -> Optional[str]:
    """Compute perceptual hash of an image using imagehash library."""
    try:
        from PIL import Image
        import imagehash
        img = Image.open(image_path)
        return str(imagehash.phash(img, hash_size=hash_size))
    except Exception as e:
        logger.warning(f"pHash failed for {image_path}: {e}")
        return None


def hamming_distance(hash1: str, hash2: str) -> int:
    """Compute Hamming distance between two hex hash strings."""
    try:
        import imagehash
        h1 = imagehash.hex_to_hash(hash1)
        h2 = imagehash.hex_to_hash(hash2)
        return h1 - h2
    except Exception:
        return 999


def deduplicate_frames(frame_paths: list[str], threshold: int = 10) -> list[str]:
    """
    Deduplicate frames by pHash similarity.
    Keeps first frame in each cluster, drops near-duplicates above threshold.
    """
    unique_frames = []
    unique_hashes = []

    for frame_path in frame_paths:
        phash = compute_phash(frame_path)
        if phash is None:
            unique_frames.append(frame_path)
            continue

        is_duplicate = False
        for existing_hash in unique_hashes:
            if hamming_distance(phash, existing_hash) <= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_frames.append(frame_path)
            unique_hashes.append(phash)

    dropped = len(frame_paths) - len(unique_frames)
    logger.info(f"Deduplication: {len(frame_paths)} -> {len(unique_frames)} frames ({dropped} dropped)")
    return unique_frames


def extract_embedding(image_path: str) -> Optional[list[float]]:
    """
    Extract a simple visual embedding for diversity sampling.
    Uses flattened histogram as lightweight embedding (no neural net required).
    For production: swap with a proper feature extractor.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        img_resized = cv2.resize(img, (64, 64))
        hist_b = cv2.calcHist([img_resized], [0], None, [32], [0, 256]).flatten()
        hist_g = cv2.calcHist([img_resized], [1], None, [32], [0, 256]).flatten()
        hist_r = cv2.calcHist([img_resized], [2], None, [32], [0, 256]).flatten()
        embedding = np.concatenate([hist_b, hist_g, hist_r])
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.tolist()
    except Exception as e:
        logger.warning(f"Embedding extraction failed for {image_path}: {e}")
        return None


def index_in_actian(job_id: str, frame_paths: list[str], embeddings: list[list[float]]) -> bool:
    """
    Store frame embeddings in Actian VectorAI DB.
    Returns True on success, False on any failure (caller falls back to random sampling).
    """
    if not ACTIAN_VECTORAI_URL:
        logger.info("ACTIAN_VECTORAI_URL not set — skipping vector DB indexing.")
        return False
    try:
        import pyodbc
        conn = pyodbc.connect(ACTIAN_VECTORAI_URL)
        cursor = conn.cursor()
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS frame_embeddings_{job_id.replace('-', '_')} (
                frame_index INTEGER PRIMARY KEY,
                frame_path VARCHAR(1024),
                embedding VARBINARY(MAX)
            )
        """)
        import struct
        for i, (path, emb) in enumerate(zip(frame_paths, embeddings)):
            if emb is not None:
                blob = struct.pack(f"{len(emb)}f", *emb)
                cursor.execute(
                    f"INSERT INTO frame_embeddings_{job_id.replace('-', '_')} VALUES (?, ?, ?)",
                    (i, path, blob)
                )
        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"Indexed {len(frame_paths)} frame embeddings in Actian VectorAI DB")
        return True
    except Exception as e:
        logger.warning(f"Actian VectorAI DB indexing failed: {e}. Falling back to random sampling.")
        return False


def select_diverse_frames(
    frame_paths: list[str],
    embeddings: list[Optional[list[float]]],
    n: int = 24,
    job_id: str = "",
    actian_available: bool = False,
) -> list[str]:
    """
    Select N maximally diverse frames for coach checkpoint review.
    Uses greedy max-min distance selection over embeddings.
    Falls back to stratified random sampling if embeddings unavailable.
    """
    if len(frame_paths) <= n:
        return frame_paths

    valid = [(p, e) for p, e in zip(frame_paths, embeddings) if e is not None]

    if len(valid) < n:
        logger.warning("Not enough valid embeddings — using stratified random sampling.")
        step = len(frame_paths) // n
        return [frame_paths[i * step] for i in range(n)]

    paths = [p for p, _ in valid]
    embs = np.array([e for _, e in valid], dtype=np.float32)

    selected_indices = [0]
    for _ in range(n - 1):
        selected_embs = embs[selected_indices]
        dists = np.min(
            np.linalg.norm(embs[:, None] - selected_embs[None, :], axis=2),
            axis=1
        )
        dists[selected_indices] = -1
        next_idx = int(np.argmax(dists))
        selected_indices.append(next_idx)

    return [paths[i] for i in sorted(selected_indices)]


def run(job_id: str, frame_paths: list[str], config: dict) -> dict:
    """
    Deduplicate frames and index embeddings for diversity sampling.

    Returns:
        dict with 'unique_frames', 'checkpoint_candidates', 'actian_available'
    """
    threshold = config.get("phash_threshold", 10)
    n_checkpoint = config.get("checkpoint_frames", 24)

    logger.info(f"Starting deduplication of {len(frame_paths)} frames")
    unique_frames = deduplicate_frames(frame_paths, threshold=threshold)

    logger.info("Extracting embeddings for diversity sampling")
    embeddings = [extract_embedding(p) for p in unique_frames]

    actian_available = index_in_actian(job_id, unique_frames, embeddings)

    checkpoint_candidates = select_diverse_frames(
        unique_frames, embeddings, n=n_checkpoint,
        job_id=job_id, actian_available=actian_available
    )

    return {
        "unique_frames": unique_frames,
        "unique_count": len(unique_frames),
        "checkpoint_candidates": checkpoint_candidates,
        "actian_available": actian_available,
    }
