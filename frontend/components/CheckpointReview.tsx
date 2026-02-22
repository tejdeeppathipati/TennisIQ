"use client";

import { useEffect, useState, useCallback } from "react";
import { getPointsForReview, submitPointFeedback, finalizeReview } from "@/lib/api";
import type { DetectedPoint } from "@/lib/types";

interface Props {
  jobId: string;
  onComplete: () => Promise<boolean>;
}

export default function CheckpointReview({ jobId, onComplete }: Props) {
  const [points, setPoints] = useState<DetectedPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState<number | null>(null);
  const [finalizing, setFinalizing] = useState(false);
  const [noteInputs, setNoteInputs] = useState<Record<number, string>>({});

  const fetchPoints = useCallback(async () => {
    try {
      const res = await getPointsForReview(jobId);
      setPoints(res.points);
    } catch {
      /* will retry on next poll */
    } finally {
      setLoading(false);
    }
  }, [jobId]);

  useEffect(() => {
    fetchPoints();
  }, [fetchPoints]);

  const handleFeedback = async (pointIdx: number, action: "confirm" | "flag") => {
    setSubmitting(pointIdx);
    try {
      await submitPointFeedback(jobId, pointIdx, action, noteInputs[pointIdx]);
      setPoints((prev) =>
        prev.map((p) =>
          p.point_idx === pointIdx ? { ...p, coach_action: action, coach_note: noteInputs[pointIdx] ?? null } : p,
        ),
      );
    } finally {
      setSubmitting(null);
    }
  };

  const handleFinalize = async () => {
    setFinalizing(true);
    try {
      await finalizeReview(jobId);
      await onComplete();
    } finally {
      setFinalizing(false);
    }
  };

  if (loading) {
    return (
      <div className="rounded-2xl bg-zinc-900 border border-zinc-800 p-8 text-center">
        <p className="text-zinc-500 text-sm">Loading detected points for review...</p>
      </div>
    );
  }

  const reviewedCount = points.filter((p) => p.coach_action).length;

  return (
    <div className="rounded-2xl bg-zinc-900 border border-yellow-500/20 p-6 space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold text-white">Coach Review</h2>
          <p className="text-xs text-zinc-500 mt-0.5">
            Review each detected point. Confirm good calls, flag bad ones.
          </p>
        </div>
        <div className="text-right">
          <span className="text-sm text-zinc-400">{reviewedCount}/{points.length} reviewed</span>
        </div>
      </div>

      <div className="space-y-3">
        {points.map((pt) => (
          <div
            key={pt.point_idx}
            className={`rounded-xl border p-4 space-y-3 ${
              pt.coach_action === "confirm"
                ? "bg-green-500/5 border-green-500/20"
                : pt.coach_action === "flag"
                ? "bg-red-500/5 border-red-500/20"
                : "bg-zinc-800/50 border-zinc-700"
            }`}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-sm font-bold text-zinc-200">Point {pt.point_idx}</span>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-700 text-zinc-300">
                  {pt.end_reason}
                </span>
              </div>
              <span className="text-[10px] text-zinc-500 font-mono">
                {pt.start_sec.toFixed(1)}s - {pt.end_sec.toFixed(1)}s
              </span>
            </div>

            <div className="flex items-center gap-3 text-xs text-zinc-500">
              <span>{pt.rally_hit_count} hits</span>
              <span>{pt.bounce_count} bounces</span>
              {pt.serve_zone && <span>serve: {pt.serve_zone.replace("_", " ")}</span>}
              {pt.serve_fault_type && <span className="text-red-400">{pt.serve_fault_type} fault</span>}
              <span>conf: {(pt.confidence * 100).toFixed(0)}%</span>
            </div>

            <input
              type="text"
              placeholder="Optional note..."
              value={noteInputs[pt.point_idx] ?? pt.coach_note ?? ""}
              onChange={(e) => setNoteInputs((prev) => ({ ...prev, [pt.point_idx]: e.target.value }))}
              className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-1.5 text-xs text-zinc-300 placeholder-zinc-600 focus:outline-none focus:border-zinc-500"
            />

            <div className="flex items-center gap-2">
              <button
                onClick={() => handleFeedback(pt.point_idx, "confirm")}
                disabled={submitting === pt.point_idx}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                  pt.coach_action === "confirm"
                    ? "bg-green-500 text-white"
                    : "bg-zinc-800 hover:bg-green-500/20 text-zinc-400 hover:text-green-400"
                }`}
              >
                Confirm
              </button>
              <button
                onClick={() => handleFeedback(pt.point_idx, "flag")}
                disabled={submitting === pt.point_idx}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                  pt.coach_action === "flag"
                    ? "bg-red-500 text-white"
                    : "bg-zinc-800 hover:bg-red-500/20 text-zinc-400 hover:text-red-400"
                }`}
              >
                Flag
              </button>
            </div>
          </div>
        ))}
      </div>

      <button
        onClick={handleFinalize}
        disabled={finalizing}
        className="w-full py-3 rounded-xl bg-green-600 hover:bg-green-500 text-white font-semibold text-sm transition-colors disabled:opacity-50"
      >
        {finalizing ? "Finalizing..." : "Finalize Review & Generate Output"}
      </button>
    </div>
  );
}
