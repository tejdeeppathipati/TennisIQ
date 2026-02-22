"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { getCoachNotes } from "@/lib/api";
import type { DetectedPoint, CoachNote } from "@/lib/types";
import CoachNotes from "./CoachNotes";

interface Props {
  jobId: string;
  points: DetectedPoint[];
  clipBaseUrl: string;
  rawVideoUrl?: string;
  initialPointIdx?: number | null;
}

const END_REASON_COLORS: Record<string, string> = {
  OUT: "bg-red-500/15 text-red-400",
  NET: "bg-amber-500/15 text-amber-400",
  DOUBLE_BOUNCE: "bg-orange-500/15 text-orange-400",
  BALL_LOST: "bg-zinc-700 text-zinc-400",
  WINNER: "bg-green-500/15 text-green-400",
};

function formatDuration(sec: number): string {
  if (sec < 60) return `${sec.toFixed(1)}s`;
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function PointClipViewer({ jobId, points, clipBaseUrl, rawVideoUrl, initialPointIdx }: Props) {
  const [selectedIdx, setSelectedIdx] = useState<number>(initialPointIdx ?? (points.length > 0 ? points[0].point_idx : 0));
  const [notes, setNotes] = useState<CoachNote[]>([]);
  const [videoTime, setVideoTime] = useState(0);
  const [clipFailed, setClipFailed] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const loadNotes = useCallback(async () => {
    try {
      const resp = await getCoachNotes(jobId);
      setNotes(resp.notes);
    } catch {
      /* notes may not exist yet */
    }
  }, [jobId]);

  useEffect(() => {
    loadNotes();
  }, [loadNotes]);

  useEffect(() => {
    if (initialPointIdx != null && initialPointIdx >= 0) {
      setSelectedIdx(initialPointIdx);
    }
  }, [initialPointIdx]);

  useEffect(() => {
    setClipFailed(false);
    if (videoRef.current) {
      videoRef.current.load();
      videoRef.current.currentTime = 0;
      setVideoTime(0);
    }
  }, [selectedIdx]);

  useEffect(() => {
    const el = listRef.current?.querySelector(`[data-point-idx="${selectedIdx}"]`);
    el?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [selectedIdx]);

  const selectedPoint = points.find((p) => p.point_idx === selectedIdx);
  const clipUrl = `${clipBaseUrl}/point_${selectedIdx}.mp4`;

  const useRawFallback = clipFailed && rawVideoUrl;
  const videoSrc = useRawFallback ? rawVideoUrl : clipUrl;

  const handleVideoError = useCallback(() => {
    if (!clipFailed && rawVideoUrl) {
      setClipFailed(true);
    }
  }, [clipFailed, rawVideoUrl]);

  useEffect(() => {
    if (clipFailed && rawVideoUrl && videoRef.current && selectedPoint) {
      videoRef.current.addEventListener("loadedmetadata", function seekOnce() {
        if (videoRef.current) {
          videoRef.current.currentTime = selectedPoint.start_sec;
          videoRef.current.removeEventListener("loadedmetadata", seekOnce);
        }
      });
      videoRef.current.load();
    }
  }, [clipFailed, rawVideoUrl, selectedPoint]);

  return (
    <div className="space-y-3">
      <h3 className="text-lg font-semibold text-white px-1">Point Clips &amp; Coach Notes</h3>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Left: point list */}
        <div
          ref={listRef}
          className="lg:col-span-1 max-h-[520px] overflow-y-auto space-y-1.5 rounded-xl bg-zinc-900 border border-zinc-800 p-3"
        >
          {points.map((pt) => {
            const active = pt.point_idx === selectedIdx;
            const noteCount = notes.filter((n) => n.point_idx === pt.point_idx).length;
            return (
              <button
                key={pt.point_idx}
                data-point-idx={pt.point_idx}
                onClick={() => setSelectedIdx(pt.point_idx)}
                className={`w-full text-left rounded-lg px-3 py-2 transition-colors ${
                  active
                    ? "bg-green-600/15 border border-green-500/30"
                    : "bg-zinc-800/50 border border-transparent hover:bg-zinc-800"
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className={`text-sm font-medium ${active ? "text-green-400" : "text-zinc-300"}`}>
                    Point {pt.point_idx + 1}
                  </span>
                  <span
                    className={`text-[10px] px-1.5 py-0.5 rounded ${END_REASON_COLORS[pt.end_reason] ?? "bg-zinc-700 text-zinc-400"}`}
                  >
                    {pt.end_reason}
                  </span>
                </div>
                <div className="flex items-center gap-2 mt-0.5 text-xs text-zinc-500">
                  <span>{pt.rally_hit_count} hits</span>
                  <span>&middot;</span>
                  <span>{formatDuration(pt.end_sec - pt.start_sec)}</span>
                  {noteCount > 0 && (
                    <>
                      <span>&middot;</span>
                      <span className="text-green-500">{noteCount} note{noteCount > 1 ? "s" : ""}</span>
                    </>
                  )}
                </div>
              </button>
            );
          })}
          {points.length === 0 && (
            <p className="text-sm text-zinc-600 text-center py-4">No points detected.</p>
          )}
        </div>

        {/* Right: video + notes */}
        <div className="lg:col-span-2 space-y-4">
          <div className="rounded-xl overflow-hidden bg-black relative">
            <video
              ref={videoRef}
              src={videoSrc}
              className="w-full aspect-video"
              controls
              playsInline
              onTimeUpdate={() => {
                if (videoRef.current) setVideoTime(videoRef.current.currentTime);
              }}
              onError={handleVideoError}
            >
              <track kind="captions" />
            </video>
            {useRawFallback && (
              <div className="absolute top-2 right-2 bg-zinc-900/80 text-zinc-400 text-[10px] px-2 py-1 rounded">
                Full video &middot; seeked to point start
              </div>
            )}
          </div>

          {selectedPoint && (
            <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-4 space-y-1 text-sm">
              <div className="flex items-center gap-3 text-zinc-400">
                <span>Point {selectedPoint.point_idx + 1}</span>
                <span
                  className={`text-[10px] px-1.5 py-0.5 rounded ${END_REASON_COLORS[selectedPoint.end_reason] ?? "bg-zinc-700 text-zinc-400"}`}
                >
                  {selectedPoint.end_reason}
                </span>
                <span>{selectedPoint.rally_hit_count} hits</span>
                <span>{formatDuration(selectedPoint.end_sec - selectedPoint.start_sec)}</span>
                {selectedPoint.serve_zone && <span>Serve: {selectedPoint.serve_zone}</span>}
              </div>
            </div>
          )}

          <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-4">
            <CoachNotes
              jobId={jobId}
              pointIdx={selectedIdx}
              notes={notes}
              currentVideoTime={videoTime}
              onNotesChanged={loadNotes}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
