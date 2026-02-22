"use client";

import type { DetectedPoint } from "@/lib/types";

interface Props {
  points: DetectedPoint[];
  currentTime: number;
  onSeekToPoint: (sec: number) => void;
}

const END_REASON_COLORS: Record<string, string> = {
  OUT: "bg-red-500/20 text-red-400 border-red-500/30",
  DOUBLE_BOUNCE: "bg-orange-500/20 text-orange-400 border-orange-500/30",
  NET: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
  BALL_LOST: "bg-zinc-500/20 text-zinc-400 border-zinc-500/30",
};

export default function PointTimeline({ points, currentTime, onSeekToPoint }: Props) {
  if (!points.length) {
    return (
      <div className="rounded-xl bg-zinc-900 border border-zinc-800 p-6 text-center">
        <p className="text-zinc-500 text-sm">No points detected in this segment.</p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold text-zinc-300 px-1">Point Timeline</h3>
      <div className="space-y-1.5">
        {points.map((pt) => {
          const isActive = currentTime >= pt.start_sec && currentTime <= pt.end_sec;
          const duration = pt.end_sec - pt.start_sec;
          const colorClass = END_REASON_COLORS[pt.end_reason] ?? END_REASON_COLORS.BALL_LOST;

          return (
            <button
              key={pt.point_idx}
              onClick={() => onSeekToPoint(pt.start_sec)}
              className={`w-full text-left rounded-lg border px-3 py-2.5 transition-all ${
                isActive
                  ? "bg-green-500/10 border-green-500/40 ring-1 ring-green-500/20"
                  : "bg-zinc-900 border-zinc-800 hover:border-zinc-700"
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold text-zinc-300">P{pt.point_idx}</span>
                  <span className={`text-[10px] px-1.5 py-0.5 rounded border ${colorClass}`}>
                    {pt.end_reason}
                  </span>
                  {pt.serve_fault_type && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-red-500/10 text-red-400 border border-red-500/20">
                      {pt.serve_fault_type} fault
                    </span>
                  )}
                </div>
                <span className="text-[10px] text-zinc-500 font-mono">
                  {pt.start_sec.toFixed(1)}s - {pt.end_sec.toFixed(1)}s
                </span>
              </div>
              <div className="flex items-center gap-3 mt-1 text-[11px] text-zinc-500">
                <span>{pt.rally_hit_count} hits</span>
                <span>{pt.bounce_count} bounces</span>
                <span>{duration.toFixed(1)}s</span>
                {pt.serve_zone && <span className="text-zinc-400">serve: {pt.serve_zone.replace("_", " ")}</span>}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
