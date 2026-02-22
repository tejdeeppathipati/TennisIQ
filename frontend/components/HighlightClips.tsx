"use client";

import type { DetectedPoint } from "@/lib/types";

interface Props {
  points: DetectedPoint[];
  clipBaseUrl: string;
}

export default function HighlightClips({ points, clipBaseUrl }: Props) {
  if (!points.length) return null;

  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold text-zinc-300 px-1">Point Clips</h3>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {points.map((pt) => {
          const src = `${clipBaseUrl}/point_${pt.point_idx}.mp4`;
          return (
            <div key={pt.point_idx} className="rounded-xl bg-zinc-900 border border-zinc-800 overflow-hidden">
              <video
                src={src}
                controls
                muted
                playsInline
                className="w-full aspect-video bg-black"
              />
              <div className="px-3 py-2 flex items-center justify-between">
                <span className="text-xs font-medium text-zinc-300">
                  Point {pt.point_idx}
                </span>
                <span className="text-[10px] text-zinc-500">
                  {pt.rally_hit_count} hits &middot; {pt.end_reason}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
