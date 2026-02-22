"use client";

import type { CoachingCard, DetectedPoint } from "@/lib/types";

type Props = {
  points: DetectedPoint[];
  coachingCards: CoachingCard[];
  clipBaseUrl: string;
  onSeekToPoint: (sec: number) => void;
};

const END_COLORS: Record<string, string> = {
  OUT: "bg-red-500/15 text-red-300 border-red-500/30",
  NET: "bg-yellow-500/15 text-yellow-300 border-yellow-500/30",
  DOUBLE_BOUNCE: "bg-orange-500/15 text-orange-300 border-orange-500/30",
  BALL_LOST: "bg-zinc-500/15 text-zinc-300 border-zinc-500/30",
};

function badge(text: string, cls: string) {
  return (
    <span className={`text-[10px] px-1.5 py-0.5 rounded border ${cls}`}>
      {text}
    </span>
  );
}

function findCard(pointIdx: number, cards: CoachingCard[]): CoachingCard | undefined {
  return cards.find((c) => c.point_idx === pointIdx);
}

function pointVideoUrl(pointIdx: number, base: string) {
  return `${base}/point_${pointIdx}.mp4`;
}

export default function PointDeck({ points, coachingCards, clipBaseUrl, onSeekToPoint }: Props) {
  if (!points.length) return null;

  const groups: Record<string, DetectedPoint[]> = {};
  points.forEach((p) => {
    const key = p.end_reason || "UNKNOWN";
    groups[key] = groups[key] || [];
    groups[key].push(p);
  });

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between px-1">
        <h3 className="text-sm font-semibold text-zinc-300">Points</h3>
        <span className="text-[11px] text-zinc-500">{points.length} total</span>
      </div>

      {Object.entries(groups).map(([reason, pts]) => (
        <div key={reason} className="space-y-3">
          <div className="text-xs uppercase tracking-wide text-zinc-500 px-1">
            {reason.replace("_", " ")} ({pts.length})
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
            {pts.map((pt) => {
              const dur = (pt.end_sec - pt.start_sec).toFixed(1);
              const confPct = Math.round(pt.confidence * 100);
              const card = findCard(pt.point_idx, coachingCards);
              const clipUrl = pointVideoUrl(pt.point_idx, clipBaseUrl);
              return (
                <div
                  key={pt.point_idx}
                  className="rounded-xl bg-zinc-900 border border-zinc-800 hover:border-zinc-700 transition-colors overflow-hidden flex flex-col"
                >
                  <div className="p-3 space-y-2">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-bold text-white">P{pt.point_idx}</span>
                      {badge(reason.replace("_", " ").toLowerCase(), END_COLORS[reason] ?? END_COLORS.BALL_LOST)}
                      {pt.serve_fault_type && badge(`${pt.serve_fault_type} fault`, "bg-red-500/10 text-red-300 border-red-500/20")}
                      {badge(`${confPct}% conf`, confPct >= 70 ? "bg-green-500/10 text-green-300 border-green-500/20" : confPct >= 40 ? "bg-yellow-500/10 text-yellow-300 border-yellow-500/20" : "bg-red-500/10 text-red-300 border-red-500/20")}
                    </div>
                    <div className="flex flex-wrap gap-3 text-[11px] text-zinc-500">
                      <span>{pt.rally_hit_count} hits</span>
                      <span>{pt.bounce_count} bounces</span>
                      <span>{dur}s</span>
                      {pt.serve_zone && <span>serve: {pt.serve_zone.replace("_", " ")}</span>}
                      {pt.serve_player && <span>{pt.serve_player}</span>}
                    </div>
                    {card && (
                      <div className="text-xs text-zinc-300 leading-snug">
                        {card.summary}
                      </div>
                    )}
                    {card && (
                      <div className="bg-green-500/5 border border-green-500/15 rounded-lg px-3 py-2 text-[11px] text-green-400 leading-snug">
                        {card.suggestion}
                      </div>
                    )}
                  </div>
                  <div className="bg-black/40">
                    <video
                      className="w-full h-40 object-cover bg-black cursor-pointer"
                      muted
                      controls
                      onPlay={() => onSeekToPoint(pt.start_sec)}
                      src={clipUrl}
                    />
                    <div className="flex items-center justify-between px-3 py-2 text-[11px] text-zinc-500">
                      <span>Clip</span>
                      <button
                        onClick={() => onSeekToPoint(pt.start_sec)}
                        className="text-zinc-300 hover:text-white"
                      >
                        Jump
                      </button>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
