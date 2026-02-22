"use client";

import type { CoachingCard } from "@/lib/types";

interface Props {
  cards: CoachingCard[];
  onSeekToPoint?: (sec: number) => void;
}

const REASON_ICON: Record<string, string> = {
  OUT: "O",
  DOUBLE_BOUNCE: "D",
  NET: "N",
  BALL_LOST: "?",
};

export default function CoachingCards({ cards, onSeekToPoint }: Props) {
  if (!cards.length) return null;

  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold text-zinc-300 px-1">Coaching Cards</h3>
      <div className="space-y-3">
        {cards.map((card) => (
          <div
            key={card.point_idx}
            className="rounded-xl bg-zinc-900 border border-zinc-800 p-4 space-y-3"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="w-7 h-7 rounded-lg bg-zinc-800 flex items-center justify-center text-xs font-bold text-zinc-300">
                  {REASON_ICON[card.end_reason] ?? "?"}
                </span>
                <div>
                  <span className="text-sm font-semibold text-white">Point {card.point_idx}</span>
                  <span className="text-xs text-zinc-500 ml-2">
                    {card.rally_hit_count} hits &middot; {(card.end_sec - card.start_sec).toFixed(1)}s
                  </span>
                </div>
              </div>
              {onSeekToPoint && (
                <button
                  onClick={() => onSeekToPoint(card.start_sec)}
                  className="text-[10px] px-2 py-1 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-400 transition-colors"
                >
                  Watch
                </button>
              )}
            </div>

            <p className="text-sm text-zinc-300 leading-relaxed">{card.summary}</p>

            <div className="bg-green-500/5 border border-green-500/15 rounded-lg px-3 py-2">
              <p className="text-xs text-green-400 leading-relaxed">{card.suggestion}</p>
            </div>

            <div className="flex items-center gap-3 text-[10px] text-zinc-600">
              <span>Confidence: {(card.confidence * 100).toFixed(0)}%</span>
              {card.serve_zone && <span>Serve zone: {card.serve_zone.replace("_", " ")}</span>}
              {card.serve_fault_type && <span>Fault: {card.serve_fault_type}</span>}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
