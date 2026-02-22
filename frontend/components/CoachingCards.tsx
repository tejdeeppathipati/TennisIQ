"use client";

import { useState } from "react";
import type { CoachingCard } from "@/lib/types";

interface Props {
  cards: CoachingCard[];
  onSeekToPoint?: (sec: number) => void;
}

const REASON_BADGE: Record<string, { label: string; color: string }> = {
  OUT: { label: "OUT", color: "bg-red-500/15 text-red-400 border-red-500/20" },
  NET: { label: "NET", color: "bg-amber-500/15 text-amber-400 border-amber-500/20" },
  NET_FAULT: { label: "NET FAULT", color: "bg-amber-500/15 text-amber-400 border-amber-500/20" },
  OUT_LONG: { label: "LONG", color: "bg-red-500/15 text-red-400 border-red-500/20" },
  OUT_WIDE: { label: "WIDE", color: "bg-red-500/15 text-red-400 border-red-500/20" },
  DOUBLE_BOUNCE: { label: "DBL BOUNCE", color: "bg-orange-500/15 text-orange-400 border-orange-500/20" },
  BALL_LOST: { label: "LOST", color: "bg-zinc-500/15 text-zinc-400 border-zinc-500/20" },
  IN: { label: "IN PLAY", color: "bg-emerald-500/15 text-emerald-400 border-emerald-500/20" },
};

const SHOT_TYPE_COLOR: Record<string, string> = {
  forehand: "text-blue-400",
  backhand: "text-amber-400",
  serve: "text-emerald-400",
  neutral: "text-zinc-500",
  unknown: "text-zinc-600",
};

function KeyShots({ shots, expanded }: {
  shots: NonNullable<CoachingCard["shot_sequence"]>;
  expanded: boolean;
}) {
  if (shots.length === 0) return null;

  // Key shots: first 2 + last 4 (or all if ≤ 8)
  let display = shots;
  let truncated = false;
  if (!expanded && shots.length > 8) {
    const first = shots.slice(0, 2);
    const last = shots.slice(-4);
    display = [...first, ...last];
    truncated = true;
  }

  return (
    <div className="flex flex-wrap items-center gap-1">
      {display.map((s, i) => {
        const isA = s.owner === "player_a";
        const typeColor = SHOT_TYPE_COLOR[s.shot_type] || "text-zinc-500";
        const showGap = truncated && i === 2;

        return (
          <div key={i} className="flex items-center gap-1">
            {showGap && (
              <span className="text-[9px] text-zinc-600 px-1">···</span>
            )}
            <div
              className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] border ${
                isA
                  ? "bg-blue-500/5 border-blue-500/15"
                  : "bg-orange-500/5 border-orange-500/15"
              }`}
            >
              <span className={`font-bold ${isA ? "text-blue-400" : "text-orange-400"}`}>
                {s.owner_short}
              </span>
              <span className={typeColor}>{s.shot_type}</span>
              {s.direction !== "unknown" && (
                <span className="text-zinc-600">{s.direction.replace(/_/g, " ")}</span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function SingleCard({ card, onSeekToPoint }: {
  card: CoachingCard;
  onSeekToPoint?: (sec: number) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const badge = REASON_BADGE[card.end_reason] ?? REASON_BADGE.BALL_LOST;
  const duration = (card.end_sec - card.start_sec).toFixed(1);
  const hasShots = card.shot_sequence && card.shot_sequence.length > 0;

  return (
    <div className="rounded-xl bg-zinc-900 border border-zinc-800 p-4 space-y-2.5">
      {/* Row 1: outcome badge + rally stats + watch button */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <span className={`px-2 py-0.5 rounded border text-[10px] font-bold ${badge.color}`}>
            {badge.label}
          </span>
          <span className="text-sm font-semibold text-white">Point {card.point_idx + 1}</span>
          <span className="text-xs text-zinc-500">
            {card.rally_hit_count} hits &middot; {duration}s
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          {hasShots && (
            <button
              onClick={() => setExpanded((v) => !v)}
              className="text-[10px] px-2 py-1 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-400 transition-colors"
            >
              {expanded ? "Less" : "Details"}
            </button>
          )}
          {onSeekToPoint && (
            <button
              onClick={() => onSeekToPoint(card.start_sec)}
              className="text-[10px] px-2 py-1 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-400 transition-colors"
            >
              Watch
            </button>
          )}
        </div>
      </div>

      {/* Row 2: 1-sentence coaching suggestion (always visible) */}
      <p className="text-sm text-zinc-300 leading-relaxed">{card.suggestion}</p>

      {/* Row 3: Pattern context — key pattern in 1 line */}
      {card.pattern_context && (
        <p className="text-xs text-amber-400/80">{card.pattern_context}</p>
      )}

      {/* Expandable details */}
      {expanded && (
        <div className="space-y-2.5 pt-1 border-t border-zinc-800">
          {/* Summary narrative */}
          {card.summary && card.summary !== card.suggestion && (
            <p className="text-xs text-zinc-400 leading-relaxed">{card.summary}</p>
          )}

          {/* Key shot sequence */}
          {hasShots && (
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-zinc-500 uppercase font-medium">
                  Key Shots
                </span>
              </div>
              <KeyShots shots={card.shot_sequence!} expanded={expanded} />
            </div>
          )}

          {/* Meta info */}
          <div className="flex items-center gap-3 text-[10px] text-zinc-600">
            <span>Confidence: {(card.confidence * 100).toFixed(0)}%</span>
            {card.serve_zone && <span>Serve: {card.serve_zone.replace(/_/g, " ")}</span>}
            {card.serve_fault_type && <span>Fault: {card.serve_fault_type}</span>}
          </div>
        </div>
      )}
    </div>
  );
}

export default function CoachingCards({ cards, onSeekToPoint }: Props) {
  if (!cards.length) return null;

  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold text-zinc-300 px-1">
        Coaching Cards
        <span className="text-zinc-500 font-normal ml-2">{cards.length} points</span>
      </h3>
      <div className="space-y-2">
        {cards.map((card) => (
          <SingleCard key={card.point_idx} card={card} onSeekToPoint={onSeekToPoint} />
        ))}
      </div>
    </div>
  );
}
