"use client";

import type { PlayerCard } from "@/lib/types";

interface Props {
  playerACard: PlayerCard | null;
  playerBCard: PlayerCard | null;
}

function SeverityBar({ severity }: { severity: number }) {
  const width = Math.max(severity * 100, 5);
  const color = severity > 0.6 ? "bg-red-500" : severity > 0.3 ? "bg-amber-500" : "bg-yellow-500";

  return (
    <div className="w-full h-1.5 bg-zinc-800 rounded-full overflow-hidden">
      <div
        className={`h-full rounded-full ${color} transition-all`}
        style={{ width: `${width}%` }}
      />
    </div>
  );
}

function PlayerWeaknesses({ card, isA }: { card: PlayerCard; isA: boolean }) {
  const name = isA ? "Player A" : "Player B";
  const weaknesses = card.weaknesses.weaknesses;

  if (weaknesses.length === 0) {
    return (
      <div className="rounded-xl bg-zinc-900 border border-zinc-800 p-4">
        <div className="flex items-center gap-2 mb-2">
          <span className={`w-6 h-6 rounded-md ${isA ? "bg-blue-500/20" : "bg-orange-500/20"} flex items-center justify-center text-xs font-bold ${isA ? "text-blue-400" : "text-orange-400"}`}>
            {isA ? "A" : "B"}
          </span>
          <span className="text-sm font-semibold text-white">{name}</span>
        </div>
        <p className="text-sm text-zinc-500">Insufficient data to identify weaknesses.</p>
      </div>
    );
  }

  return (
    <div className="rounded-xl bg-zinc-900 border border-zinc-800 p-4 space-y-3">
      <div className="flex items-center gap-2">
        <span className={`w-6 h-6 rounded-md ${isA ? "bg-blue-500/20" : "bg-orange-500/20"} flex items-center justify-center text-xs font-bold ${isA ? "text-blue-400" : "text-orange-400"}`}>
          {isA ? "A" : "B"}
        </span>
        <span className="text-sm font-semibold text-white">{name} — Exploitable Patterns</span>
      </div>

      {weaknesses.map((w, i) => (
        <div key={i} className="space-y-2 bg-zinc-800/40 rounded-lg p-3">
          <div className="flex items-start justify-between gap-2">
            <p className="text-sm text-zinc-200 leading-relaxed">{w.description}</p>
            {w.points_cost > 0 && (
              <span className="shrink-0 text-[10px] px-2 py-0.5 rounded bg-red-500/15 text-red-400 font-medium">
                -{w.points_cost} pts
              </span>
            )}
          </div>
          <SeverityBar severity={w.severity} />
          <p className="text-xs text-zinc-500">{w.data_point}</p>
        </div>
      ))}
    </div>
  );
}

export default function WeaknessReport({ playerACard, playerBCard }: Props) {
  const hasA = playerACard && playerACard.weaknesses.weaknesses.length > 0;
  const hasB = playerBCard && playerBCard.weaknesses.weaknesses.length > 0;

  if (!hasA && !hasB) return null;

  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold text-zinc-300 px-1">Weakness Reports</h3>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {playerACard && <PlayerWeaknesses card={playerACard} isA />}
        {playerBCard && <PlayerWeaknesses card={playerBCard} isA={false} />}
      </div>
    </div>
  );
}
