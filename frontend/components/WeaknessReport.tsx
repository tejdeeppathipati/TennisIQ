"use client";

import type { PlayerCard, DetectedPoint } from "@/lib/types";

interface Props {
  playerACard: PlayerCard | null;
  playerBCard: PlayerCard | null;
  points: DetectedPoint[];
  onEvidenceClick: (pointIdx: number) => void;
}

// ── Severity bar ───────────────────────────────────────────────────────────────
function SeverityBar({ severity }: { severity: number }) {
  const pct = Math.min(100, Math.max(0, Math.round(severity * 100)));
  const color = pct >= 70 ? "bg-red-500" : pct >= 40 ? "bg-amber-500" : "bg-yellow-400";
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 rounded-full bg-zinc-800">
        <div className={`h-1.5 rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-[10px] text-zinc-500 w-8 text-right">{pct}%</span>
    </div>
  );
}

// ── Evidence button (click-to-seek) ───────────────────────────────────────────
function EvidenceButton({
  point,
  idx,
  onClick,
}: {
  point: DetectedPoint;
  idx: number;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-zinc-800 hover:bg-zinc-700 
                 text-[10px] text-zinc-400 hover:text-white transition-colors cursor-pointer"
      title={`Jump to P${idx}: ${point.start_sec.toFixed(1)}s`}
    >
      <span className="text-zinc-500">&#9654;</span>
      P{idx}&nbsp;{point.start_sec.toFixed(1)}s
    </button>
  );
}

// ── Single-player weaknesses ───────────────────────────────────────────────────
function PlayerWeaknesses({
  card,
  points,
  playerKey,
  onEvidenceClick,
}: {
  card: PlayerCard;
  points: DetectedPoint[];
  playerKey: "player_a" | "player_b";
  onEvidenceClick: (pointIdx: number) => void;
}) {
  const isA = playerKey === "player_a";
  const name = isA ? "Player A" : "Player B";
  const accent = isA ? "border-blue-500/30" : "border-orange-500/30";
  const nameColor = isA ? "text-blue-400" : "text-orange-400";
  const weaknesses = card.weaknesses.weaknesses.slice(0, 3);

  // Bad points as evidence: OUT or BALL_LOST, short clip
  const badPoints = points
    .map((p, i) => ({ p, i }))
    .filter(({ p }) => p.end_reason === "OUT" || p.end_reason === "BALL_LOST")
    .slice(0, 6);

  if (weaknesses.length === 0) return null;

  return (
    <div className={`rounded-xl bg-zinc-900 border ${accent} p-5 space-y-5`}>
      <div className="flex items-baseline gap-2">
        <span className={`text-base font-bold ${nameColor}`}>{name}</span>
        <span className="text-xs text-zinc-500">Top weaknesses to exploit</span>
      </div>

      {weaknesses.map((w, wi) => {
        const evidencePoints = badPoints.slice(wi * 2, wi * 2 + 2);
        const rankColors = ["bg-red-500/20 text-red-400", "bg-amber-500/20 text-amber-400", "bg-yellow-500/20 text-yellow-400"];
        return (
          <div key={wi} className="space-y-1.5">
            {/* Rank + description */}
            <div className="flex items-start gap-2">
              <span className={`mt-0.5 text-[10px] font-bold rounded-full w-4 h-4 flex items-center justify-center shrink-0 ${rankColors[wi]}`}>
                {wi + 1}
              </span>
              <span className="text-sm font-medium text-zinc-200">{w.description}</span>
            </div>

            {/* Supporting data */}
            {w.data_point && (
              <p className="ml-6 text-xs text-zinc-500 leading-relaxed">{w.data_point}</p>
            )}

            {/* Point cost + severity */}
            <div className="ml-6 flex items-center gap-3 flex-wrap">
              <span className="text-xs text-zinc-500">
                Point cost:&nbsp;
                <span className="text-zinc-300 font-semibold">{w.points_cost}</span>
              </span>
            </div>
            <div className="ml-6">
              <SeverityBar severity={w.severity} />
            </div>

            {/* Clickable evidence clips */}
            {evidencePoints.length > 0 && (
              <div className="ml-6 flex flex-wrap gap-1.5 items-center pt-0.5">
                <span className="text-[10px] text-zinc-600">Watch:</span>
                {evidencePoints.map(({ p, i }) => (
                  <EvidenceButton key={i} point={p} idx={i} onClick={() => onEvidenceClick(i)} />
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ── Export ─────────────────────────────────────────────────────────────────────
export default function WeaknessReport({ playerACard, playerBCard, points, onEvidenceClick }: Props) {
  const hasA = (playerACard?.weaknesses.weaknesses.length ?? 0) > 0;
  const hasB = (playerBCard?.weaknesses.weaknesses.length ?? 0) > 0;
  if (!hasA && !hasB) return null;

  return (
    <div className="space-y-3">
      <h3 className="text-base font-semibold text-white px-1">Weakness Report</h3>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {hasA && playerACard && (
          <PlayerWeaknesses
            card={playerACard}
            points={points}
            playerKey="player_a"
            onEvidenceClick={onEvidenceClick}
          />
        )}
        {hasB && playerBCard && (
          <PlayerWeaknesses
            card={playerBCard}
            points={points}
            playerKey="player_b"
            onEvidenceClick={onEvidenceClick}
          />
        )}
      </div>
    </div>
  );
}
