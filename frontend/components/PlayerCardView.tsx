"use client";

import type { PlayerCard, AnalyticsData } from "@/lib/types";

interface Props {
  playerACard: PlayerCard | null;
  playerBCard: PlayerCard | null;
  analytics: AnalyticsData | null;
}

// ── SVG Donut chart ────────────────────────────────────────────────────────────
const DONUT_COLORS = ["#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#a855f7", "#14b8a6"];

function ShotDonut({ counts }: { counts: Record<string, number> }) {
  const entries = Object.entries(counts)
    .sort(([, a], [, b]) => b - a)
    .filter(([, v]) => v > 0);
  const total = entries.reduce((s, [, v]) => s + v, 0);
  if (total === 0) return null;

  const R = 28;
  const stroke = 10;
  const cx = 38;
  const cy = 38;
  const circumference = 2 * Math.PI * R;

  let offset = 0;
  const segments = entries.map(([label, value], i) => {
    const frac = value / total;
    const dash = frac * circumference;
    const gap = circumference - dash;
    const rotate = (offset / total) * 360 - 90;
    offset += value;
    return { label, value, frac, dash, gap, rotate, color: DONUT_COLORS[i % DONUT_COLORS.length] };
  });

  return (
    <div className="flex items-center gap-3">
      <svg width={cx * 2} height={cy * 2} className="shrink-0">
        {/* background ring */}
        <circle cx={cx} cy={cy} r={R} fill="none" stroke="#27272a" strokeWidth={stroke} />
        {segments.map((seg, i) => (
          <circle
            key={i}
            cx={cx} cy={cy} r={R}
            fill="none"
            stroke={seg.color}
            strokeWidth={stroke}
            strokeDasharray={`${seg.dash} ${seg.gap}`}
            strokeDashoffset={0}
            transform={`rotate(${seg.rotate} ${cx} ${cy})`}
          />
        ))}
        {/* center label */}
        <text x={cx} y={cy - 3} textAnchor="middle" fontSize="9" fill="#a1a1aa" fontFamily="helvetica">shots</text>
        <text x={cx} y={cy + 7} textAnchor="middle" fontSize="11" fontWeight="bold" fill="#ffffff" fontFamily="helvetica">{total}</text>
      </svg>
      {/* Legend */}
      <div className="flex flex-col gap-0.5">
        {segments.map((seg) => (
          <div key={seg.label} className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full shrink-0" style={{ background: seg.color }} />
            <span className="text-xs text-zinc-400">
              {seg.label.charAt(0).toUpperCase() + seg.label.slice(1)}{" "}
              <span className="text-zinc-300 font-medium">{Math.round(seg.frac * 100)}%</span>
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Single player card ─────────────────────────────────────────────────────────
function SinglePlayerCard({
  card,
  analytics,
  playerKey,
}: {
  card: PlayerCard;
  analytics: AnalyticsData | null;
  playerKey: "player_a" | "player_b";
}) {
  const isA = playerKey === "player_a";
  const name = isA ? "Player A" : "Player B";
  const accent = isA ? "border-blue-500/30" : "border-orange-500/30";
  const nameColor = isA ? "text-blue-400" : "text-orange-400";
  const pa = analytics ? (isA ? analytics.player_a : analytics.player_b) : null;
  const tendencies = card.card.tendencies.slice(0, 3);

  return (
    <div className={`rounded-xl bg-zinc-900 border ${accent} p-5 space-y-4`}>
      {/* Header */}
      <div className="flex items-baseline gap-2">
        <span className={`text-base font-bold ${nameColor}`}>{name}</span>
        {pa && (
          <span className="text-xs text-zinc-500">
            {pa.total_shots} shots &middot; {Math.round(pa.avg_shot_speed_m_s * 3.6)} km/h avg
            {(pa.points_won + pa.points_lost) > 0 && (
              <> &middot; {pa.points_won}W / {pa.points_lost}L</>
            )}
          </span>
        )}
      </div>

      {/* Shot donut */}
      {pa && Object.keys(pa.shot_type_counts).length > 0 && (
        <div className="space-y-1.5">
          <h5 className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider">Shot Distribution</h5>
          <ShotDonut counts={pa.shot_type_counts} />
        </div>
      )}

      {/* Top 3 tendencies as bullets */}
      {tendencies.length > 0 && (
        <div className="space-y-1.5">
          <h5 className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider">Key Tendencies</h5>
          <ul className="space-y-1">
            {tendencies.map((t, i) => (
              <li key={i} className="flex items-start gap-2 text-xs text-zinc-300">
                <span className={`mt-1.5 w-1 h-1 rounded-full shrink-0 ${isA ? "bg-blue-400" : "bg-orange-400"}`} />
                {t}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Serve summary (one line) */}
      {card.card.serve_summary && (
        <div className="space-y-0.5">
          <h5 className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider">Serve</h5>
          <p className="text-xs text-zinc-400 leading-relaxed">{card.card.serve_summary}</p>
        </div>
      )}
    </div>
  );
}

// ── Export ─────────────────────────────────────────────────────────────────────
export default function PlayerCardView({ playerACard, playerBCard, analytics }: Props) {
  if (!playerACard && !playerBCard) return null;
  return (
    <div className="space-y-3">
      <h3 className="text-base font-semibold text-white px-1">Player Cards</h3>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {playerACard && (
          <SinglePlayerCard card={playerACard} analytics={analytics} playerKey="player_a" />
        )}
        {playerBCard && (
          <SinglePlayerCard card={playerBCard} analytics={analytics} playerKey="player_b" />
        )}
      </div>
    </div>
  );
}
