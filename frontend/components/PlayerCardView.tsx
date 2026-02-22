"use client";

import type { PlayerCard, AnalyticsData, PatternItem } from "@/lib/types";

interface Props {
  playerACard: PlayerCard | null;
  playerBCard: PlayerCard | null;
  analytics: AnalyticsData | null;
}

function ShotDistPie({ counts, label }: { counts: Record<string, number>; label: string }) {
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  if (total === 0) return null;

  const colors: Record<string, string> = {
    forehand: "#3b82f6",
    backhand: "#f59e0b",
    serve: "#10b981",
    neutral: "#6b7280",
    unknown: "#374151",
  };

  let offset = 0;
  const segments = Object.entries(counts)
    .sort(([, a], [, b]) => b - a)
    .map(([type, count]) => {
      const pct = (count / total) * 100;
      const seg = { type, count, pct, offset, color: colors[type] || "#6b7280" };
      offset += pct;
      return seg;
    });

  return (
    <div className="flex items-center gap-4">
      <svg width="64" height="64" viewBox="0 0 36 36" className="shrink-0">
        {segments.map((seg) => (
          <circle
            key={seg.type}
            cx="18" cy="18" r="14"
            fill="none"
            stroke={seg.color}
            strokeWidth="4"
            strokeDasharray={`${seg.pct * 0.88} ${88 - seg.pct * 0.88}`}
            strokeDashoffset={`${-seg.offset * 0.88 + 22}`}
          />
        ))}
      </svg>
      <div className="space-y-1">
        {segments.map((seg) => (
          <div key={seg.type} className="flex items-center gap-2 text-xs">
            <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: seg.color }} />
            <span className="text-zinc-400 capitalize">{seg.type}</span>
            <span className="text-zinc-300 font-medium">{seg.count}</span>
            <span className="text-zinc-500">({seg.pct.toFixed(0)}%)</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function ShotDirectionBreakdown({ dirPcts }: { dirPcts: Record<string, Record<string, number>> }) {
  const entries = Object.entries(dirPcts).filter(([, dirs]) => Object.keys(dirs).length > 0);
  if (entries.length === 0) return null;

  const dirColors: Record<string, string> = {
    cross_court: "bg-cyan-500/20 text-cyan-400",
    down_the_line: "bg-purple-500/20 text-purple-400",
    middle: "bg-zinc-500/20 text-zinc-400",
  };

  return (
    <div className="space-y-2">
      {entries.map(([shotType, dirs]) => (
        <div key={shotType} className="space-y-1">
          <span className="text-[10px] text-zinc-500 uppercase font-medium capitalize">{shotType}</span>
          <div className="flex gap-1.5 flex-wrap">
            {Object.entries(dirs)
              .sort(([, a], [, b]) => b - a)
              .map(([dir, pct]) => (
                <span key={dir} className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${dirColors[dir] || "bg-zinc-800 text-zinc-400"}`}>
                  {dir.replace(/_/g, " ")} {pct.toFixed(0)}%
                </span>
              ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function ErrorRateSection({
  byShot,
  byRally,
}: {
  byShot: Record<string, number>;
  byRally: Record<string, number>;
}) {
  const hasShot = Object.keys(byShot).length > 0;
  const hasRally = Object.keys(byRally).length > 0;
  if (!hasShot && !hasRally) return null;

  return (
    <div className="space-y-2">
      {hasShot && (
        <div className="space-y-1">
          <span className="text-[10px] text-zinc-500 uppercase font-medium">Error rate by shot</span>
          <div className="flex gap-1.5 flex-wrap">
            {Object.entries(byShot).map(([type, rate]) => (
              <span key={type} className="text-[10px] px-2 py-0.5 rounded-full bg-red-500/15 text-red-400 font-medium capitalize">
                {type} {rate.toFixed(0)}%
              </span>
            ))}
          </div>
        </div>
      )}
      {hasRally && (
        <div className="space-y-1">
          <span className="text-[10px] text-zinc-500 uppercase font-medium">Error rate by rally length</span>
          <div className="flex gap-1.5 flex-wrap">
            {Object.entries(byRally).map(([bucket, rate]) => (
              <span key={bucket} className="text-[10px] px-2 py-0.5 rounded-full bg-red-500/15 text-red-400 font-medium">
                {bucket} shots: {rate.toFixed(0)}%
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function TopPatterns({ patterns }: { patterns: PatternItem[] }) {
  if (!patterns || patterns.length === 0) return null;
  const top = patterns.slice(0, 3);

  const shotColors: Record<string, string> = {
    forehand: "border-blue-500/30 bg-blue-500/5",
    backhand: "border-amber-500/30 bg-amber-500/5",
    serve: "border-emerald-500/30 bg-emerald-500/5",
    neutral: "border-zinc-700 bg-zinc-800/50",
  };

  return (
    <div className="space-y-1.5">
      {top.map((p) => (
        <div key={p.pattern} className={`flex items-center justify-between rounded-lg px-3 py-1.5 border ${shotColors[p.shot_type] || "border-zinc-700 bg-zinc-800/50"}`}>
          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-300 capitalize font-medium">{p.shot_type}</span>
            <span className="text-[10px] text-zinc-500">{p.direction.replace(/_/g, " ")}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-400 font-mono">{p.count}x</span>
            <span className="text-[10px] text-zinc-500">({p.pct.toFixed(0)}%)</span>
          </div>
        </div>
      ))}
    </div>
  );
}

function SinglePlayerCard({ card, analytics, playerKey }: {
  card: PlayerCard;
  analytics: AnalyticsData | null;
  playerKey: "player_a" | "player_b";
}) {
  const isA = playerKey === "player_a";
  const name = isA ? "Player A" : "Player B";
  const playerAnalytics = analytics ? (isA ? analytics.player_a : analytics.player_b) : null;
  const patterns = analytics?.shot_pattern_dominance?.[playerKey] ?? [];

  return (
    <div className={`rounded-xl bg-zinc-900 border ${isA ? "border-blue-500/30" : "border-orange-500/30"} p-5 space-y-4`}>
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className={`w-10 h-10 rounded-lg ${isA ? "bg-blue-500/20" : "bg-orange-500/20"} flex items-center justify-center`}>
          <span className={`text-lg font-bold ${isA ? "text-blue-400" : "text-orange-400"}`}>
            {isA ? "A" : "B"}
          </span>
        </div>
        <div>
          <h4 className="text-sm font-semibold text-white">{name}</h4>
          {playerAnalytics && (
            <span className="text-xs text-zinc-500">
              {playerAnalytics.total_shots} shots &middot; {Math.round(playerAnalytics.avg_shot_speed_m_s * 3.6)} km/h avg
            </span>
          )}
        </div>
      </div>

      {/* Exploit Plan - the headline coaching insight */}
      {card.card.exploit_plan && (
        <div className={`rounded-lg px-4 py-3 ${isA ? "bg-blue-500/5 border border-blue-500/20" : "bg-orange-500/5 border border-orange-500/20"}`}>
          <div className="flex items-center gap-1.5 mb-1">
            <span className="text-[10px] font-bold uppercase tracking-wider text-amber-400">Game Plan</span>
          </div>
          <p className="text-sm text-zinc-200 leading-relaxed">{card.card.exploit_plan}</p>
        </div>
      )}

      {/* Shot Distribution */}
      {playerAnalytics && Object.keys(playerAnalytics.shot_type_counts).length > 0 && (
        <div className="space-y-2">
          <h5 className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Shot Distribution</h5>
          <ShotDistPie counts={playerAnalytics.shot_type_counts} label={name} />
        </div>
      )}

      {/* Top Shot Patterns */}
      {patterns.length > 0 && (
        <div className="space-y-2">
          <h5 className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Top Patterns</h5>
          <TopPatterns patterns={patterns} />
        </div>
      )}

      {/* Shot Direction Breakdown */}
      {playerAnalytics && Object.keys(playerAnalytics.shot_direction_pcts).length > 0 && (
        <div className="space-y-2">
          <h5 className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Shot Directions</h5>
          <ShotDirectionBreakdown dirPcts={playerAnalytics.shot_direction_pcts} />
        </div>
      )}

      {/* Error Rates */}
      {playerAnalytics && (
        <ErrorRateSection
          byShot={playerAnalytics.error_rate_by_shot_type}
          byRally={playerAnalytics.error_rate_by_rally_length}
        />
      )}

      {/* Tendencies */}
      {card.card.tendencies.length > 0 && (
        <div className="space-y-2">
          <h5 className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Tendencies</h5>
          {card.card.tendencies.map((t, i) => (
            <p key={i} className="text-sm text-zinc-300 leading-relaxed">{t}</p>
          ))}
        </div>
      )}

      {/* Serve Summary */}
      {card.card.serve_summary && (
        <div className="space-y-1">
          <h5 className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Serve</h5>
          <p className="text-sm text-zinc-300">{card.card.serve_summary}</p>
        </div>
      )}

      {/* Serve Zone Win Rates */}
      {playerAnalytics && Object.keys(playerAnalytics.serve_zone_win_rate).length > 0 && (
        <div className="space-y-2">
          <h5 className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Serve Zone Win Rate</h5>
          <div className="grid grid-cols-2 gap-2">
            {Object.entries(playerAnalytics.serve_zone_win_rate).map(([zone, rate]) => (
              <div key={zone} className="bg-zinc-800/50 rounded-lg px-3 py-2">
                <span className="text-xs text-zinc-500 capitalize">{zone.replace(/_/g, " ")}</span>
                <div className="text-sm font-semibold text-white">{rate}%</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Weaknesses */}
      {card.weaknesses.weaknesses.length > 0 && (
        <div className="space-y-2">
          <h5 className="text-xs font-medium text-red-400/80 uppercase tracking-wider">Weaknesses</h5>
          {card.weaknesses.weaknesses.map((w, i) => (
            <div key={i} className="bg-red-500/5 border border-red-500/15 rounded-lg px-3 py-2 space-y-1">
              <p className="text-sm text-red-300">{w.description}</p>
              <p className="text-xs text-red-400/60">{w.data_point}</p>
              {w.points_cost > 0 && (
                <span className="inline-block text-[10px] px-2 py-0.5 rounded bg-red-500/10 text-red-400">
                  Cost: {w.points_cost} points
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function PlayerCardView({ playerACard, playerBCard, analytics }: Props) {
  if (!playerACard && !playerBCard) return null;

  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold text-zinc-300 px-1">Player Analysis</h3>
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
