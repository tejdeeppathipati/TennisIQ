"use client";

import type { AnalyticsData, MatchFlowData, PatternItem } from "@/lib/types";

interface Props {
  analytics: AnalyticsData | null;
  matchFlow: MatchFlowData | null;
}

function MomentumChart({ data }: { data: AnalyticsData }) {
  const points = data.momentum_data;
  if (points.length < 2) return null;

  const width = 600;
  const height = 120;
  const padding = { top: 20, right: 20, bottom: 25, left: 30 };
  const plotW = width - padding.left - padding.right;
  const plotH = height - padding.top - padding.bottom;

  const maxMomentum = Math.max(...points.map((p) => Math.max(p.a_momentum, p.b_momentum)), 1);
  const xStep = plotW / Math.max(points.length - 1, 1);

  const aPath = points
    .map((p, i) => {
      const x = padding.left + i * xStep;
      const y = padding.top + plotH - (p.a_momentum / maxMomentum) * plotH;
      return `${i === 0 ? "M" : "L"} ${x} ${y}`;
    })
    .join(" ");

  const bPath = points
    .map((p, i) => {
      const x = padding.left + i * xStep;
      const y = padding.top + plotH - (p.b_momentum / maxMomentum) * plotH;
      return `${i === 0 ? "M" : "L"} ${x} ${y}`;
    })
    .join(" ");

  return (
    <div className="space-y-2">
      <h5 className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Momentum</h5>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full" preserveAspectRatio="xMidYMid meet">
        {/* Grid */}
        <line x1={padding.left} y1={padding.top + plotH / 2} x2={width - padding.right} y2={padding.top + plotH / 2} stroke="#27272a" strokeWidth="1" strokeDasharray="4 4" />
        <line x1={padding.left} y1={padding.top + plotH} x2={width - padding.right} y2={padding.top + plotH} stroke="#3f3f46" strokeWidth="1" />

        {/* Lines */}
        <path d={aPath} fill="none" stroke="#3b82f6" strokeWidth="2" strokeLinejoin="round" />
        <path d={bPath} fill="none" stroke="#f97316" strokeWidth="2" strokeLinejoin="round" />

        {/* Labels */}
        <text x={padding.left} y={height - 4} fill="#71717a" fontSize="9" fontFamily="system-ui">Point 0</text>
        <text x={width - padding.right} y={height - 4} fill="#71717a" fontSize="9" fontFamily="system-ui" textAnchor="end">Point {points.length - 1}</text>

        {/* Legend */}
        <rect x={width - 120} y={4} width="8" height="8" rx="2" fill="#3b82f6" />
        <text x={width - 108} y={12} fill="#a1a1aa" fontSize="9" fontFamily="system-ui">Player A</text>
        <rect x={width - 60} y={4} width="8" height="8" rx="2" fill="#f97316" />
        <text x={width - 48} y={12} fill="#a1a1aa" fontSize="9" fontFamily="system-ui">Player B</text>
      </svg>
    </div>
  );
}

function RallyLengthChart({ data }: { data: Record<string, number> }) {
  const entries = Object.entries(data);
  if (entries.length === 0) return null;

  const maxVal = Math.max(...entries.map(([, v]) => v), 1);

  return (
    <div className="space-y-2">
      <h5 className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Rally Length Distribution</h5>
      <div className="flex items-end gap-3 h-20">
        {entries.map(([bucket, count]) => (
          <div key={bucket} className="flex flex-col items-center gap-1 flex-1">
            <div
              className="w-full bg-emerald-500/30 rounded-t transition-all"
              style={{ height: `${(count / maxVal) * 100}%`, minHeight: count > 0 ? 4 : 0 }}
            />
            <span className="text-[10px] text-zinc-500">{bucket}</span>
            <span className="text-[10px] text-zinc-400 font-medium">{count}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function ShotPatternDominance({ data }: { data: Record<string, PatternItem[]> }) {
  const aPatterns = data.player_a ?? [];
  const bPatterns = data.player_b ?? [];
  if (aPatterns.length === 0 && bPatterns.length === 0) return null;

  const shotColors: Record<string, string> = {
    forehand: "bg-blue-500",
    backhand: "bg-amber-500",
    serve: "bg-emerald-500",
    neutral: "bg-zinc-600",
  };

  function PatternBar({ patterns, label, isA }: { patterns: PatternItem[]; label: string; isA: boolean }) {
    const top = patterns.slice(0, 4);
    const maxPct = Math.max(...top.map((p) => p.pct), 1);

    return (
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <span className={`w-5 h-5 rounded text-[10px] font-bold flex items-center justify-center ${isA ? "bg-blue-500/20 text-blue-400" : "bg-orange-500/20 text-orange-400"}`}>
            {isA ? "A" : "B"}
          </span>
          <span className="text-xs text-zinc-400 font-medium">{label}</span>
        </div>
        <div className="space-y-1.5">
          {top.map((p) => (
            <div key={p.pattern} className="flex items-center gap-2">
              <div className="w-20 text-[10px] text-zinc-500 capitalize truncate">{p.shot_type}</div>
              <div className="flex-1 h-4 bg-zinc-800 rounded-full overflow-hidden relative">
                <div
                  className={`h-full rounded-full ${shotColors[p.shot_type] || "bg-zinc-600"} transition-all`}
                  style={{ width: `${(p.pct / maxPct) * 100}%`, opacity: 0.7 }}
                />
                <span className="absolute inset-0 flex items-center pl-2 text-[9px] font-medium text-white/80">
                  {p.direction.replace(/_/g, " ")}
                </span>
              </div>
              <span className="w-10 text-right text-[10px] text-zinc-400 font-mono">{p.pct.toFixed(0)}%</span>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <h5 className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Shot Pattern Dominance</h5>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {aPatterns.length > 0 && <PatternBar patterns={aPatterns} label="Player A" isA />}
        {bPatterns.length > 0 && <PatternBar patterns={bPatterns} label="Player B" isA={false} />}
      </div>
    </div>
  );
}

export default function MatchFlowChart({ analytics, matchFlow }: Props) {
  if (!analytics && !matchFlow) return null;

  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold text-zinc-300 px-1">Match Flow</h3>
      <div className="rounded-xl bg-zinc-900 border border-zinc-800 p-5 space-y-5">
        {/* Summary stats */}
        {analytics && (
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
            <div className="bg-zinc-800/50 rounded-lg px-3 py-2 text-center">
              <div className="text-lg font-bold text-white">{analytics.total_points}</div>
              <div className="text-[10px] text-zinc-500 uppercase">Points</div>
            </div>
            <div className="bg-zinc-800/50 rounded-lg px-3 py-2 text-center">
              <div className="text-lg font-bold text-white">{analytics.total_shots}</div>
              <div className="text-[10px] text-zinc-500 uppercase">Total Shots</div>
            </div>
            <div className="bg-zinc-800/50 rounded-lg px-3 py-2 text-center">
              <div className="text-lg font-bold text-white">{analytics.rally_length_avg.toFixed(1)}</div>
              <div className="text-[10px] text-zinc-500 uppercase">Avg Rally</div>
            </div>
            <div className="bg-zinc-800/50 rounded-lg px-3 py-2 text-center">
              <div className="text-lg font-bold text-blue-400">
                {analytics.player_a.total_shots}
              </div>
              <div className="text-[10px] text-zinc-500 uppercase">A Shots</div>
            </div>
            <div className="bg-zinc-800/50 rounded-lg px-3 py-2 text-center">
              <div className="text-lg font-bold text-orange-400">
                {analytics.player_b.total_shots}
              </div>
              <div className="text-[10px] text-zinc-500 uppercase">B Shots</div>
            </div>
          </div>
        )}

        {/* Momentum chart */}
        {analytics && analytics.momentum_data.length > 1 && (
          <MomentumChart data={analytics} />
        )}

        {/* Rally length distribution */}
        {analytics && Object.keys(analytics.rally_length_distribution).length > 0 && (
          <RallyLengthChart data={analytics.rally_length_distribution} />
        )}

        {/* Shot Pattern Dominance */}
        {analytics && analytics.shot_pattern_dominance && (
          <ShotPatternDominance data={analytics.shot_pattern_dominance} />
        )}

        {/* Match flow insights */}
        {matchFlow && matchFlow.insights.length > 0 && (
          <div className="space-y-2">
            <h5 className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Insights</h5>
            {matchFlow.insights.map((insight, i) => (
              <div key={i} className="bg-zinc-800/40 rounded-lg px-3 py-2">
                <p className="text-sm text-zinc-300 leading-relaxed">{insight.description}</p>
                {insight.timestamp_range && (
                  <span className="text-[10px] text-zinc-600 mt-1 block">
                    {insight.timestamp_range[0].toFixed(1)}s &ndash; {insight.timestamp_range[1].toFixed(1)}s
                  </span>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
