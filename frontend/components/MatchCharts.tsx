"use client";

import { useMemo, useState } from "react";
import type { DetectedPoint, AnalyticsData } from "@/lib/types";

interface Props {
  points: DetectedPoint[];
  analytics?: AnalyticsData | null;
  onPointClick?: (idx: number) => void;
}

// ── Rally Length helpers ───────────────────────────────────────────────────────
function bucket(hits: number): string {
  if (hits <= 3) return "1-3";
  if (hits <= 6) return "4-6";
  if (hits <= 9) return "7-9";
  return "10+";
}

const BUCKET_LABELS = ["1-3", "4-6", "7-9", "10+"];
const BUCKET_COLORS = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444"];

// ── Rally Length Bar Chart ─────────────────────────────────────────────────────
function RallyLengthChart({ points }: { points: DetectedPoint[] }) {
  const [hovered, setHovered] = useState<string | null>(null);

  const counts = useMemo(() => {
    const c: Record<string, number> = { "1-3": 0, "4-6": 0, "7-9": 0, "10+": 0 };
    points.forEach((p) => c[bucket(p.rally_hit_count)]++);
    return c;
  }, [points]);

  const max = Math.max(...Object.values(counts), 1);
  const total = Object.values(counts).reduce((s, v) => s + v, 0);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h5 className="text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">
          Rally Length Distribution
        </h5>
        <span className="text-[10px] text-zinc-600">{total} points</span>
      </div>
      <div className="flex items-end gap-2 h-24">
        {BUCKET_LABELS.map((label, i) => {
          const val = counts[label];
          const heightPct = (val / max) * 100;
          const isHov = hovered === label;
          return (
            <div
              key={label}
              className="flex flex-col items-center gap-1 flex-1 cursor-default"
              onMouseEnter={() => setHovered(label)}
              onMouseLeave={() => setHovered(null)}
            >
              {/* Value tooltip */}
              <span className={`text-[10px] font-bold transition-opacity duration-150 ${isHov ? "opacity-100 text-white" : "opacity-0"}`}>
                {val}
              </span>
              {/* Bar container */}
              <div className="w-full flex items-end justify-center" style={{ height: "60px" }}>
                <div
                  className="w-full rounded-t transition-all duration-200"
                  style={{
                    height: val === 0 ? "3px" : `${Math.max(6, heightPct)}%`,
                    background: BUCKET_COLORS[i],
                    opacity: isHov ? 1 : 0.65,
                  }}
                />
              </div>
              <span className="text-[10px] text-zinc-500 font-medium">{label}</span>
              <span className="text-[9px] text-zinc-600">{val}</span>
            </div>
          );
        })}
      </div>
      <p className="text-[10px] text-zinc-600">Hits per rally &middot; hover bar to highlight</p>
    </div>
  );
}

// ── Momentum Line Chart ────────────────────────────────────────────────────────
// Uses analytics.momentum_data which has winner per point
function MomentumChart({
  analytics,
  points,
  onPointClick,
}: {
  analytics: AnalyticsData;
  points: DetectedPoint[];
  onPointClick: (idx: number) => void;
}) {
  const [hovered, setHovered] = useState<number | null>(null);

  const W = 360;
  const H = 80;
  const PAD = { x: 16, y: 8 };

  const series = useMemo(() => {
    // Build running score from momentum_data
    if (analytics.momentum_data && analytics.momentum_data.length > 0) {
      let score = 0;
      return analytics.momentum_data.map((m) => {
        if (m.winner === "player_a") score++;
        else if (m.winner === "player_b") score--;
        return { score, i: m.point_idx, ts: m.timestamp_sec };
      });
    }
    // Fallback: build from points using end_reason heuristic (OUT = current player error, BALL_LOST = ambiguous)
    let score = 0;
    return points.map((p, i) => {
      if (p.end_reason === "OUT") score--; // typically the last hitter lost
      else if (p.rally_hit_count > 6) score++; // winner of long rally
      return { score, i, ts: p.start_sec };
    });
  }, [analytics, points]);

  if (series.length < 2) return null;

  const minScore = Math.min(...series.map((s) => s.score));
  const maxScore = Math.max(...series.map((s) => s.score));
  const range = Math.max(maxScore - minScore, 1);

  const toX = (i: number) => PAD.x + (i / (series.length - 1)) * (W - 2 * PAD.x);
  const toY = (score: number) => PAD.y + ((maxScore - score) / range) * (H - 2 * PAD.y);

  const polyline = series.map((s, idx) => `${toX(idx)},${toY(s.score)}`).join(" ");
  const zeroY = toY(0);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h5 className="text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">
          Momentum
        </h5>
        <span className="text-[10px] text-zinc-600">Click point to watch clip</span>
      </div>
      <div className="w-full overflow-x-auto">
        <svg
          viewBox={`0 0 ${W} ${H + 4}`}
          className="w-full"
          style={{ minWidth: "220px", height: `${H + 20}px` }}
        >
          {/* Zero line */}
          <line
            x1={PAD.x} y1={zeroY} x2={W - PAD.x} y2={zeroY}
            stroke="#3f3f46" strokeWidth={0.8} strokeDasharray="3 3"
          />

          {/* Gradient fill */}
          <defs>
            <linearGradient id="mgGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.25" />
              <stop offset="100%" stopColor="#3b82f6" stopOpacity="0.02" />
            </linearGradient>
          </defs>
          <polygon
            points={`${PAD.x},${zeroY} ${polyline} ${W - PAD.x},${zeroY}`}
            fill="url(#mgGrad)"
          />

          {/* Main line */}
          <polyline
            points={polyline}
            fill="none"
            stroke="#3b82f6"
            strokeWidth={1.5}
            strokeLinejoin="round"
          />

          {/* Interactive data points */}
          {series.map((s, idx) => {
            const x = toX(idx);
            const y = toY(s.score);
            const isHov = hovered === idx;
            return (
              <g key={idx}>
                <rect
                  x={x - 8} y={0} width={16} height={H + 4}
                  fill="transparent"
                  className="cursor-pointer"
                  onMouseEnter={() => setHovered(idx)}
                  onMouseLeave={() => setHovered(null)}
                  onClick={() => onPointClick(s.i)}
                />
                <circle
                  cx={x} cy={y}
                  r={isHov ? 4 : 2}
                  fill={isHov ? "#ffffff" : "#3b82f6"}
                  stroke={isHov ? "#3b82f6" : "none"}
                  strokeWidth={1}
                  className="pointer-events-none transition-all"
                />
                {isHov && (
                  <g className="pointer-events-none">
                    <rect
                      x={Math.min(x - 22, W - 62)} y={y - 22}
                      width={60} height={14}
                      rx={3} fill="#18181b" stroke="#3f3f46" strokeWidth={0.5}
                    />
                    <text
                      x={Math.min(x - 22, W - 62) + 30}
                      y={y - 12}
                      textAnchor="middle" fontSize={7.5} fill="#e4e4e7"
                    >
                      P{s.i} &middot; score:{s.score}
                    </text>
                  </g>
                )}
              </g>
            );
          })}

          {/* Axis labels */}
          <text x={PAD.x} y={H + 2} fontSize={7} fill="#52525b">P0</text>
          <text x={W - PAD.x} y={H + 2} textAnchor="end" fontSize={7} fill="#52525b">
            P{series.length - 1}
          </text>
          <text x={PAD.x - 2} y={PAD.y + 6} fontSize={7} fill="#3b82f6" textAnchor="middle">+A</text>
          <text x={PAD.x - 2} y={H - PAD.y} fontSize={7} fill="#ef4444" textAnchor="middle">+B</text>
        </svg>
      </div>
      <p className="text-[10px] text-zinc-600">
        Rising = Player A winning &middot; Falling = Player B winning
      </p>
    </div>
  );
}

// ── Export ─────────────────────────────────────────────────────────────────────
export default function MatchCharts({ points, analytics, onPointClick }: Props) {
  if (!points || points.length === 0) return null;
  return (
    <div className="space-y-3">
      <h3 className="text-base font-semibold text-white px-1">Match Trends</h3>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="rounded-xl bg-zinc-900 border border-zinc-800 p-5">
          <RallyLengthChart points={points} />
        </div>
        {analytics && (
          <div className="rounded-xl bg-zinc-900 border border-zinc-800 p-5">
            <MomentumChart
              analytics={analytics}
              points={points}
              onPointClick={onPointClick ?? (() => {})}
            />
          </div>
        )}
      </div>
    </div>
  );
}
