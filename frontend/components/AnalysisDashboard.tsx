"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import * as Plot from "@observablehq/plot";
import type { AnalysisData, StatSummary } from "@/lib/types";

type Props = {
  analysis: AnalysisData;
};

export default function AnalysisDashboard({ analysis }: Props) {
  const quality = analysis.quality ?? {};
  const serve = analysis.serve ?? {};
  const rally = analysis.rally ?? {};
  const errors = analysis.errors ?? {};
  const players = analysis.players ?? {};
  const ball = analysis.ball ?? {};
  const eventTimeline = useMemo(
    () => analysis.events?.timeline ?? [],
    [analysis.events]
  );

  const ballCoverage = quality.ball_coverage_pct ?? 0;
  const homography = quality.homography_reliable_pct ?? 0;
  const eventCount = quality.events_total ?? 0;

  const ballSpeedKmH = useMemo(() => {
    const values = ball.speed_samples_m_s ?? [];
    return values.map((v) => v * 3.6).filter((v) => Number.isFinite(v));
  }, [ball.speed_samples_m_s]);

  const serveDepth = serve.depth_samples_m ?? [];
  const serveWidth = serve.width_samples_m ?? [];
  const rallyHits = rally.rally_hits ?? [];
  const rallyDurations = rally.rally_durations_sec ?? [];
  const rallyReasons = rally.end_reason_counts ?? {};
  const serveZones = serve.zone_counts ?? {};
  const serveSamples = serve.sample_count ?? 0;

  const hitDeltas = useMemo(() => ball.hit_speed_deltas ?? [], [ball.hit_speed_deltas]);
  const avgHitDelta = useMemo(() => {
    const vals = hitDeltas
      .map((d) => d.delta)
      .filter((v): v is number => typeof v === "number" && Number.isFinite(v));
    if (!vals.length) return null;
    return vals.reduce((a, b) => a + b, 0) / vals.length;
  }, [hitDeltas]);

  const [logOpen, setLogOpen] = useState(false);
  const [filterType, setFilterType] = useState<string>("all");
  const [filterSide, setFilterSide] = useState<string>("all");
  const [filterPlayer, setFilterPlayer] = useState<string>("all");

  const filteredTimeline = useMemo(() => {
    return eventTimeline.filter((e) => {
      if (filterType !== "all" && e.type !== filterType) return false;
      if (filterSide !== "all" && e.side !== filterSide) return false;
      if (filterPlayer !== "all" && e.player !== filterPlayer) return false;
      return true;
    });
  }, [eventTimeline, filterType, filterSide, filterPlayer]);

  return (
    <section className="space-y-6">
      <div className="rounded-xl bg-zinc-900 border border-zinc-800 p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-zinc-200">Quality & Coverage</h3>
          <div className="text-xs text-zinc-500">Always-on confidence</div>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
          <CoverageBadge label="Ball coverage" value={ballCoverage} warnBelow={40} suffix="%" />
          <CoverageBadge label="Homography" value={homography} warnBelow={85} suffix="%" />
          <CoverageBadge label="Events" value={eventCount} warnBelow={3} suffix="" />
          <CoverageBadge label="Points" value={quality.points_total ?? 0} warnBelow={1} suffix="" />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="rounded-xl bg-zinc-900 border border-zinc-800 p-4 space-y-3">
          <h3 className="text-sm font-semibold text-zinc-200">Serve Quality</h3>
          <div className="text-xs text-zinc-500">
            Fault rate: {formatPercent(serve.fault_rate)} · samples: {serveSamples}
          </div>
          <KeyValueGrid
            items={[
              { label: "Depth mean (m)", value: statValue(serve.depth_stats, "mean") },
              { label: "Width mean (m)", value: statValue(serve.width_stats, "mean") },
              { label: "Depth p95 (m)", value: statValue(serve.depth_stats, "p95") },
              { label: "Width p95 (m)", value: statValue(serve.width_stats, "p95") },
            ]}
          />
          {Object.keys(serveZones).length > 0 && (
            <div className="flex flex-wrap gap-2 text-[11px] text-zinc-400">
              {Object.entries(serveZones).map(([zone, count]) => (
                <span key={zone} className="px-2 py-1 rounded-md bg-zinc-800 text-zinc-200">
                  {zone.replace("_", " ")}: {count}
                </span>
              ))}
            </div>
          )}
          {serveSamples < 3 ? (
            <div className="text-[11px] text-zinc-500">Too few serves for charts (need 3+).</div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <PlotHistogram
                values={serveDepth}
                title="Serve depth (m)"
                xLabel="meters from service line"
              />
              <PlotHistogram
                values={serveWidth}
                title="Serve width (m)"
                xLabel="meters from center"
              />
            </div>
          )}
        </div>

        <div className="rounded-xl bg-zinc-900 border border-zinc-800 p-4 space-y-3">
          <h3 className="text-sm font-semibold text-zinc-200">Rally Structure</h3>
          {eventCount < 3 && (
            <div className="text-xs text-yellow-400">Low event count: tempo charts may be unreliable.</div>
          )}
          {rallyHits.length < 5 ? (
            <div className="text-[11px] text-zinc-500">Too few rallies for charts (need 5+).</div>
          ) : (
            <>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <PlotHistogram values={rallyHits} title="Rally hits" xLabel="hits per rally" />
                <PlotBar
                  title="End reasons"
                  data={Object.entries(rallyReasons).map(([key, value]) => ({ key, value }))}
                />
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <PlotHistogram values={rallyDurations} title="Rally duration (s)" xLabel="seconds" />
              </div>
            </>
          )}
          <KeyValueGrid
            items={[
              { label: "Hits/sec (mean)", value: formatNumber(rally.tempo_stats?.mean_hits_per_sec) },
              { label: "Inter-hit mean (s)", value: formatNumber(rally.tempo_stats?.mean_inter_hit_sec, 3) },
              { label: "Inter-hit p95 (s)", value: formatNumber(rally.tempo_stats?.p95_inter_hit_sec, 3) },
            ]}
          />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="rounded-xl bg-zinc-900 border border-zinc-800 p-4 space-y-3">
          <h3 className="text-sm font-semibold text-zinc-200">Errors</h3>
          <KeyValueGrid
            items={[
              { label: "Out count", value: formatNumber(errors.out_count ?? 0) },
              { label: "Out dist mean (m)", value: statValue(errors.out_distance_stats, "mean") },
              { label: "Out dist p95 (m)", value: statValue(errors.out_distance_stats, "p95") },
            ]}
          />
        </div>

        <div className="rounded-xl bg-zinc-900 border border-zinc-800 p-4 space-y-3">
          <h3 className="text-sm font-semibold text-zinc-200">Player Movement</h3>
          <PlayerStats label="Player A" stats={players.player_a ?? null} />
          <PlayerStats label="Player B" stats={players.player_b ?? null} />
        </div>

        <div className="rounded-xl bg-zinc-900 border border-zinc-800 p-4 space-y-3">
          <h3 className="text-sm font-semibold text-zinc-200">Ball Pace</h3>
          {ballCoverage < 40 ? (
            <div className="text-xs text-yellow-400">Low ball coverage: summary only.</div>
          ) : ballSpeedKmH.length < 5 ? (
            <div className="text-xs text-zinc-500">Too few speed samples for histogram (need 5+).</div>
          ) : (
            <PlotHistogram values={ballSpeedKmH} title="Ball speed (km/h)" xLabel="km/h" />
          )}
          <KeyValueGrid
            items={[
              { label: "Speed mean (m/s)", value: statValue(ball.speed_stats, "mean") },
              { label: "Speed p95 (m/s)", value: statValue(ball.speed_stats, "p95") },
              { label: "Hit delta mean (m/s)", value: formatNumber(avgHitDelta, 2) },
            ]}
          />
        </div>
      </div>

      <div className="rounded-xl bg-zinc-900 border border-zinc-800 p-4 space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-zinc-200">Atomic Event Log</h3>
          <div className="flex items-center gap-2 text-xs text-zinc-500">
            <span>{filteredTimeline.length}/{eventTimeline.length} events</span>
            <button
              onClick={() => setLogOpen((v) => !v)}
              className="px-2 py-1 rounded bg-zinc-800 text-[11px] hover:bg-zinc-700"
            >
              {logOpen ? "Hide" : "Show"}
            </button>
          </div>
        </div>
        {logOpen && (
          <>
            <div className="flex flex-wrap gap-2 text-[11px] text-zinc-400">
              <select
                value={filterType}
                onChange={(e) => setFilterType(e.target.value)}
                className="bg-zinc-900 border border-zinc-700 rounded px-2 py-1"
              >
                <option value="all">All types</option>
                <option value="hit">Hit</option>
                <option value="bounce">Bounce</option>
              </select>
              <select
                value={filterSide}
                onChange={(e) => setFilterSide(e.target.value)}
                className="bg-zinc-900 border border-zinc-700 rounded px-2 py-1"
              >
                <option value="all">All sides</option>
                <option value="near">Near</option>
                <option value="far">Far</option>
              </select>
              <select
                value={filterPlayer}
                onChange={(e) => setFilterPlayer(e.target.value)}
                className="bg-zinc-900 border border-zinc-700 rounded px-2 py-1"
              >
                <option value="all">All players</option>
                <option value="player_a">Player A</option>
                <option value="player_b">Player B</option>
              </select>
            </div>
            {filteredTimeline.length === 0 ? (
              <div className="text-xs text-zinc-500">No events match the filters.</div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-xs text-zinc-400">
                  <thead className="text-[11px] uppercase tracking-wide text-zinc-500">
                    <tr>
                      <th className="py-2 px-2 text-left">Time</th>
                      <th className="py-2 px-2 text-left">Type</th>
                      <th className="py-2 px-2 text-left">Side</th>
                      <th className="py-2 px-2 text-left">In/Out</th>
                      <th className="py-2 px-2 text-left">Speed Δ</th>
                      <th className="py-2 px-2 text-left">Angle</th>
                      <th className="py-2 px-2 text-left">Player</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-800">
                    {filteredTimeline.map((evt, idx) => (
                      <tr key={`${evt.t ?? idx}-${idx}`} className="hover:bg-zinc-950/60">
                        <td className="py-2 px-2">{formatNumber(evt.t, 3)}s</td>
                        <td className="py-2 px-2 text-zinc-200">{evt.type ?? "-"}</td>
                        <td className="py-2 px-2">{evt.side ?? "-"}</td>
                        <td className="py-2 px-2">{evt.in_out ?? "-"}</td>
                        <td className="py-2 px-2">
                          {formatSpeedDelta(evt.speed_before_m_s, evt.speed_after_m_s)}
                        </td>
                        <td className="py-2 px-2">{formatNumber(evt.direction_change_deg, 1)}</td>
                        <td className="py-2 px-2">{evt.player ?? "-"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}
      </div>
    </section>
  );
}

function CoverageBadge({
  label,
  value,
  warnBelow,
  suffix,
}: {
  label: string;
  value: number;
  warnBelow: number;
  suffix: string;
}) {
  const isWarn = value < warnBelow;
  const color = isWarn ? "text-yellow-300 bg-yellow-950 border-yellow-800" : "text-emerald-300 bg-emerald-950 border-emerald-800";
  return (
    <div className={`border rounded-lg px-3 py-2 ${color}`}>
      <div className="text-[11px] uppercase tracking-wide">{label}</div>
      <div className="text-sm font-semibold">{formatNumber(value)}{suffix}</div>
    </div>
  );
}

function PlotHistogram({
  values,
  title,
  xLabel,
}: {
  values: number[];
  title: string;
  xLabel: string;
}) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    ref.current.innerHTML = "";
    if (!values.length) return;
    const plot = Plot.plot({
      height: 160,
      marginLeft: 40,
      marginRight: 10,
      marginTop: 20,
      marginBottom: 30,
      x: { label: xLabel, tickFormat: (d) => String(d) },
      y: { label: "count", grid: true },
      style: plotStyle,
      marks: [
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        Plot.rectY(values, Plot.binX({ y: "count" }, { x: (d: any) => d, fill: "#22c55e", fillOpacity: 0.7 } as any)),
        Plot.ruleY([0]),
      ],
    });
    ref.current.append(plot);
    return () => plot.remove();
  }, [values, xLabel]);

  return (
    <div className="rounded-lg bg-zinc-950 border border-zinc-800 p-3">
      <div className="text-xs text-zinc-400 mb-2">{title}</div>
      <div ref={ref} />
    </div>
  );
}

function PlotBar({
  data,
  title,
}: {
  data: { key: string; value: number }[];
  title: string;
}) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    ref.current.innerHTML = "";
    if (!data.length) return;
    const plot = Plot.plot({
      height: 160,
      marginLeft: 60,
      marginRight: 10,
      marginTop: 20,
      marginBottom: 30,
      y: { label: null },
      x: { label: "count", grid: true },
      style: plotStyle,
      marks: [
        Plot.barX(data, { x: "value", y: "key", fill: "#22c55e" }),
        Plot.ruleX([0]),
      ],
    });
    ref.current.append(plot);
    return () => plot.remove();
  }, [data]);

  return (
    <div className="rounded-lg bg-zinc-950 border border-zinc-800 p-3">
      <div className="text-xs text-zinc-400 mb-2">{title}</div>
      <div ref={ref} />
    </div>
  );
}

function PlayerStats({
  label,
  stats,
}: {
  label: string;
  stats: { distance_m?: number; speed_stats?: StatSummary | null; zone_time_pct?: Record<string, number> | null } | null;
}) {
  if (!stats) {
    return (
      <div className="text-xs text-zinc-500">{label}: no data</div>
    );
  }
  return (
    <div className="text-xs text-zinc-400 space-y-1">
      <div className="text-zinc-200 font-semibold">{label}</div>
      <div>Distance: {formatNumber(stats.distance_m)} m</div>
      <div>Speed mean: {statValue(stats.speed_stats, "mean")} m/s</div>
      {stats.zone_time_pct && (
        <div className="flex gap-2">
          {Object.entries(stats.zone_time_pct).map(([key, value]) => (
            <span key={key} className="text-[11px] text-zinc-500">
              {key}: {formatNumber(value)}%
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function KeyValueGrid({ items }: { items: { label: string; value: string }[] }) {
  return (
    <div className="grid grid-cols-2 gap-2 text-xs text-zinc-400">
      {items.map((item) => (
        <div key={item.label} className="flex justify-between gap-2">
          <span>{item.label}</span>
          <span className="text-zinc-200">{item.value}</span>
        </div>
      ))}
    </div>
  );
}

function formatNumber(value?: number | null, decimals: number = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return value.toFixed(decimals);
}

function formatPercent(value?: number | null) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${(value * 100).toFixed(0)}%`;
}

function statValue(stats?: StatSummary | null, key?: keyof StatSummary) {
  if (!stats || !key) return "-";
  const val = stats[key];
  if (val === null || val === undefined || Number.isNaN(val)) return "-";
  return typeof val === "number" ? val.toFixed(2) : "-";
}

const plotStyle = {
  background: "transparent",
  color: "#e5e7eb",
  fontSize: "11px",
};

function formatSpeedDelta(before?: number | null, after?: number | null) {
  if (before === null || after === null || before === undefined || after === undefined) return "-";
  const delta = after - before;
  const sign = delta >= 0 ? "+" : "";
  return `${sign}${delta.toFixed(2)} m/s`;
}
