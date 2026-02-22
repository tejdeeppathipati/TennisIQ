"use client";

import type { HistoricalInsights } from "@/lib/types";

interface Props {
  history: HistoricalInsights | null | undefined;
}

function PlayerHistoryPanel({
  label,
  data,
  accent,
}: {
  label: string;
  data: HistoricalInsights["player_a"];
  accent: string;
}) {
  const topWeaknesses = data.persistent_weaknesses
    .filter((w) => w.confirmed)
    .slice(0, 2);

  return (
    <div className={`rounded-xl border ${accent} bg-zinc-900 p-4 space-y-2`}>
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-semibold text-white">{label}</h4>
        <span className="text-[11px] text-zinc-500">{data.match_count} matches</span>
      </div>
      {data.summary.length > 0 ? (
        <ul className="space-y-1">
          {data.summary.slice(0, 2).map((line, i) => (
            <li key={i} className="text-xs text-zinc-300">
              {line}
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-xs text-zinc-500">Not enough historical signal yet.</p>
      )}

      {topWeaknesses.length > 0 && (
        <div className="pt-1 border-t border-zinc-800 space-y-1">
          {topWeaknesses.map((w) => (
            <p key={w.name} className="text-xs text-zinc-400">
              {w.name}: <span className="text-zinc-300">{w.baseline_rate_pct}%</span> baseline
              ({w.matches_triggered}/{w.matches_with_data} matches)
            </p>
          ))}
        </div>
      )}
    </div>
  );
}

export default function HistoricalInsightsCard({ history }: Props) {
  if (!history) return null;

  if (!history.enabled) {
    return (
      <div className="rounded-xl border border-zinc-800 bg-zinc-900 p-4">
        <h3 className="text-base font-semibold text-white mb-1">Historical Trends</h3>
        <p className="text-sm text-zinc-400">
          {history.total_matches_considered} match{history.total_matches_considered === 1 ? "" : "es"} on record.
          Add {Math.max(0, history.minimum_matches - history.total_matches_considered)} more to unlock persistent-vs-fluke insights.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <h3 className="text-base font-semibold text-white px-1">Historical Trends</h3>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <PlayerHistoryPanel label="Player A" data={history.player_a} accent="border-blue-500/30" />
        <PlayerHistoryPanel label="Player B" data={history.player_b} accent="border-orange-500/30" />
      </div>
    </div>
  );
}

