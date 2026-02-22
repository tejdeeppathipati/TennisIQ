"use client";

import type { HeatmapData } from "@/lib/types";

interface Props {
  errorHeatmap: HeatmapData | null;
  playerAHeatmap: HeatmapData | null;
  playerBHeatmap: HeatmapData | null;
}

const CELL_W = 16;
const CELL_H = 12;

function HeatmapGrid({ data, title, colorFn }: { data: HeatmapData; title: string; colorFn: (v: number, max: number) => string }) {
  if (!data.grid.length) {
    return (
      <div className="rounded-lg bg-zinc-900 border border-zinc-800 p-3 text-center">
        <p className="text-zinc-500 text-[11px]">{title}</p>
        <p className="text-zinc-600 text-[10px]">No positions recorded — player or bounce not detected reliably.</p>
      </div>
    );
  }

  const max = Math.max(...data.grid.flat(), 1);
  const rows = data.grid.length;
  const cols = data.grid[0]?.length ?? 0;
  const width = cols * CELL_W;
  const height = rows * CELL_H;

  return (
    <div className="space-y-1">
      <p className="text-xs font-medium text-zinc-400">{title}</p>
      <div className="rounded-lg overflow-hidden border border-zinc-800 inline-block">
        <svg width={width} height={height}>
          {data.grid.map((row, ri) =>
            row.map((val, ci) => (
              <rect
                key={`${ri}-${ci}`}
                x={ci * CELL_W}
                y={ri * CELL_H}
                width={CELL_W}
                height={CELL_H}
                fill={colorFn(val, max)}
              />
            ))
          )}
        </svg>
      </div>
    </div>
  );
}

function redScale(v: number, max: number) {
  const t = max > 0 ? v / max : 0;
  const r = Math.round(30 + t * 225);
  const g = Math.round(30 - t * 20);
  const b = Math.round(30 - t * 20);
  return `rgb(${r},${g},${b})`;
}

function blueScale(v: number, max: number) {
  const t = max > 0 ? v / max : 0;
  const r = Math.round(30 - t * 20);
  const g = Math.round(30 + t * 80);
  const b = Math.round(30 + t * 225);
  return `rgb(${r},${g},${b})`;
}

function orangeScale(v: number, max: number) {
  const t = max > 0 ? v / max : 0;
  const r = Math.round(30 + t * 225);
  const g = Math.round(30 + t * 130);
  const b = Math.round(30 - t * 20);
  return `rgb(${r},${g},${b})`;
}

export default function HeatmapViewer({ errorHeatmap, playerAHeatmap, playerBHeatmap }: Props) {
  const hasAny = errorHeatmap || playerAHeatmap || playerBHeatmap;
  if (!hasAny) return null;

  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold text-zinc-300 px-1">Court Heatmaps</h3>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 rounded-xl bg-zinc-900 border border-zinc-800 p-4">
        {errorHeatmap && (
          <HeatmapGrid
            data={errorHeatmap}
            title={`Errors (${errorHeatmap.total_out_bounces ?? 0} out)`}
            colorFn={redScale}
          />
        )}
        {playerAHeatmap && (
          <HeatmapGrid
            data={playerAHeatmap}
            title={`Player A (${playerAHeatmap.total_frames ?? 0} frames)`}
            colorFn={blueScale}
          />
        )}
        {playerBHeatmap && (
          <HeatmapGrid
            data={playerBHeatmap}
            title={`Player B (${playerBHeatmap.total_frames ?? 0} frames)`}
            colorFn={orangeScale}
          />
        )}
      </div>
    </div>
  );
}
