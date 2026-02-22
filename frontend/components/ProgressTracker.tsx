"use client";

import type { StatusResponse } from "@/lib/types";
import { STAGE_LABELS, STAGE_ORDER } from "@/lib/types";

interface Props {
  status: StatusResponse;
}

export default function ProgressTracker({ status }: Props) {
  const currentIdx = STAGE_ORDER.indexOf(status.stage);
  const progress = currentIdx >= 0 ? Math.round((currentIdx / (STAGE_ORDER.length - 1)) * 100) : 0;

  return (
    <div className="rounded-2xl bg-zinc-900 border border-zinc-800 p-6 space-y-5">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-white">Pipeline Progress</h2>
        <span className="text-sm text-zinc-400 font-mono">{progress}%</span>
      </div>

      {/* Progress bar */}
      <div className="h-2 rounded-full bg-zinc-800 overflow-hidden">
        <div
          className="h-full rounded-full bg-gradient-to-r from-green-500 to-emerald-400 transition-all duration-700"
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* Current stage description */}
      <div className="flex items-start gap-3">
        <div className="mt-1">
          {status.status === "error" ? (
            <div className="w-3 h-3 rounded-full bg-red-500" />
          ) : status.status === "complete" ? (
            <div className="w-3 h-3 rounded-full bg-green-500" />
          ) : (
            <div className="w-3 h-3 rounded-full bg-blue-500 animate-pulse" />
          )}
        </div>
        <div>
          <p className="text-sm font-medium text-white">
            {STAGE_LABELS[status.stage] ?? status.stage}
          </p>
          <p className="text-xs text-zinc-500 mt-0.5">{status.stage_description}</p>
        </div>
      </div>

      {/* Stage list */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        {STAGE_ORDER.slice(0, -1).map((stage, i) => {
          const isDone = i < currentIdx;
          const isCurrent = stage === status.stage;
          return (
            <div
              key={stage}
              className={`text-[10px] px-2 py-1.5 rounded-lg border text-center ${
                isDone
                  ? "bg-green-500/10 border-green-500/20 text-green-400"
                  : isCurrent
                  ? "bg-blue-500/10 border-blue-500/30 text-blue-400"
                  : "bg-zinc-900 border-zinc-800 text-zinc-600"
              }`}
            >
              {STAGE_LABELS[stage] ?? stage}
            </div>
          );
        })}
      </div>

      {status.error_message && (
        <div className="bg-red-950/50 border border-red-800 rounded-xl p-4">
          <p className="text-red-400 text-sm">{status.error_message}</p>
        </div>
      )}
    </div>
  );
}
