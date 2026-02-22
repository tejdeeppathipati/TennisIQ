"use client";

import { useState } from "react";
import SideBySidePlayer from "./SideBySidePlayer";

interface Props {
  rawVideoUrl?: string;
  overlayVideoUrl?: string;
}

export default function MatchVision({ rawVideoUrl, overlayVideoUrl }: Props) {
  const [open, setOpen] = useState(false);

  if (!rawVideoUrl && !overlayVideoUrl) return null;

  return (
    <div className="rounded-xl border border-zinc-800 overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-5 py-3 bg-zinc-900 hover:bg-zinc-800/80 transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-zinc-300">Match Vision</span>
          <span className="text-[10px] px-2 py-0.5 rounded bg-zinc-800 text-zinc-500">CV Overlay</span>
        </div>
        <svg
          width="16"
          height="16"
          viewBox="0 0 16 16"
          fill="currentColor"
          className={`text-zinc-500 transition-transform ${open ? "rotate-180" : ""}`}
        >
          <path d="M4 6l4 4 4-4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>

      {open && (
        <div className="p-4 bg-zinc-950 border-t border-zinc-800">
          <SideBySidePlayer
            rawVideoUrl={rawVideoUrl}
            overlayVideoUrl={overlayVideoUrl}
          />
        </div>
      )}
    </div>
  );
}
