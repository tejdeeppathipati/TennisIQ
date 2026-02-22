"use client";

import { useState, useCallback } from "react";
import { createCoachNote, deleteCoachNote } from "@/lib/api";
import type { CoachNote } from "@/lib/types";

interface Props {
  jobId: string;
  pointIdx: number;
  notes: CoachNote[];
  currentVideoTime: number;
  onNotesChanged: () => void;
}

function formatTimestamp(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}

const PLAYER_COLORS = {
  player_a: { bg: "bg-blue-500/20", text: "text-blue-400", border: "border-blue-500/40", label: "Player A" },
  player_b: { bg: "bg-orange-500/20", text: "text-orange-400", border: "border-orange-500/40", label: "Player B" },
} as const;

export default function CoachNotes({ jobId, pointIdx, notes, currentVideoTime, onNotesChanged }: Props) {
  const [text, setText] = useState("");
  const [player, setPlayer] = useState<"player_a" | "player_b">("player_a");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const pointNotes = notes
    .filter((n) => n.point_idx === pointIdx)
    .sort((a, b) => a.timestamp_sec - b.timestamp_sec);

  const handleAdd = useCallback(async () => {
    const trimmed = text.trim();
    if (!trimmed) return;
    setSaving(true);
    setError(null);
    try {
      await createCoachNote(jobId, pointIdx, currentVideoTime, trimmed, player);
      setText("");
      onNotesChanged();
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
      console.error("Failed to save note:", e);
    } finally {
      setSaving(false);
    }
  }, [jobId, pointIdx, currentVideoTime, text, player, onNotesChanged]);

  const handleDelete = useCallback(
    async (noteId: number) => {
      try {
        await deleteCoachNote(jobId, noteId);
        onNotesChanged();
      } catch (e) {
        console.error("Failed to delete note:", e);
      }
    },
    [jobId, onNotesChanged],
  );

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleAdd();
    }
  };

  return (
    <div className="space-y-3">
      <h5 className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Coach Notes</h5>

      {pointNotes.length > 0 && (
        <div className="space-y-1.5 max-h-48 overflow-y-auto">
          {pointNotes.map((n) => {
            const pc = PLAYER_COLORS[n.player] ?? PLAYER_COLORS.player_a;
            return (
              <div key={n.id} className="flex items-start gap-2 bg-zinc-800/60 rounded-lg px-3 py-2 group">
                <span className="text-[10px] text-green-400 font-mono shrink-0 pt-0.5">
                  {formatTimestamp(n.timestamp_sec)}
                </span>
                <span className={`text-[10px] px-1.5 py-0.5 rounded shrink-0 ${pc.bg} ${pc.text}`}>
                  {pc.label}
                </span>
                <span className="text-sm text-zinc-300 flex-1">{n.note_text}</span>
                <button
                  onClick={() => handleDelete(n.id)}
                  className="text-zinc-600 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-all text-xs shrink-0"
                >
                  &times;
                </button>
              </div>
            );
          })}
        </div>
      )}

      {/* Player selector */}
      <div className="flex gap-1.5">
        {(["player_a", "player_b"] as const).map((p) => {
          const pc = PLAYER_COLORS[p];
          const active = player === p;
          return (
            <button
              key={p}
              onClick={() => setPlayer(p)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors border ${
                active
                  ? `${pc.bg} ${pc.text} ${pc.border}`
                  : "bg-zinc-800/50 text-zinc-500 border-transparent hover:bg-zinc-800"
              }`}
            >
              {pc.label}
            </button>
          );
        })}
      </div>

      {/* Input row */}
      <div className="flex gap-2">
        <div className="text-[10px] text-zinc-500 font-mono shrink-0 pt-2.5">
          @ {formatTimestamp(currentVideoTime)}
        </div>
        <input
          type="text"
          value={text}
          onChange={(e) => { setText(e.target.value); setError(null); }}
          onKeyDown={handleKeyDown}
          placeholder={`Add a note for ${PLAYER_COLORS[player].label}...`}
          className="flex-1 bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-green-500/50"
        />
        <button
          onClick={handleAdd}
          disabled={saving || !text.trim()}
          className="px-3 py-2 rounded-lg bg-green-600 hover:bg-green-500 disabled:bg-zinc-700 disabled:text-zinc-500 text-white text-sm font-medium transition-colors shrink-0"
        >
          {saving ? "..." : "Add"}
        </button>
      </div>

      {error && (
        <p className="text-xs text-red-400 px-1">{error}</p>
      )}
    </div>
  );
}
