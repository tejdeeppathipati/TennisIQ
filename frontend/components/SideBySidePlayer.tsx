"use client";

import { useRef, useState, useEffect, useCallback } from "react";

interface Props {
  rawVideoUrl?: string;
  overlayVideoUrl?: string;
  onTimeUpdate?: (sec: number) => void;
  seekTo?: number | null;
}

export default function SideBySidePlayer({ rawVideoUrl, overlayVideoUrl, onTimeUpdate, seekTo }: Props) {
  const rawRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLVideoElement>(null);
  const scrubRef = useRef<HTMLInputElement>(null);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const syncing = useRef(false);

  const syncVideos = useCallback((time: number) => {
    if (syncing.current) return;
    syncing.current = true;
    if (rawRef.current && Math.abs(rawRef.current.currentTime - time) > 0.1) {
      rawRef.current.currentTime = time;
    }
    if (overlayRef.current && Math.abs(overlayRef.current.currentTime - time) > 0.1) {
      overlayRef.current.currentTime = time;
    }
    syncing.current = false;
  }, []);

  useEffect(() => {
    if (seekTo != null && seekTo >= 0) {
      syncVideos(seekTo);
      requestAnimationFrame(() => {
        setCurrentTime(seekTo);
      });
      onTimeUpdate?.(seekTo);
    }
  }, [seekTo, syncVideos, onTimeUpdate]);

  const handleTimeUpdate = () => {
    const ref = overlayRef.current ?? rawRef.current;
    if (!ref || syncing.current) return;
    const t = ref.currentTime;
    setCurrentTime(t);
    onTimeUpdate?.(t);
    syncVideos(t);
  };

  const handleLoadedMetadata = () => {
    const ref = overlayRef.current ?? rawRef.current;
    if (ref) setDuration(ref.duration);
  };

  const handleScrub = (e: React.ChangeEvent<HTMLInputElement>) => {
    const t = parseFloat(e.target.value);
    setCurrentTime(t);
    syncVideos(t);
    onTimeUpdate?.(t);
  };

  const togglePlay = () => {
    if (playing) {
      rawRef.current?.pause();
      overlayRef.current?.pause();
    } else {
      rawRef.current?.play();
      overlayRef.current?.play();
    }
    setPlaying(!playing);
  };

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, "0")}`;
  };

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {rawVideoUrl ? (
          <div className="relative rounded-xl overflow-hidden bg-black">
            <div className="absolute top-2 left-2 z-10 px-2 py-0.5 rounded bg-black/70 text-xs text-gray-300">
              Raw Footage
            </div>
            <video
              ref={rawRef}
              src={rawVideoUrl}
              className="w-full aspect-video"
              onLoadedMetadata={handleLoadedMetadata}
              muted
              playsInline
            />
          </div>
        ) : (
          <div className="rounded-xl bg-zinc-900 flex items-center justify-center aspect-video">
            <span className="text-zinc-600 text-sm">Raw footage not available</span>
          </div>
        )}

        {overlayVideoUrl ? (
          <div className="relative rounded-xl overflow-hidden bg-black">
            <div className="absolute top-2 left-2 z-10 px-2 py-0.5 rounded bg-black/70 text-xs text-gray-300">
              Overlay Analysis
            </div>
            <video
              ref={overlayRef}
              src={overlayVideoUrl}
              className="w-full aspect-video"
              onTimeUpdate={handleTimeUpdate}
              onLoadedMetadata={handleLoadedMetadata}
              muted
              playsInline
            />
          </div>
        ) : (
          <div className="rounded-xl bg-zinc-900 flex items-center justify-center aspect-video">
            <span className="text-zinc-600 text-sm">Overlay not available</span>
          </div>
        )}
      </div>

      {/* Unified scrub bar */}
      <div className="flex items-center gap-3 px-1">
        <button
          onClick={togglePlay}
          className="w-9 h-9 rounded-lg bg-zinc-800 hover:bg-zinc-700 flex items-center justify-center text-white transition-colors shrink-0"
        >
          {playing ? (
            <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor"><rect x="2" y="1" width="3.5" height="12" rx="1" /><rect x="8.5" y="1" width="3.5" height="12" rx="1" /></svg>
          ) : (
            <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor"><path d="M3 1.5v11l9-5.5z" /></svg>
          )}
        </button>
        <span className="text-xs text-zinc-500 font-mono w-10 shrink-0">{formatTime(currentTime)}</span>
        <input
          ref={scrubRef}
          type="range"
          min={0}
          max={duration || 1}
          step={0.01}
          value={currentTime}
          onChange={handleScrub}
          className="flex-1 h-1.5 rounded-full appearance-none bg-zinc-700 accent-green-500 cursor-pointer"
        />
        <span className="text-xs text-zinc-500 font-mono w-10 shrink-0 text-right">{formatTime(duration)}</span>
      </div>
    </div>
  );
}
