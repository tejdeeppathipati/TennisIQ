"use client";

import { useState, useRef } from "react";
import { useRouter } from "next/navigation";
import { ingestURL, ingestUpload } from "@/lib/api";

export default function Home() {
  const router = useRouter();
  const [url, setUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleSubmitURL = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!url.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const { job_id } = await ingestURL(url.trim());
      router.push(`/results/${job_id}`);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to start pipeline.");
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const { job_id } = await ingestUpload(file);
      router.push(`/results/${job_id}`);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Upload failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-black flex flex-col items-center justify-center px-4">
      <div className="w-full max-w-lg space-y-8">
        {/* Logo */}
        <div className="text-center space-y-2">
          <div className="w-16 h-16 rounded-2xl bg-green-600 flex items-center justify-center text-2xl font-black text-white mx-auto">
            T
          </div>
          <h1 className="text-3xl font-bold text-white">
            Tennis<span className="text-green-400">IQ</span>
          </h1>
          <p className="text-sm text-zinc-500">
            AI-powered tennis match analysis for college coaches
          </p>
        </div>

        {/* URL input */}
        <form onSubmit={handleSubmitURL} className="space-y-3">
          <div className="relative">
            <input
              type="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="Paste a YouTube match URL..."
              className="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-3.5 text-sm text-white placeholder-zinc-600 focus:outline-none focus:border-green-500/50 transition-colors"
              disabled={loading}
            />
          </div>
          <button
            type="submit"
            disabled={loading || !url.trim()}
            className="w-full py-3.5 rounded-xl bg-green-600 hover:bg-green-500 text-white font-semibold text-sm transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {loading ? "Starting pipeline..." : "Analyze Match"}
          </button>
        </form>

        {/* Divider */}
        <div className="flex items-center gap-3">
          <div className="flex-1 h-px bg-zinc-800" />
          <span className="text-xs text-zinc-600">or</span>
          <div className="flex-1 h-px bg-zinc-800" />
        </div>

        {/* Upload */}
        <button
          onClick={() => fileRef.current?.click()}
          disabled={loading}
          className="w-full py-3.5 rounded-xl border border-zinc-800 hover:border-zinc-700 text-zinc-400 hover:text-white text-sm transition-colors disabled:opacity-40"
        >
          Upload an MP4 file
        </button>
        <input
          ref={fileRef}
          type="file"
          accept=".mp4,video/mp4"
          onChange={handleUpload}
          className="hidden"
        />

        {error && (
          <div className="bg-red-950/50 border border-red-800 rounded-xl p-3">
            <p className="text-red-400 text-xs">{error}</p>
          </div>
        )}
      </div>
    </div>
  );
}
