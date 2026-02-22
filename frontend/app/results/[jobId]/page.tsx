"use client";

import { useEffect, useState, useCallback, useRef, use } from "react";
import Link from "next/link";
import { getStatus, getResultsData, getCoachNotes } from "@/lib/api";
import type {
  StatusResponse,
  ResultsDataResponse,
  DetectedPoint,
  PlayerCard,
  ShotEvent,
  CoachingCard,
  AnalyticsData,
  AnalysisData,
  CoachNote,
} from "@/lib/types";
import { STAGE_LABELS } from "@/lib/types";
import ProgressTracker from "@/components/ProgressTracker";
import CheckpointReview from "@/components/CheckpointReview";
import PointClipViewer from "@/components/PointClipViewer";
import PDFExport from "@/components/PDFExport";
import PlayerCardView from "@/components/PlayerCardView";
import WeaknessReport from "@/components/WeaknessReport";
import MatchCharts from "@/components/MatchCharts";
import HistoricalInsightsCard from "@/components/HistoricalInsightsCard";

const POLL_INTERVAL = 5000;
const API_URL = (process.env.NEXT_PUBLIC_API_URL || "").trim().replace(/\/+$/, "") || "/backend";

export default function ResultsPage({
  params,
}: {
  params: Promise<{ jobId: string }>;
}) {
  const { jobId } = use(params);
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [data, setData] = useState<ResultsDataResponse | null>(null);
  const [notes, setNotes] = useState<CoachNote[]>([]);
  const [error, setError] = useState<string | null>(null);
  // Which point the user wants to jump to (driven by weakness evidence clicks / momentum clicks)
  const [focusPointIdx, setFocusPointIdx] = useState<number | null>(null);
  const pointClipsRef = useRef<HTMLDivElement>(null);

  const fetchStatus = useCallback(async () => {
    try {
      const s = await getStatus(jobId);
      setStatus(s);
      if (s.status === "complete" || s.status === "error") {
        if (s.status === "complete") {
          const d = await getResultsData(jobId);
          setData(d);
          try {
            const nr = await getCoachNotes(jobId);
            setNotes(nr.notes);
          } catch { /* no notes yet */ }
        }
        return false;
      }
      return true;
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
      return false;
    }
  }, [jobId]);

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    let active = true;

    async function poll() {
      const shouldContinue = await fetchStatus();
      if (shouldContinue && active) {
        timer = setTimeout(poll, POLL_INTERVAL);
      }
    }

    poll();
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [fetchStatus]);

  // Scroll to point clips and set focus index when a user clicks an evidence/momentum link
  const handlePointFocus = useCallback((idx: number) => {
    setFocusPointIdx(idx);
    setTimeout(() => {
      pointClipsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 50);
  }, []);

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center p-8">
        <div className="bg-red-950 border border-red-800 rounded-2xl p-8 max-w-md text-center">
          <h2 className="text-xl font-bold text-white mb-2">Something went wrong</h2>
          <p className="text-red-300 text-sm">{error}</p>
          <Link
            href="/"
            className="mt-6 inline-block px-6 py-3 bg-zinc-800 hover:bg-zinc-700 rounded-xl text-white text-sm transition-colors"
          >
            Start over
          </Link>
        </div>
      </div>
    );
  }

  if (!status) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-zinc-500 text-sm">Loading pipeline status...</div>
      </div>
    );
  }

  const isComplete = status.status === "complete" && data !== null;
  const isReview = status.status === "awaiting_point_review";
  const isError = status.status === "error";
  const isRunning = !isComplete && !isReview && !isError;

  const points: DetectedPoint[] = data?.points ?? [];
  const clipBaseUrl = `${API_URL}/outputs/${jobId}/clips`;

  const rawVideoUrl = data?.raw_video_url
    ? (data.raw_video_url.startsWith("/") ? `${API_URL}${data.raw_video_url}` : data.raw_video_url)
    : undefined;
  const overlayVideoUrl = data?.overlay_video_url
    ? (data.overlay_video_url.startsWith("/") ? `${API_URL}${data.overlay_video_url}` : data.overlay_video_url)
    : undefined;

  const playerACard: PlayerCard | null = data?.player_a_card ?? null;
  const playerBCard: PlayerCard | null = data?.player_b_card ?? null;
  const analyticsData: AnalyticsData | null = data?.analytics ?? null;
  const analysisBundle: AnalysisData | null = data?.analysis ?? null;
  const historicalInsights = data?.historical_insights ?? null;

  return (
    <div className="min-h-screen bg-black text-white">
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="border-b border-zinc-800 px-6 py-4 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-xl bg-green-600 flex items-center justify-center text-sm font-black text-white">
            T
          </div>
          <span className="text-white font-bold">
            Tennis<span className="text-green-400">IQ</span>
          </span>
        </Link>
        <div className="flex items-center gap-2 text-sm">
          <StatusBadge status={status.status} />
          <span className="text-zinc-500 font-mono text-xs">{jobId.slice(0, 8)}</span>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 py-8 space-y-10">
        {/* Error banner */}
        {isError && (
          <div className="bg-red-950 border border-red-800 rounded-2xl p-6">
            <h2 className="text-red-300 font-bold mb-2">Pipeline Error</h2>
            <p className="text-red-400 text-sm">{status.error_message || "An unexpected error occurred."}</p>
          </div>
        )}

        {isRunning && <ProgressTracker status={status} />}
        {isReview && <CheckpointReview jobId={jobId} onComplete={fetchStatus} />}

        {/* ── 1. Full Match Video: Raw + CV Overlay side by side (Top) ─────── */}
        {isComplete && (rawVideoUrl || overlayVideoUrl) && (
          <div className="space-y-3">
            <h3 className="text-base font-semibold text-white px-1">Full Match</h3>
            <div
              className={`grid gap-4 ${rawVideoUrl && overlayVideoUrl ? "grid-cols-1 lg:grid-cols-2" : "grid-cols-1"}`}
            >
              {rawVideoUrl && (
                <div className="space-y-1.5">
                  <span className="text-xs font-medium text-zinc-500 uppercase tracking-wider px-1">
                    Raw Footage
                  </span>
                  <div className="rounded-xl overflow-hidden bg-zinc-900 border border-zinc-800">
                    <video src={rawVideoUrl} className="w-full aspect-video" controls playsInline>
                      <track kind="captions" />
                    </video>
                  </div>
                </div>
              )}
              {overlayVideoUrl && (
                <div className="space-y-1.5">
                  <span className="text-xs font-medium text-zinc-500 uppercase tracking-wider px-1">
                    CV Detection Overlay
                  </span>
                  <div className="rounded-xl overflow-hidden bg-zinc-900 border border-green-500/20">
                    <video src={overlayVideoUrl} className="w-full aspect-video" controls playsInline>
                      <track kind="captions" />
                    </video>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── 2. Player Cards ─────────────────────────────────────────────── */}
        {isComplete && (playerACard || playerBCard) && (
          <PlayerCardView
            playerACard={playerACard}
            playerBCard={playerBCard}
            analytics={analyticsData}
          />
        )}

        {isComplete && <HistoricalInsightsCard history={historicalInsights} />}

        {/* ── 3. Weakness Report (clickable evidence → jump to clip) ───────── */}
        {isComplete && (playerACard || playerBCard) && points.length > 0 && (
          <WeaknessReport
            playerACard={playerACard}
            playerBCard={playerBCard}
            points={points}
            onEvidenceClick={handlePointFocus}
          />
        )}

        {/* ── 4. Match Trends (rally length + momentum, momentum is clickable) */}
        {isComplete && points.length > 0 && (
          <MatchCharts
            points={points}
            analytics={analyticsData}
            onPointClick={handlePointFocus}
          />
        )}

        {/* ── 5. Point Clips + Coach Notes ────────────────────────────────── */}
        {isComplete && points.length > 0 && (
          <div ref={pointClipsRef} id="point-clips-section" className="scroll-mt-6">
            <PointClipViewer
              jobId={jobId}
              points={points}
              clipBaseUrl={clipBaseUrl}
              rawVideoUrl={rawVideoUrl}
              initialPointIdx={focusPointIdx}
            />
          </div>
        )}

        {/* ── 6. Export Report (PDF) ──────────────────────────────────────── */}
        {isComplete && (
          <div className="border-t border-zinc-800 pt-8">
            <div className="flex flex-col items-center gap-2 text-center mb-4">
              <h3 className="text-base font-semibold text-white">Export Coach Report</h3>
              <p className="text-xs text-zinc-500">
                Full point breakdown, weakness analysis, and coach notes — formatted for players to take to practice.
              </p>
            </div>
            <div className="flex justify-center">
              <PDFExport
                playerACard={playerACard}
                playerBCard={playerBCard}
                analytics={analyticsData}
                analysis={analysisBundle}
                notes={notes}
                points={points}
                shots={data?.shots ?? []}
                coachingCards={data?.coaching_cards ?? []}
                jobId={jobId}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    queued: "bg-zinc-800 text-zinc-400",
    running: "bg-blue-950 text-blue-400",
    awaiting_point_review: "bg-yellow-950 text-yellow-400",
    finalizing: "bg-blue-950 text-blue-400",
    complete: "bg-green-950 text-green-400",
    error: "bg-red-950 text-red-400",
  };
  return (
    <span className={`px-2 py-1 rounded-lg text-xs font-medium ${colors[status] ?? "bg-zinc-800 text-zinc-400"}`}>
      {STAGE_LABELS[status] ?? status}
    </span>
  );
}
