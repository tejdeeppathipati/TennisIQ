"use client";

import { useEffect, useState, useCallback, use } from "react";
import Link from "next/link";
import { getStatus, getResultsData } from "@/lib/api";
import type {
  StatusResponse,
  ResultsDataResponse,
  DetectedPoint,
  CoachingCard,
  ServePlacement,
  HeatmapData,
  DownloadItem,
  PlayerCard,
  AnalyticsData,
  AnalysisData,
  MatchFlowData,
} from "@/lib/types";
import { STAGE_LABELS } from "@/lib/types";
import ProgressTracker from "@/components/ProgressTracker";
import CheckpointReview from "@/components/CheckpointReview";
import SideBySidePlayer from "@/components/SideBySidePlayer";
import PointTimeline from "@/components/PointTimeline";
import CoachingCards from "@/components/CoachingCards";
import PlayerCardView from "@/components/PlayerCardView";
import WeaknessReport from "@/components/WeaknessReport";
import MatchFlowChart from "@/components/MatchFlowChart";
import ServePlacementChart from "@/components/ServePlacementChart";
import HeatmapViewer from "@/components/HeatmapViewer";
import HighlightClips from "@/components/HighlightClips";
import DownloadPanel from "@/components/DownloadPanel";
import AnalysisDashboard from "@/components/AnalysisDashboard";

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
  const [seekTo, setSeekTo] = useState<number | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = useCallback(async () => {
    try {
      const s = await getStatus(jobId);
      setStatus(s);
      if (s.status === "complete" || s.status === "error") {
        if (s.status === "complete") {
          const d = await getResultsData(jobId);
          setData(d);
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

  const handleSeekToPoint = (sec: number) => {
    setSeekTo(sec);
    setTimeout(() => setSeekTo(null), 100);
  };

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

  const overlayVideoUrl = data?.overlay_video_url
    ? `${API_URL}${data.overlay_video_url}`
    : undefined;
  const rawVideoUrl = data?.raw_video_url
    ? (data.raw_video_url.startsWith("/") ? `${API_URL}${data.raw_video_url}` : data.raw_video_url)
    : undefined;

  const points: DetectedPoint[] = data?.points ?? [];
  const coachingCards: CoachingCard[] = data?.coaching_cards ?? [];
  const servePlacement: ServePlacement | null = data?.serve_placement ?? null;
  const errorHeatmap: HeatmapData | null = data?.error_heatmap ?? null;
  const playerAHeatmap: HeatmapData | null = data?.player_a_heatmap ?? null;
  const playerBHeatmap: HeatmapData | null = data?.player_b_heatmap ?? null;
  const downloads: DownloadItem[] = (data?.downloads ?? []).map((d) => ({
    ...d,
    href: `${API_URL}${d.href}`,
  }));
  const clipBaseUrl = `${API_URL}/outputs/${jobId}/clips`;

  const playerACard: PlayerCard | null = data?.player_a_card ?? null;
  const playerBCard: PlayerCard | null = data?.player_b_card ?? null;
  const analyticsData: AnalyticsData | null = data?.analytics ?? null;
  const matchFlow: MatchFlowData | null = data?.match_flow ?? null;
  const analysisBundle: AnalysisData | null = data?.analysis ?? null;

  return (
    <div className="min-h-screen bg-black text-white">
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

      <div className="max-w-7xl mx-auto px-4 py-8 space-y-8">

        {isError && (
          <div className="bg-red-950 border border-red-800 rounded-2xl p-6">
            <h2 className="text-red-300 font-bold mb-2">Pipeline Error</h2>
            <p className="text-red-400 text-sm">{status.error_message || "An unexpected error occurred."}</p>
          </div>
        )}

        {isRunning && <ProgressTracker status={status} />}

        {isReview && <CheckpointReview jobId={jobId} onComplete={fetchStatus} />}

        {/* Priority 1: Player Cards (the real product) */}
        {isComplete && (playerACard || playerBCard) && (
          <PlayerCardView
            playerACard={playerACard}
            playerBCard={playerBCard}
            analytics={analyticsData}
          />
        )}

        {/* Priority 2: Weakness Reports */}
        {isComplete && (playerACard || playerBCard) && (
          <WeaknessReport
            playerACard={playerACard}
            playerBCard={playerBCard}
          />
        )}

        {/* Priority 3: Match Flow */}
        {isComplete && (analyticsData || matchFlow) && (
          <MatchFlowChart
            analytics={analyticsData}
            matchFlow={matchFlow}
          />
        )}

        {/* Priority 3.5: Analysis Dashboard (quality, serve, rally, errors charts) */}
        {isComplete && analysisBundle && (
          <AnalysisDashboard analysis={analysisBundle} />
        )}

        {/* Priority 4: Enhanced Coaching Cards */}
        {isComplete && coachingCards.length > 0 && (
          <CoachingCards
            cards={coachingCards}
            onSeekToPoint={handleSeekToPoint}
          />
        )}

        {/* Demo features: Video, Timeline, Visualizations */}
        {isComplete && (
          <SideBySidePlayer
            rawVideoUrl={rawVideoUrl}
            overlayVideoUrl={overlayVideoUrl}
            onTimeUpdate={setCurrentTime}
            seekTo={seekTo}
          />
        )}

        {isComplete && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 space-y-6">
              <PointTimeline
                points={points}
                currentTime={currentTime}
                onSeekToPoint={handleSeekToPoint}
              />

              <HighlightClips
                points={points}
                clipBaseUrl={clipBaseUrl}
              />
            </div>

            <div className="space-y-6">
              <ServePlacementChart data={servePlacement} />

              <HeatmapViewer
                errorHeatmap={errorHeatmap}
                playerAHeatmap={playerAHeatmap}
                playerBHeatmap={playerBHeatmap}
              />

              <DownloadPanel items={downloads} />

              {data?.stats && (
                <div className="rounded-xl bg-zinc-900 border border-zinc-800 p-4 space-y-3">
                  <h3 className="text-sm font-semibold text-zinc-300">Detection Stats</h3>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="text-zinc-500">Points detected</div>
                    <div className="text-zinc-300">{points.length}</div>
                    <div className="text-zinc-500">Total events</div>
                    <div className="text-zinc-300">
                      {(data.stats as Record<string, unknown>)?.events
                        ? ((data.stats as Record<string, Record<string, number>>).events?.total ?? 0)
                        : data.events?.length ?? 0}
                    </div>
                    <div className="text-zinc-500">Total shots</div>
                    <div className="text-zinc-300">{data.shots?.length ?? 0}</div>
                    <div className="text-zinc-500">Avg rally hits</div>
                    <div className="text-zinc-300">
                      {(data.stats as Record<string, Record<string, number>>)?.points?.avg_rally_hits?.toFixed(1) ?? "-"}
                    </div>
                  </div>
                </div>
              )}
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
    <span
      className={`px-2 py-1 rounded-lg text-xs font-medium ${colors[status] ?? "bg-zinc-800 text-zinc-400"}`}
    >
      {STAGE_LABELS[status] ?? status}
    </span>
  );
}
