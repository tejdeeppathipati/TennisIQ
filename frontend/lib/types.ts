export type JobStatus =
  | "queued"
  | "running"
  | "awaiting_point_review"
  | "finalizing"
  | "complete"
  | "error";

export interface Decision {
  id: number;
  job_id: string;
  iteration: number;
  action: string;
  justification: string;
  created_at: string;
}

export interface EvalResult {
  id: number;
  job_id: string;
  iteration: number;
  player_map: number | null;
  ball_map: number | null;
  court_lines_map: number | null;
  net_map: number | null;
  fp_rate: number | null;
  generalization_score: number | null;
  frame_count: number;
  criteria_met: number;
  details: string;
  created_at: string;
}

export interface StatusResponse {
  job_id: string;
  status: JobStatus;
  stage: string;
  stage_description: string;
  iteration: number;
  error_message: string | null;
  decisions: Decision[];
  latest_eval: EvalResult | null;
  created_at: string;
  updated_at: string;
}

export interface Artifact {
  id: number;
  job_id: string;
  artifact_type: string;
  artifact_path: string;
  metadata: string;
  created_at: string;
}

export interface PointFeedbackEntry {
  id: number;
  job_id: string;
  point_idx: number;
  action: "confirm" | "flag";
  note: string | null;
  created_at: string;
}

export interface ResultsResponse {
  job_id: string;
  status: string;
  footage_url: string;
  artifacts: Record<string, Artifact[]>;
  eval_results: EvalResult[];
  decisions: Decision[];
  coach_feedback: unknown[];
  point_feedback: PointFeedbackEntry[];
  created_at: string;
  updated_at: string;
}

export interface DetectedPoint {
  point_idx: number;
  start_frame: number;
  end_frame: number;
  start_sec: number;
  end_sec: number;
  serve_frame: number | null;
  serve_player: string | null;
  first_bounce_frame: number | null;
  first_bounce_court_xy: [number, number] | null;
  serve_zone: string | null;
  serve_fault_type: string | null;
  end_reason: string;
  rally_hit_count: number;
  bounce_count: number;
  bounce_frames: number[];
  confidence: number;
  event_count: number;
  coach_action?: "confirm" | "flag" | null;
  coach_note?: string | null;
  excluded?: boolean;
}

export interface PointReviewResponse {
  job_id: string;
  status: string;
  points: DetectedPoint[];
  overlay_video_path: string | null;
  review_complete: boolean;
}

export interface CoachingCard {
  point_idx: number;
  summary: string;
  suggestion: string;
  start_sec: number;
  end_sec: number;
  rally_hit_count: number;
  bounce_count?: number;
  end_reason: string;
  serve_zone?: string | null;
  serve_fault_type?: string | null;
  confidence: number;
  shot_sequence?: ShotSequenceItem[];
  pattern_context?: string;
}

export interface ShotSequenceItem {
  owner: string;
  owner_short: string;
  shot_type: string;
  direction: string;
  speed_m_s: number | null;
  court_side: string | null;
}

export interface ShotEvent {
  frame_idx: number;
  timestamp_sec: number;
  owner: string;
  ball_court_xy: [number, number];
  shot_type: string | null;
  shot_type_confidence: number;
  ball_direction_deg: number | null;
  ball_direction_label: string;
  speed_m_s: number | null;
  court_side: string | null;
  ownership_method: string;
}

export interface PlayerCard {
  card: {
    label: string;
    exploit_plan?: string;
    tendencies: string[];
    serve_summary: string;
    shot_distribution_summary: string;
    coverage_summary: string;
  };
  weaknesses: {
    label: string;
    weaknesses: WeaknessItem[];
  };
}

export interface WeaknessItem {
  description: string;
  data_point: string;
  points_cost: number;
  severity: number;
}

export interface MatchFlowData {
  insights: MatchFlowInsight[];
}

export interface MatchFlowInsight {
  description: string;
  timestamp_range: [number, number] | null;
}

export interface AnalyticsData {
  player_a: PlayerAnalytics;
  player_b: PlayerAnalytics;
  rally_length_distribution: Record<string, number>;
  rally_length_avg: number;
  total_points: number;
  total_shots: number;
  momentum_data: MomentumPoint[];
  match_flow: MatchFlowPoint[];
  shot_pattern_dominance: Record<string, PatternItem[]>;
}

export interface PlayerAnalytics {
  label: string;
  total_shots: number;
  shot_type_counts: Record<string, number>;
  shot_type_pcts: Record<string, number>;
  shot_direction_counts: Record<string, Record<string, number>>;
  shot_direction_pcts: Record<string, Record<string, number>>;
  error_by_shot_type: Record<string, number>;
  error_rate_by_shot_type: Record<string, number>;
  error_by_rally_length: Record<string, number>;
  error_rate_by_rally_length: Record<string, number>;
  avg_shot_speed_m_s: number;
  total_distance_covered: number;
  center_of_gravity: [number, number];
  first_serve_pct: number;
  double_fault_count: number;
  serve_zone_win_rate: Record<string, number>;
  serve_placement_counts: Record<string, number>;
  points_won: number;
  points_lost: number;
}

export interface MomentumPoint {
  point_idx: number;
  timestamp_sec: number;
  winner: string;
  a_momentum: number;
  b_momentum: number;
  rally_length: number;
}

export interface MatchFlowPoint {
  point_idx: number;
  timestamp_sec: number;
  rally_length: number;
  end_reason: string;
  duration_sec: number;
}

export interface PatternItem {
  pattern: string;
  shot_type: string;
  direction: string;
  count: number;
  pct: number;
}

export interface ServePlacement {
  serves: {
    point_idx: number;
    court_x: number;
    court_y: number;
    serve_zone: string | null;
    is_fault: boolean;
    fault_type: string | null;
    serve_player: string | null;
  }[];
  service_boxes: Record<string, { x_min: number; y_min: number; x_max: number; y_max: number }>;
}

export interface StatSummary {
  mean?: number | null;
  median?: number | null;
  p90?: number | null;
  p95?: number | null;
  max?: number | null;
}

export interface AnalysisData {
  meta?: {
    job_id?: string | null;
    fps?: number | null;
    duration_sec?: number | null;
    meters_per_unit?: number | null;
    court?: { width_units: number; height_units: number };
  };
  quality?: {
    frames_total?: number;
    ball_coverage_pct?: number;
    ball_projected_pct?: number;
    homography_reliable_pct?: number;
    player_visibility?: {
      player_a_pct?: number;
      player_b_pct?: number;
      both_pct?: number;
    };
    events_total?: number;
    points_total?: number;
  };
  serve?: {
    zone_counts?: Record<string, number>;
    fault_rate?: number | null;
    depth_stats?: StatSummary | null;
    width_stats?: StatSummary | null;
    depth_samples_m?: number[];
    width_samples_m?: number[];
    sample_count?: number;
  };
  rally?: {
    rally_hits?: number[];
    rally_durations_sec?: number[];
    end_reason_counts?: Record<string, number>;
    tempo_stats?: {
      mean_hits_per_sec?: number | null;
      mean_inter_hit_sec?: number | null;
      p95_inter_hit_sec?: number | null;
    };
  };
  errors?: {
    out_count?: number;
    out_distance_stats?: StatSummary | null;
    error_positions?: { x: number | null; y: number | null }[];
  };
  players?: {
    player_a?: {
      distance_m?: number;
      speed_stats?: StatSummary | null;
      zone_time_pct?: Record<string, number> | null;
    } | null;
    player_b?: {
      distance_m?: number;
      speed_stats?: StatSummary | null;
      zone_time_pct?: Record<string, number> | null;
    } | null;
  };
  ball?: {
    speed_stats?: StatSummary | null;
    accel_stats?: { mean?: number | null; p95_abs?: number | null } | null;
    speed_samples_m_s?: number[];
    hit_speed_deltas?: { t?: number | null; before?: number | null; after?: number | null; delta?: number | null }[];
  };
  events?: {
    timeline?: {
      t?: number | null;
      type?: string | null;
      side?: string | null;
      in_out?: string | null;
      speed_before_m_s?: number | null;
      speed_after_m_s?: number | null;
      direction_change_deg?: number | null;
      player?: string | null;
    }[];
  };
}

export interface HeatmapData {
  grid: number[][];
  x_edges: number[];
  y_edges: number[];
  total_out_bounces?: number;
  total_frames?: number;
  positions?: { x: number; y: number }[];
}

export interface DownloadItem {
  label: string;
  href: string;
}

export interface ResultsDataResponse {
  job_id: string;
  status: string;
  footage_url: string | null;
  overlay_video_url: string | null;
  raw_video_url: string | null;
  points: DetectedPoint[];
  events: unknown[];
  coaching_cards: CoachingCard[];
  serve_placement: ServePlacement | null;
  error_heatmap: HeatmapData | null;
  player_a_heatmap: HeatmapData | null;
  player_b_heatmap: HeatmapData | null;
  analysis?: AnalysisData | null;
  stats: Record<string, unknown> | null;
  clips: { filename: string; url: string }[];
  downloads: DownloadItem[];
  point_feedback: PointFeedbackEntry[];
  shots: ShotEvent[];
  analytics: AnalyticsData | null;
  player_a_card: PlayerCard | null;
  player_b_card: PlayerCard | null;
  match_flow: MatchFlowData | null;
  historical_insights?: HistoricalInsights | null;
}

export interface PersistentWeaknessInsight {
  name: string;
  baseline_rate_pct: number;
  matches_triggered: number;
  matches_with_data: number;
  confirmed: boolean;
}

export interface PlayerHistoryInsights {
  match_count: number;
  baseline: {
    backhand_error_rate_pct: number | null;
    long_rally_error_rate_pct: number | null;
    first_serve_pct: number | null;
    avg_shot_speed_kmh: number | null;
  };
  current_vs_baseline: {
    backhand_error_delta_pp: number | null;
    long_rally_error_delta_pp: number | null;
    first_serve_delta_pp: number | null;
    avg_shot_speed_delta_kmh: number | null;
  };
  persistent_weaknesses: PersistentWeaknessInsight[];
  summary: string[];
}

export interface HistoricalInsights {
  enabled: boolean;
  minimum_matches: number;
  total_matches_considered: number;
  player_a: PlayerHistoryInsights;
  player_b: PlayerHistoryInsights;
}

export interface CoachNote {
  id: number;
  job_id: string;
  point_idx: number;
  timestamp_sec: number;
  note_text: string;
  player: "player_a" | "player_b";
  created_at: string;
}

export interface Session {
  id: string;
  coach_id: string;
  job_id: string;
  footage_url: string | null;
  footage_type: string | null;
  fps: number | null;
  frame_count: number | null;
  duration_sec: number | null;
  total_points: number;
  total_events: number;
  coach_feedback: PointFeedbackEntry[];
  detection_summary: Record<string, unknown>;
  preferences: Record<string, unknown>;
  created_at: string;
}

export const STAGE_LABELS: Record<string, string> = {
  queued: "Queued",
  downloading: "Preparing Video",
  segmenting: "Splitting into Segments",
  inference: "Running Analysis",
  court_detection: "Court Detection",
  homography: "Computing Homography",
  ball_detection: "Ball Detection",
  ball_physics: "Ball Physics",
  player_detection: "Player Detection",
  event_detection: "Detecting Events",
  shot_detection: "Detecting Shots & Ownership",
  shot_classification: "Classifying Shot Types",
  point_segmentation: "Segmenting Points",
  match_analytics: "Computing Match Analytics",
  generating_outputs: "Generating Outputs",
  overlay_video: "Rendering Overlay",
  clip_extraction: "Extracting Clips",
  awaiting_point_review: "Coach Review",
  finalizing: "Applying Feedback",
  complete: "Complete",
  error: "Error",
};

export const STAGE_ORDER = [
  "queued",
  "downloading",
  "segmenting",
  "inference",
  "court_detection",
  "homography",
  "ball_detection",
  "ball_physics",
  "player_detection",
  "event_detection",
  "shot_detection",
  "shot_classification",
  "point_segmentation",
  "match_analytics",
  "generating_outputs",
  "overlay_video",
  "clip_extraction",
  "awaiting_point_review",
  "finalizing",
  "complete",
];
