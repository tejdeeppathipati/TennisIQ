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
  bounce_count: number;
  end_reason: string;
  serve_zone: string | null;
  serve_fault_type: string | null;
  confidence: number;
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
  stats: Record<string, unknown> | null;
  clips: { filename: string; url: string }[];
  downloads: DownloadItem[];
  point_feedback: PointFeedbackEntry[];
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
  point_segmentation: "Segmenting Points",
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
  "point_segmentation",
  "generating_outputs",
  "overlay_video",
  "clip_extraction",
  "awaiting_point_review",
  "finalizing",
  "complete",
];
