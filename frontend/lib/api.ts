import type {
  StatusResponse,
  ResultsResponse,
  ResultsDataResponse,
  PointReviewResponse,
  Session,
} from "./types";

const ENV_API_URL = (process.env.NEXT_PUBLIC_API_URL || "").trim().replace(/\/+$/, "");
const API_BASE = ENV_API_URL || "/backend";

export async function getApiBaseUrl(): Promise<string> {
  return API_BASE;
}

function ensureAbsoluteUrl(baseUrl: string, path: string): string {
  if (/^https?:\/\//i.test(path)) {
    return path;
  }
  const safePath = path.startsWith("/") ? path : `/${path}`;
  return `${baseUrl}${safePath}`;
}

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const apiUrl = API_BASE;
  let res: Response;
  try {
    res = await fetch(ensureAbsoluteUrl(apiUrl, path), {
      headers: { "Content-Type": "application/json", ...options?.headers },
      ...options,
    });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `Cannot reach backend via ${apiUrl}. Start backend with: cd backend && python -m uvicorn main:app --host 127.0.0.1 --port 8002 (${message})`,
    );
  }
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export async function ingestURL(url: string): Promise<{ job_id: string; status: string }> {
  return apiFetch("/ingest", {
    method: "POST",
    body: JSON.stringify({ url }),
  });
}

export async function ingestUpload(file: File): Promise<{ job_id: string; status: string }> {
  const formData = new FormData();
  formData.append("file", file);
  const apiUrl = API_BASE;
  let res: Response;
  try {
    res = await fetch(ensureAbsoluteUrl(apiUrl, "/ingest/upload"), {
      method: "POST",
      body: formData,
    });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `Cannot reach backend via ${apiUrl}. Start backend with: cd backend && python -m uvicorn main:app --host 127.0.0.1 --port 8002 (${message})`,
    );
  }
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Upload error ${res.status}: ${text}`);
  }
  return res.json();
}

export async function getStatus(jobId: string): Promise<StatusResponse> {
  return apiFetch(`/status/${jobId}`);
}

export async function getResults(jobId: string): Promise<ResultsResponse> {
  return apiFetch(`/results/${jobId}`);
}

export async function getResultsData(jobId: string): Promise<ResultsDataResponse> {
  return apiFetch(`/results/${jobId}/data`);
}

export async function getPointsForReview(jobId: string): Promise<PointReviewResponse> {
  return apiFetch(`/checkpoint/${jobId}/points`);
}

export async function submitPointFeedback(
  jobId: string,
  pointIdx: number,
  action: "confirm" | "flag",
  note?: string,
): Promise<{ ok: boolean }> {
  return apiFetch(`/checkpoint/${jobId}/points/${pointIdx}/feedback`, {
    method: "POST",
    body: JSON.stringify({ action, note }),
  });
}

export async function finalizeReview(
  jobId: string,
): Promise<{ ok: boolean; total_points: number; confirmed: number; flagged: number; excluded: number }> {
  return apiFetch(`/checkpoint/${jobId}/finalize`, { method: "POST" });
}

export async function saveSession(
  jobId: string,
  coachId?: string,
  preferences?: Record<string, unknown>,
): Promise<{ ok: boolean; session_id: string }> {
  return apiFetch(`/sessions/${jobId}/save`, {
    method: "POST",
    body: JSON.stringify({ coach_id: coachId || "default", preferences }),
  });
}

export async function listSessions(coachId?: string): Promise<{ coach_id: string; sessions: Session[] }> {
  const cid = coachId || "default";
  return apiFetch(`/sessions?coach_id=${encodeURIComponent(cid)}`);
}

export async function getSession(sessionId: string): Promise<Session> {
  return apiFetch(`/sessions/${sessionId}`);
}

export async function getLatestSession(coachId?: string): Promise<Session> {
  const cid = coachId || "default";
  return apiFetch(`/sessions/latest/${encodeURIComponent(cid)}`);
}
