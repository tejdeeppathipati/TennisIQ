"use client";

import { useState } from "react";
import { getCoachNotes } from "@/lib/api";
import type {
  PlayerCard, AnalyticsData, AnalysisData, CoachNote,
  DetectedPoint, ShotEvent, CoachingCard,
} from "@/lib/types";

interface Props {
  playerACard: PlayerCard | null;
  playerBCard: PlayerCard | null;
  analytics: AnalyticsData | null;
  analysis: AnalysisData | null;
  notes: CoachNote[];
  points: DetectedPoint[];
  shots: ShotEvent[];
  coachingCards: CoachingCard[];
  jobId: string;
}

// ─── Text helpers ─────────────────────────────────────────────────────────────

function clean(s: string): string {
  return s
    .replace(/[\u2018\u2019\u201A]/g, "'")
    .replace(/[\u201C\u201D\u201E]/g, '"')
    .replace(/[\u2013\u2014]/g, "-")
    .replace(/\u2212/g, "-")
    .replace(/\u2026/g, "...")
    .replace(/[\u2022\u00B7]/g, "-")
    .replace(/[^\x20-\x7E\n\r\t]/g, "")
    .replace(/\s{2,}/g, " ")
    .trim();
}

function pct(n: number): string { return `${n.toFixed(0)}%`; }
function kmh(ms: number | null | undefined): string {
  return ms != null ? `${Math.round(ms * 3.6)} km/h` : "--";
}
function fmtNum(v: number | null | undefined, d = 1): string {
  return v == null ? "--" : v.toFixed(d);
}

// ─── Coaching narrative generators ────────────────────────────────────────────

function buildMatchStory(
  card: PlayerCard | null,
  pa: ReturnType<typeof getPlayerAnalytics>,
  label: string,
): string {
  if (!pa && !card) return `No sufficient data to build a match story for ${label}.`;

  const lines: string[] = [];

  // Dominant shot type
  if (pa && Object.keys(pa.shot_type_counts).length > 0) {
    const sorted = Object.entries(pa.shot_type_counts).sort(([, a], [, b]) => b - a);
    const top = sorted[0];
    const topTotal = Object.values(pa.shot_type_counts).reduce((a, b) => a + b, 0);
    const topPct = topTotal > 0 ? (top[1] / topTotal) * 100 : 0;
    if (topPct > 30) {
      lines.push(`${label} is a ${top[0]}-dominant player (${pct(topPct)} of shots).`);
    }
  }

  // Biggest weakness in plain English
  if (card?.weaknesses?.weaknesses?.length) {
    const worst = card.weaknesses.weaknesses[0];
    lines.push(worst.description);
  }

  // Long rally problem
  if (pa) {
    const longRate = pa.error_rate_by_rally_length?.["7-9"] ?? pa.error_rate_by_rally_length?.["10+"];
    if (longRate != null && longRate > 40) {
      lines.push(`Breaks down in extended exchanges - error rate climbs to ${pct(longRate)} in rallies longer than 7 shots.`);
    }
  }

  // Points record
  if (pa) {
    const total = pa.points_won + pa.points_lost;
    if (total > 0) {
      lines.push(`Won ${pa.points_won} of ${total} points in this match.`);
    }
  }

  return lines.join(" ") || `${label} analysis complete.`;
}

// Generate a unique, specific sentence for each point - no boilerplate repeats
function buildPointSentence(
  pt: DetectedPoint,
  cc: CoachingCard | undefined,
  playerShots: ShotEvent[],
  pa: ReturnType<typeof getPlayerAnalytics>,
  playerLabel: string,
  seenPatterns: Set<string>,
): string {
  const duration = (pt.end_sec - pt.start_sec).toFixed(1);
  const hits = pt.rally_hit_count;

  // Find the decisive last shot
  const lastShot = playerShots.length > 0 ? playerShots[playerShots.length - 1] : null;
  const shotType = lastShot?.shot_type ?? cc?.shot_sequence?.[cc.shot_sequence.length - 1]?.shot_type ?? null;
  const dir = lastShot?.ball_direction_label ?? cc?.shot_sequence?.[cc.shot_sequence.length - 1]?.direction?.replace(/_/g, " ") ?? null;
  const speed = lastShot?.speed_m_s ?? null;

  // Serve fault - specific and unique
  if (pt.serve_fault_type) {
    const faultType = pt.serve_fault_type.replace(/_/g, " ");
    if (faultType.includes("long")) return `Serve went long. ${hits} shot rally - point never got started.`;
    if (faultType.includes("net")) return `Serve clipped the net. ${hits} shot rally.`;
    return `Serve fault (${faultType}). ${hits} shots played.`;
  }

  // Outcome-specific sentences, varied by shot data
  if (pt.end_reason === "OUT") {
    if (shotType && dir && speed != null) {
      const speedStr = Math.round(speed * 3.6);
      const isOverhit = pa?.avg_shot_speed_m_s != null && speed > pa.avg_shot_speed_m_s * 1.2;
      if (isOverhit) {
        return `${playerLabel} went for too much on a ${shotType} ${dir} at ${speedStr} km/h - ${Math.round((speed / pa!.avg_shot_speed_m_s - 1) * 100)}% above their average pace. Missed long after ${hits} hits.`;
      }
      return `${playerLabel} missed a ${shotType} ${dir} (${speedStr} km/h) after ${hits} hits. ${duration}s rally.`;
    }
    if (shotType) {
      return `${shotType} error ended the point after ${hits} hits.`;
    }
    return `Unforced error - ball went out after ${hits} hits over ${duration}s.`;
  }

  if (pt.end_reason === "NET") {
    if (shotType && dir) {
      return `${playerLabel}'s ${shotType} ${dir} found the net after ${hits} hits. ${duration}s rally.`;
    }
    return `Ball into the net ended a ${hits}-hit, ${duration}s rally.`;
  }

  if (pt.end_reason === "WINNER") {
    if (shotType && dir && speed != null) {
      return `${playerLabel} put away a ${shotType} ${dir} winner at ${Math.round(speed * 3.6)} km/h. ${hits} hits, ${duration}s.`;
    }
    if (shotType) return `${playerLabel} hit a ${shotType} winner to end the ${hits}-hit rally.`;
    return `Winner - rally finished in ${playerLabel}'s favor after ${hits} hits.`;
  }

  if (pt.end_reason === "DOUBLE_BOUNCE") {
    return `Opponent couldn't reach the ball - it bounced twice after ${hits} hits.`;
  }

  if (pt.end_reason === "BALL_LOST") {
    // Only report the long-rally pattern ONCE in seenPatterns
    const longRate = pa?.error_rate_by_rally_length?.["7-9"] ?? pa?.error_rate_by_rally_length?.["10+"];
    if (hits >= 7 && longRate != null && longRate > 40 && !seenPatterns.has("long_rally_error")) {
      seenPatterns.add("long_rally_error");
      return `${hits}-hit rally - tracking was lost, but this is ${playerLabel}'s danger zone. Their error rate rises to ${pct(longRate)} in rallies this long.`;
    }
    if (shotType) {
      return `Tracking lost during a ${shotType} exchange. ${hits} hits, ${duration}s.`;
    }
    return `Ball tracking lost after ${hits} hits. ${duration}s rally.`;
  }

  // Fallback using coaching card summary if unique
  if (cc?.summary) {
    const key = cc.summary.slice(0, 40);
    if (!seenPatterns.has(key)) {
      seenPatterns.add(key);
      return clean(cc.summary);
    }
  }

  return `${hits}-hit rally over ${duration}s.`;
}

function getPlayerAnalytics(analytics: AnalyticsData | null, playerKey: "player_a" | "player_b") {
  if (!analytics) return null;
  return playerKey === "player_a" ? analytics.player_a : analytics.player_b;
}

// ─── Chart drawing primitives ─────────────────────────────────────────────────

type JsPDFDoc = InstanceType<Awaited<ReturnType<typeof import("jspdf")>>["jsPDF"]>;

const COLORS = [
  [41, 128, 185], [231, 76, 60], [46, 204, 113],
  [243, 156, 18], [155, 89, 182], [26, 188, 156],
] as const;

function drawBars(
  doc: JsPDFDoc,
  x: number, y: number, w: number,
  entries: { label: string; pct: number }[],
  title: string,
): number {
  if (entries.length === 0) return y;
  if (title) {
    doc.setFontSize(9);
    doc.setFont("helvetica", "bold");
    doc.setTextColor(0);
    doc.text(clean(title), x, y);
    y += 5;
  }
  const labelW = 36;
  const barW = w - labelW - 14;
  const barH = 5;
  const gap = 2.5;
  for (let i = 0; i < entries.length; i++) {
    const e = entries[i];
    const c = COLORS[i % COLORS.length];
    doc.setFontSize(7.5);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(60);
    doc.text(clean(e.label), x, y + barH * 0.72);
    doc.setFillColor(232, 232, 232);
    doc.roundedRect(x + labelW, y, barW, barH, 1, 1, "F");
    const fill = Math.max((e.pct / 100) * barW, 1);
    doc.setFillColor(c[0], c[1], c[2]);
    doc.roundedRect(x + labelW, y, fill, barH, 1, 1, "F");
    doc.setFontSize(7);
    doc.setTextColor(0);
    doc.text(`${e.pct.toFixed(0)}%`, x + labelW + barW + 2, y + barH * 0.72);
    y += barH + gap;
  }
  return y + 2;
}

function drawSeverityBar(
  doc: JsPDFDoc, x: number, y: number, w: number,
  severity: number, pointsCost: number,
): number {
  const bw = w * 0.55;
  doc.setFillColor(225, 225, 225);
  doc.roundedRect(x, y, bw, 3, 1, 1, "F");
  const fill = Math.max(severity * bw, 2);
  if (severity > 0.6) doc.setFillColor(231, 76, 60);
  else if (severity > 0.3) doc.setFillColor(243, 156, 18);
  else doc.setFillColor(241, 196, 15);
  doc.roundedRect(x, y, fill, 3, 1, 1, "F");
  if (pointsCost > 0) {
    doc.setFontSize(7);
    doc.setTextColor(200, 50, 50);
    doc.text(clean(`-${pointsCost} pts`), x + bw + 3, y + 2.2);
    doc.setTextColor(0);
  }
  return y + 5;
}

function drawPie(
  doc: JsPDFDoc,
  cx: number, cy: number, r: number,
  entries: { label: string; pct: number }[],
): void {
  let a = -Math.PI / 2;
  for (let i = 0; i < entries.length; i++) {
    const e = entries[i];
    if (e.pct <= 0) continue;
    const sweep = (e.pct / 100) * 2 * Math.PI;
    const c = COLORS[i % COLORS.length];
    doc.setFillColor(c[0], c[1], c[2]);
    const steps = Math.max(Math.ceil(sweep / 0.08), 4);
    for (let s = 0; s < steps; s++) {
      const a1 = a + (sweep * s) / steps;
      const a2 = a + (sweep * (s + 1)) / steps;
      doc.triangle(cx, cy, cx + r * Math.cos(a1), cy + r * Math.sin(a1), cx + r * Math.cos(a2), cy + r * Math.sin(a2), "F");
    }
    a += sweep;
  }
  // Legend
  let ly = cy - r;
  const lx = cx + r + 5;
  for (let i = 0; i < entries.length; i++) {
    const e = entries[i];
    if (e.pct <= 0) continue;
    const c = COLORS[i % COLORS.length];
    doc.setFillColor(c[0], c[1], c[2]);
    doc.rect(lx, ly, 3, 3, "F");
    doc.setFontSize(7);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(60);
    doc.text(clean(`${e.label} ${e.pct.toFixed(0)}%`), lx + 4.5, ly + 2.3);
    ly += 4.5;
  }
}

// ─── Main PDF generator ───────────────────────────────────────────────────────

async function generatePlayerPDF(
  card: PlayerCard | null,
  playerKey: "player_a" | "player_b",
  label: string,
  analytics: AnalyticsData | null,
  analysis: AnalysisData | null,
  notes: CoachNote[],
  points: DetectedPoint[],
  shots: ShotEvent[],
  coachingCards: CoachingCard[],
  jobId: string,
) {
  const { jsPDF } = await import("jspdf");
  const doc = new jsPDF({ orientation: "portrait", unit: "mm", format: "a4" });
  const PW = doc.internal.pageSize.getWidth();
  const M = 15;
  const CW = PW - M * 2;
  const pa = getPlayerAnalytics(analytics, playerKey);

  // Layout helpers
  function space(y: number, needed: number): number {
    if (y + needed > 280) { doc.addPage(); return M; }
    return y;
  }

  function rule(y: number, alpha = 180): number {
    doc.setDrawColor(alpha);
    doc.setLineWidth(0.25);
    doc.line(M, y, M + CW, y);
    return y + 3;
  }

  function h1(text: string, y: number): number {
    y = space(y, 14);
    doc.setFontSize(13);
    doc.setFont("helvetica", "bold");
    doc.setTextColor(0);
    doc.text(clean(text), M, y);
    return rule(y + 2, 160);
  }

  function h2(text: string, y: number): number {
    y = space(y, 10);
    doc.setFontSize(10);
    doc.setFont("helvetica", "bold");
    doc.setTextColor(40);
    doc.text(clean(text), M, y);
    return y + 5;
  }

  function body(text: string, y: number, indent = 0): number {
    doc.setFontSize(9);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(40);
    const lines = doc.splitTextToSize(clean(text), CW - indent);
    y = space(y, lines.length * 4.3);
    doc.text(lines, M + indent, y);
    return y + lines.length * 4.3 + 0.5;
  }

  function muted(text: string, y: number, indent = 0): number {
    doc.setFontSize(8);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(130);
    const lines = doc.splitTextToSize(clean(text), CW - indent);
    y = space(y, lines.length * 3.8);
    doc.text(lines, M + indent, y);
    doc.setTextColor(0);
    return y + lines.length * 3.8 + 0.5;
  }

  function indent_bullet(text: string, y: number, indentX = 4): number {
    doc.setFontSize(8.5);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(60);
    const lines = doc.splitTextToSize(clean(text), CW - indentX - 2);
    y = space(y, lines.length * 4);
    doc.setFillColor(100);
    doc.circle(M + 1.5, y - 0.8, 0.5, "F");
    doc.text(lines, M + indentX, y);
    return y + lines.length * 4 + 0.5;
  }

  let y = M;

  // ═════════════════════════════════════════════════════════
  // PAGE 1 — TITLE + STORY + TOP 3 FIXES
  // ═════════════════════════════════════════════════════════

  // Title block
  doc.setFontSize(22);
  doc.setFont("helvetica", "bold");
  doc.setTextColor(0);
  doc.text(clean(`${label.toUpperCase()} - MATCH REPORT`), M, y);
  y += 7;
  doc.setFontSize(8);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(150);
  const dateStr = new Date().toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" });
  doc.text(clean(`TennisIQ  |  ${dateStr}  |  Job: ${jobId.slice(0, 8)}`), M, y);
  y += 2;
  doc.setDrawColor(0);
  doc.setLineWidth(0.7);
  doc.line(M, y, M + CW, y);
  y += 4;
  doc.setTextColor(0);

  // Stats summary bar
  if (pa) {
    doc.setFontSize(9.5);
    doc.setFont("helvetica", "bold");
    doc.setTextColor(0);
    const total = pa.points_won + pa.points_lost;
    const statLine = clean(
      `${pa.total_shots} shots   ${kmh(pa.avg_shot_speed_m_s)} avg pace   ${pa.points_won}W / ${pa.points_lost}L${total > 0 ? ` (${pct((pa.points_won / total) * 100)} win rate)` : ""}`
    );
    doc.text(statLine, M, y);
    y += 7;
  }

  // ── THE STORY OF THIS MATCH ──
  y = h1("THE STORY OF THIS MATCH", y);
  const story = buildMatchStory(card, pa, label);
  y = body(story, y);
  y += 4;

  // ── TOP 3 THINGS TO WORK ON ──
  const weaknesses = card?.weaknesses?.weaknesses?.slice(0, 3) ?? [];
  if (weaknesses.length > 0) {
    y = h1("TOP THINGS TO WORK ON", y);
    for (let i = 0; i < weaknesses.length; i++) {
      const w = weaknesses[i];
      y = space(y, 22);

      // Number + description on same line
      doc.setFontSize(10);
      doc.setFont("helvetica", "bold");
      doc.setTextColor(0);
      doc.text(clean(`${i + 1}.`), M, y);
      const descLines = doc.splitTextToSize(clean(w.description), CW - 7);
      doc.setFont("helvetica", "normal");
      doc.setTextColor(30);
      doc.text(descLines, M + 7, y);
      y += descLines.length * 4.5;

      // Severity bar
      y = drawSeverityBar(doc, M + 7, y, CW - 7, w.severity, w.points_cost);

      // Data point as arrow indent
      if (w.data_point) {
        y = indent_bullet(`-> ${w.data_point}`, y, 7);
      }

      // Extra context from analytics
      if (pa) {
        const extraLines: string[] = [];
        const desc = w.description.toLowerCase();

        if (desc.includes("backhand")) {
          const errRate = pa.error_rate_by_shot_type?.["backhand"];
          const dirData = pa.shot_direction_counts?.["backhand"];
          if (errRate != null) extraLines.push(`-> ${pct(errRate)} error rate on backhands overall`);
          if (dirData) {
            const total = Object.values(dirData).reduce((a, b) => a + b, 0);
            const top = Object.entries(dirData).sort(([, a], [, b]) => b - a)[0];
            if (top && total > 0) extraLines.push(`-> Goes ${top[0].replace(/_/g, " ")} ${pct((top[1] / total) * 100)} of the time - predictable`);
          }
        }
        if (desc.includes("forehand")) {
          const errRate = pa.error_rate_by_shot_type?.["forehand"];
          if (errRate != null) extraLines.push(`-> ${pct(errRate)} error rate on forehands overall`);
        }
        if (desc.includes("rally") || desc.includes("long")) {
          const longRate = pa.error_rate_by_rally_length?.["7-9"] ?? pa.error_rate_by_rally_length?.["10+"];
          if (longRate != null) extraLines.push(`-> ${pct(longRate)} error rate in rallies longer than 7 shots`);
          const shortRate = pa.error_rate_by_rally_length?.["1-3"] ?? pa.error_rate_by_rally_length?.["1-6"];
          if (shortRate != null) extraLines.push(`-> Only ${pct(shortRate)} error rate in short rallies - big contrast`);
        }
        if (desc.includes("serve")) {
          if (pa.first_serve_pct > 0) extraLines.push(`-> First serve in: ${pct(pa.first_serve_pct)}`);
          if (pa.double_fault_count > 0) extraLines.push(`-> Double faults this match: ${pa.double_fault_count}`);
        }

        for (const line of extraLines) {
          y = indent_bullet(line, y, 7);
        }
      }
      y += 3;
    }
  }

  // ── SHOT DISTRIBUTION (pie + bars on same page) ──
  if (pa && Object.keys(pa.shot_type_counts).length > 0) {
    const totalShots = Object.values(pa.shot_type_counts).reduce((a, b) => a + b, 0);
    const shotEntries = Object.entries(pa.shot_type_counts)
      .sort(([, a], [, b]) => b - a)
      .map(([type, count]) => ({
        label: type.charAt(0).toUpperCase() + type.slice(1),
        pct: totalShots > 0 ? (count / totalShots) * 100 : 0,
      }));

    y = space(y, 42);
    y = h1("SHOT DISTRIBUTION", y);

    const pieR = 14;
    const pieCX = M + pieR + 2;
    const pieCY = y + pieR;
    drawPie(doc, pieCX, pieCY, pieR, shotEntries);

    const barX = M + pieR * 2 + 40;
    const barW = CW - (pieR * 2 + 42);
    drawBars(doc, barX, y + 2, barW, shotEntries, "");
    y += Math.max(pieR * 2 + 4, shotEntries.length * 8.5) + 3;
  }

  // ── TENDENCIES (concise) ──
  if (card?.card?.tendencies?.length) {
    y = space(y, 16);
    y = h1("TENDENCIES", y);
    for (const t of card.card.tendencies) {
      y = indent_bullet(t, y);
    }
    y += 2;
  }

  // ── SERVE (if relevant) ──
  if (card?.card?.serve_summary) {
    y = space(y, 12);
    y = h1("SERVE", y);
    y = body(card.card.serve_summary, y);
    y += 2;
  }

  // ── COACH NOTES ──
  const playerNotes = notes.filter((n) => n.player === playerKey);
  if (playerNotes.length > 0) {
    y = space(y, 14);
    y = h1("COACH NOTES", y);
    for (const n of playerNotes) {
      const mm = Math.floor(n.timestamp_sec / 60);
      const ss = (n.timestamp_sec % 60).toFixed(1).padStart(4, "0");
      y = indent_bullet(`Point ${n.point_idx + 1} @ ${mm}:${ss}  -  ${n.note_text}`, y);
    }
    y += 2;
  }

  // ═════════════════════════════════════════════════════════
  // PAGE 2 — POINT BREAKDOWN (plain English)
  // ═════════════════════════════════════════════════════════
  if (points.length > 0) {
    doc.addPage();
    y = M;

    doc.setFontSize(18);
    doc.setFont("helvetica", "bold");
    doc.setTextColor(0);
    doc.text(clean("POINT BREAKDOWN"), M, y);
    y += 5;
    doc.setFontSize(8.5);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(130);
    doc.text(clean("Plain English - what happened in each point and why"), M, y);
    y += 2;
    y = rule(y, 150);

    const seenPatterns = new Set<string>();

    for (const pt of points) {
      y = space(y, 20);

      const duration = (pt.end_sec - pt.start_sec).toFixed(1);
      const mm = Math.floor(pt.start_sec / 60);
      const ss = (pt.start_sec % 60).toFixed(0).padStart(2, "0");
      const cc = coachingCards.find((c) => c.point_idx === pt.point_idx);
      const pointShots = shots.filter((s) => s.timestamp_sec >= pt.start_sec && s.timestamp_sec <= pt.end_sec);
      const playerShots = pointShots.filter((s) => s.owner === playerKey);

      // Point label + outcome badge
      const outcomeLabel =
        pt.end_reason === "OUT" ? "Error"
        : pt.end_reason === "NET" ? "Net"
        : pt.end_reason === "WINNER" ? "Winner"
        : pt.end_reason === "DOUBLE_BOUNCE" ? "Won"
        : "Tracking lost";

      doc.setFontSize(9.5);
      doc.setFont("helvetica", "bold");
      doc.setTextColor(0);
      doc.text(clean(`Point ${pt.point_idx + 1}  -  ${outcomeLabel}  (${pt.rally_hit_count} hits, ${duration}s, ${mm}:${ss})`), M, y);
      y += 5;

      // Plain-English sentence
      const sentence = buildPointSentence(pt, cc, playerShots, pa, label, seenPatterns);
      y = body(sentence, y, 3);

      // Serve fault note (human readable, no coordinates)
      if (pt.serve_fault_type) {
        const ft = pt.serve_fault_type.replace(/_/g, " ");
        y = muted(`Serve fault: ${ft}${pt.serve_zone ? ` in the ${pt.serve_zone.replace(/_/g, " ")} zone` : ""}`, y, 3);
      }

      // Separator
      y += 1;
      doc.setDrawColor(220);
      doc.setLineWidth(0.12);
      doc.line(M + 2, y, M + CW, y);
      y += 3;
    }
  }

  // ═════════════════════════════════════════════════════════
  // PAGE 3 — ADVANCED STATS APPENDIX
  // ═════════════════════════════════════════════════════════
  const hasAdvanced = analysis || (pa && (pa.total_distance_covered > 0 || pa.first_serve_pct > 0 || Object.keys(pa.error_rate_by_shot_type).length > 0));

  if (hasAdvanced) {
    doc.addPage();
    y = M;

    doc.setFontSize(16);
    doc.setFont("helvetica", "bold");
    doc.setTextColor(0);
    doc.text(clean("ADVANCED STATS - APPENDIX"), M, y);
    y += 5;
    doc.setFontSize(8);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(150);
    doc.text(clean("Detailed metrics for analysts and technical coaches"), M, y);
    y += 2;
    y = rule(y, 150);

    // Error rates by shot type
    if (pa && Object.keys(pa.error_rate_by_shot_type).length > 0) {
      y = h2("Error Rate by Shot Type", y);
      const entries = Object.entries(pa.error_rate_by_shot_type).map(([t, r]) => ({
        label: t.charAt(0).toUpperCase() + t.slice(1),
        pct: r,
      }));
      y = drawBars(doc, M, y, CW, entries, "");
      y += 3;
    }

    // Error rates by rally length
    if (pa && Object.keys(pa.error_rate_by_rally_length).length > 0) {
      y = h2("Error Rate by Rally Length", y);
      const entries = Object.entries(pa.error_rate_by_rally_length).map(([b, r]) => ({
        label: clean(`${b} shots`),
        pct: r,
      }));
      y = drawBars(doc, M, y, CW, entries, "");
      y += 3;
    }

    // Top patterns bar chart
    const patterns = analytics?.shot_pattern_dominance?.[playerKey];
    if (patterns && patterns.length > 0) {
      y = h2("Shot Pattern Frequency", y);
      const entries = patterns.slice(0, 6).map((p) => ({
        label: clean(`${p.shot_type} ${p.direction.replace(/_/g, " ")}`),
        pct: p.pct,
      }));
      y = drawBars(doc, M, y, CW, entries, "");
      y += 3;
    }

    // Rally length distribution
    if (analytics && Object.keys(analytics.rally_length_distribution).length > 0) {
      y = h2("Rally Length Distribution", y);
      const maxV = Math.max(...Object.values(analytics.rally_length_distribution));
      const entries = Object.entries(analytics.rally_length_distribution).map(([b, c]) => ({
        label: clean(`${b} shots`),
        pct: maxV > 0 ? (c / maxV) * 100 : 0,
      }));
      y = drawBars(doc, M, y, CW, entries, "");
      y = muted(clean(`Average: ${fmtNum(analytics.rally_length_avg)} shots per rally`), y);
      y += 3;
    }

    // Ball pace
    if (analysis?.ball?.speed_stats) {
      const bs = analysis.ball.speed_stats;
      y = h2("Ball Pace", y);
      y = body(
        clean(`Mean: ${kmh(bs.mean)}  |  Median: ${kmh(bs.median)}  |  P90: ${kmh(bs.p90)}  |  Max: ${kmh(bs.max)}`),
        y,
      );

      if (analysis.ball.speed_samples_m_s && analysis.ball.speed_samples_m_s.length > 5) {
        const samples = analysis.ball.speed_samples_m_s.map((v) => v * 3.6);
        const buckets: Record<string, number> = { "0-50": 0, "50-100": 0, "100-150": 0, "150-200": 0, "200+": 0 };
        for (const s of samples) {
          if (s < 50) buckets["0-50"]++;
          else if (s < 100) buckets["50-100"]++;
          else if (s < 150) buckets["100-150"]++;
          else if (s < 200) buckets["150-200"]++;
          else buckets["200+"]++;
        }
        const total = samples.length;
        const bEntries = Object.entries(buckets)
          .filter(([, c]) => c > 0)
          .map(([range, count]) => ({ label: clean(`${range} km/h`), pct: (count / total) * 100 }));
        if (bEntries.length > 1) {
          y = space(y, bEntries.length * 8 + 8);
          y = drawBars(doc, M, y, CW, bEntries, "Speed Distribution");
        }
      }
      y += 2;
    }

    // Player movement
    const playerData = analysis?.players?.[playerKey];
    if (playerData || (pa && pa.total_distance_covered > 0)) {
      y = h2("Player Movement", y);
      const dist = playerData?.distance_m ?? pa?.total_distance_covered ?? 0;
      let moveLine = `Distance covered: ${fmtNum(dist)} m`;
      if (playerData?.speed_stats) {
        moveLine += `  |  Mean speed: ${fmtNum(playerData.speed_stats.mean)} m/s  |  Max: ${fmtNum(playerData.speed_stats.max)} m/s`;
      }
      y = body(clean(moveLine), y);
      if (playerData?.zone_time_pct && Object.keys(playerData.zone_time_pct).length > 0) {
        const zoneEntries = Object.entries(playerData.zone_time_pct)
          .sort(([, a], [, b]) => b - a)
          .map(([zone, p]) => ({ label: zone, pct: p * 100 }));
        y = drawBars(doc, M, y, CW, zoneEntries, "Court Zone Time");
      }
      y += 2;
    }

    // Serve metrics
    if (analysis?.serve || (pa && pa.first_serve_pct > 0)) {
      y = h2("Serve Metrics", y);
      if (pa && pa.first_serve_pct > 0) {
        y = body(clean(`First serve in: ${pct(pa.first_serve_pct)}  |  Double faults: ${pa.double_fault_count}`), y);
      }
      if (analysis?.serve) {
        const sv = analysis.serve;
        if (sv.fault_rate != null) y = body(clean(`Fault rate: ${pct(sv.fault_rate * 100)}  |  Samples: ${sv.sample_count ?? "--"}`), y);
        if (sv.depth_stats?.mean != null) {
          y = muted(clean(`Depth - Mean: ${fmtNum(sv.depth_stats.mean)} m  |  Max: ${fmtNum(sv.depth_stats.max)} m`), y);
        }
        if (sv.zone_counts && Object.keys(sv.zone_counts).length > 0) {
          const maxZ = Math.max(...Object.values(sv.zone_counts));
          const zoneEntries = Object.entries(sv.zone_counts).map(([z, c]) => ({
            label: z, pct: maxZ > 0 ? (c / maxZ) * 100 : 0,
          }));
          y = drawBars(doc, M, y, CW, zoneEntries, "Serve Zone Distribution");
        }
      }
      if (pa && Object.keys(pa.serve_zone_win_rate).length > 0) {
        const szEntries = Object.entries(pa.serve_zone_win_rate).map(([z, r]) => ({
          label: z.replace(/_/g, " "), pct: r,
        }));
        y = drawBars(doc, M, y, CW, szEntries, "Serve Zone Win Rate");
      }
      y += 2;
    }

    // Detection quality (minimal, at end)
    if (analysis?.quality) {
      y = h2("Detection Quality", y);
      const q = analysis.quality;
      const items: string[] = [];
      if (q.ball_coverage_pct != null) items.push(`Ball coverage: ${pct(q.ball_coverage_pct * 100)}`);
      if (q.homography_reliable_pct != null) items.push(`Court mapping: ${pct(q.homography_reliable_pct * 100)}`);
      if (q.frames_total) items.push(`Frames analyzed: ${q.frames_total}`);
      if (analysis.meta?.duration_sec) items.push(`Duration: ${fmtNum(analysis.meta.duration_sec / 60)} min`);
      y = muted(clean(items.join("  |  ")), y);
    }
  }

  doc.save(clean(`TennisIQ_${label.replace(/\s+/g, "_")}_${jobId.slice(0, 8)}.pdf`));
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function PDFExport({
  playerACard, playerBCard, analytics, analysis,
  notes, points, shots, coachingCards, jobId,
}: Props) {
  const [generatingA, setGeneratingA] = useState(false);
  const [generatingB, setGeneratingB] = useState(false);

  const getLatestNotes = async (): Promise<CoachNote[]> => {
    try {
      const resp = await getCoachNotes(jobId);
      return resp.notes ?? notes ?? [];
    } catch {
      // Fallback to in-memory notes if API unavailable.
      return notes ?? [];
    }
  };

  const exportA = async () => {
    setGeneratingA(true);
    try {
      const latestNotes = await getLatestNotes();
      await generatePlayerPDF(playerACard, "player_a", "Player A", analytics, analysis, latestNotes, points, shots, coachingCards, jobId);
    } catch (e) { console.error(e); } finally { setGeneratingA(false); }
  };

  const exportB = async () => {
    setGeneratingB(true);
    try {
      const latestNotes = await getLatestNotes();
      await generatePlayerPDF(playerBCard, "player_b", "Player B", analytics, analysis, latestNotes, points, shots, coachingCards, jobId);
    } catch (e) { console.error(e); } finally { setGeneratingB(false); }
  };

  if (!playerACard && !playerBCard) return null;

  return (
    <div className="flex flex-col items-center gap-3">
      <h4 className="text-sm font-medium text-zinc-500 uppercase tracking-wider">Export Reports</h4>
      <div className="flex gap-3">
        {playerACard && (
          <button
            onClick={exportA}
            disabled={generatingA}
            className="px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:bg-zinc-700 disabled:text-zinc-500 text-white font-semibold text-sm transition-colors"
          >
            {generatingA ? "Generating..." : "Player A Report"}
          </button>
        )}
        {playerBCard && (
          <button
            onClick={exportB}
            disabled={generatingB}
            className="px-5 py-2.5 rounded-xl bg-orange-600 hover:bg-orange-500 disabled:bg-zinc-700 disabled:text-zinc-500 text-white font-semibold text-sm transition-colors"
          >
            {generatingB ? "Generating..." : "Player B Report"}
          </button>
        )}
      </div>
    </div>
  );
}
