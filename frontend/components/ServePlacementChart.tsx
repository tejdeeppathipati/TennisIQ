"use client";

import type { ServePlacement } from "@/lib/types";

interface Props {
  data: ServePlacement | null;
}

const COURT_WIDTH = 280;
const COURT_HEIGHT = 400;
const PAD = 20;

export default function ServePlacementChart({ data }: Props) {
  if (!data || !data.serves.length) {
    return (
      <div className="rounded-xl bg-zinc-900 border border-zinc-800 p-4 text-center">
        <p className="text-zinc-500 text-xs">No serve data available.</p>
      </div>
    );
  }

  const boxes = Object.values(data.service_boxes);
  const allX = boxes.flatMap((b) => [b.x_min, b.x_max]);
  const allY = boxes.flatMap((b) => [b.y_min, b.y_max]);
  const minX = Math.min(...allX) - 100;
  const maxX = Math.max(...allX) + 100;
  const minY = Math.min(...allY) - 100;
  const maxY = Math.max(...allY) + 100;

  const scaleX = (x: number) => PAD + ((x - minX) / (maxX - minX)) * COURT_WIDTH;
  const scaleY = (y: number) => PAD + ((y - minY) / (maxY - minY)) * COURT_HEIGHT;

  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold text-zinc-300 px-1">Serve Placement</h3>
      <div className="rounded-xl bg-zinc-900 border border-zinc-800 p-4 flex justify-center">
        <svg
          width={COURT_WIDTH + PAD * 2}
          height={COURT_HEIGHT + PAD * 2}
          viewBox={`0 0 ${COURT_WIDTH + PAD * 2} ${COURT_HEIGHT + PAD * 2}`}
        >
          {/* Service boxes */}
          {Object.entries(data.service_boxes).map(([name, box]) => (
            <g key={name}>
              <rect
                x={scaleX(box.x_min)}
                y={scaleY(box.y_min)}
                width={scaleX(box.x_max) - scaleX(box.x_min)}
                height={scaleY(box.y_max) - scaleY(box.y_min)}
                fill="none"
                stroke="#3f3f46"
                strokeWidth={1}
              />
              <text
                x={(scaleX(box.x_min) + scaleX(box.x_max)) / 2}
                y={(scaleY(box.y_min) + scaleY(box.y_max)) / 2}
                textAnchor="middle"
                dominantBaseline="middle"
                className="fill-zinc-600 text-[8px]"
              >
                {name.replace("_", " ")}
              </text>
            </g>
          ))}

          {/* Serve dots */}
          {data.serves.map((s, i) => (
            <circle
              key={i}
              cx={scaleX(s.court_x)}
              cy={scaleY(s.court_y)}
              r={5}
              fill={s.is_fault ? "#ef4444" : "#22c55e"}
              opacity={0.85}
              stroke={s.is_fault ? "#b91c1c" : "#15803d"}
              strokeWidth={1}
            />
          ))}

          {/* Legend */}
          <circle cx={PAD} cy={COURT_HEIGHT + PAD + 10} r={4} fill="#22c55e" />
          <text x={PAD + 8} y={COURT_HEIGHT + PAD + 13} className="fill-zinc-400 text-[9px]">In</text>
          <circle cx={PAD + 30} cy={COURT_HEIGHT + PAD + 10} r={4} fill="#ef4444" />
          <text x={PAD + 38} y={COURT_HEIGHT + PAD + 13} className="fill-zinc-400 text-[9px]">Fault</text>
        </svg>
      </div>
    </div>
  );
}
