"use client";

interface DownloadItem {
  label: string;
  href: string;
  size?: string;
}

interface Props {
  items: DownloadItem[];
}

export default function DownloadPanel({ items }: Props) {
  if (!items.length) return null;

  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold text-zinc-300 px-1">Downloads</h3>
      <div className="rounded-xl bg-zinc-900 border border-zinc-800 divide-y divide-zinc-800">
        {items.map((item, i) => (
          <a
            key={i}
            href={item.href}
            download
            className="flex items-center justify-between px-4 py-3 hover:bg-zinc-800/50 transition-colors group"
          >
            <div className="flex items-center gap-2">
              <svg
                width="14"
                height="14"
                viewBox="0 0 14 14"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
                className="text-zinc-500 group-hover:text-green-400 transition-colors"
              >
                <path d="M7 1v9m0 0l-3-3m3 3l3-3M2 12h10" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              <span className="text-sm text-zinc-300 group-hover:text-white transition-colors">
                {item.label}
              </span>
            </div>
            {item.size && (
              <span className="text-[10px] text-zinc-600">{item.size}</span>
            )}
          </a>
        ))}
      </div>
    </div>
  );
}
