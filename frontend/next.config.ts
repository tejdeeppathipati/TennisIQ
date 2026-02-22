import type { NextConfig } from "next";

const backendTarget =
  (process.env.BACKEND_PROXY_URL || process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8002").replace(/\/+$/, "");

const nextConfig: NextConfig = {
  experimental: {
    // Allow large video uploads when proxying to backend (default is 10MB)
    proxyClientMaxBodySize: "512mb",
  },
  async rewrites() {
    return [
      {
        source: "/backend/:path*",
        destination: `${backendTarget}/:path*`,
      },
    ];
  },
};

export default nextConfig;
