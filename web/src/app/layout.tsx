import type { Metadata, Viewport } from "next";
import "./globals.css";

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
};

export const metadata: Metadata = {
  title: "sfp — Forgetting Leaderboard",
  description: "Does your LLM forget? Benchmark catastrophic forgetting in 5 minutes.",
  openGraph: {
    title: "sfp — Forgetting Leaderboard",
    description: "Does your LLM forget? Benchmark catastrophic forgetting in 5 minutes.",
    url: "https://github.com/paradigmxyz/sfp",
    type: "website",
  },
  twitter: {
    card: "summary",
    title: "sfp — Forgetting Leaderboard",
    description: "Does your LLM forget? Benchmark catastrophic forgetting in 5 minutes.",
  },
  icons: {
    icon: "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🧠</text></svg>",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
