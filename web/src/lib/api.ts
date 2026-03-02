import type { LeaderboardData, DetailData, PromResult } from "./types";

const API_BASE =
  typeof window !== "undefined" && window.location.port === "3000"
    ? "http://localhost:8090"
    : "";

const PROM = `${API_BASE}/api/v1`;

export async function fetchLeaderboard(): Promise<LeaderboardData | null> {
  try {
    const res = await fetch(`${API_BASE}/leaderboard`);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export async function fetchDetails(): Promise<DetailData | null> {
  try {
    const res = await fetch(`${API_BASE}/leaderboard/details`);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export async function promInstant(q: string): Promise<PromResult[]> {
  try {
    const res = await fetch(
      `${PROM}/query?query=${encodeURIComponent(q)}`
    );
    if (!res.ok) return [];
    const data = await res.json();
    return data?.data?.result ?? [];
  } catch {
    return [];
  }
}

export async function promRange(
  q: string,
  start: string,
  end: string,
  step: string
): Promise<PromResult[]> {
  try {
    const res = await fetch(
      `${PROM}/query_range?query=${encodeURIComponent(q)}&start=${start}&end=${end}&step=${step}`
    );
    if (!res.ok) return [];
    const data = await res.json();
    return data?.data?.result ?? [];
  } catch {
    return [];
  }
}
