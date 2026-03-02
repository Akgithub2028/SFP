import type { LeaderboardData, DetailData, PromResult } from "./types";
import { MOCK_LEADERBOARD, MOCK_DETAILS } from "./mock";

const API_BASE =
  typeof window !== "undefined" && window.location.port === "3000"
    ? "http://localhost:8090"
    : "";

const PROM = `${API_BASE}/api/v1`;

export async function fetchLeaderboard(): Promise<LeaderboardData | null> {
  try {
    const res = await fetch(`${API_BASE}/leaderboard`);
    if (!res.ok) throw new Error();
    return await res.json();
  } catch {
    return MOCK_LEADERBOARD;
  }
}

export async function fetchDetails(): Promise<DetailData | null> {
  try {
    const res = await fetch(`${API_BASE}/leaderboard/details`);
    if (!res.ok) throw new Error();
    return await res.json();
  } catch {
    return MOCK_DETAILS;
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
