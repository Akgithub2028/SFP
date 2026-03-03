import type { LeaderboardData, DetailData } from "./types";
import { MOCK_LEADERBOARD, MOCK_DETAILS } from "./mock";

let _lbCache: Promise<LeaderboardData | null> | null = null;

export function fetchLeaderboard(): Promise<LeaderboardData | null> {
  if (!_lbCache) {
    _lbCache = fetch("/api/leaderboard")
      .then((res) => (res.ok ? res.json() : MOCK_LEADERBOARD))
      .catch(() => MOCK_LEADERBOARD);
  }
  return _lbCache;
}

export async function fetchDetails(): Promise<DetailData | null> {
  try {
    const res = await fetch("/api/leaderboard/details");
    if (!res.ok) throw new Error();
    return await res.json();
  } catch {
    return MOCK_DETAILS;
  }
}
