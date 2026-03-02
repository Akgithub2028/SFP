import type { LeaderboardData, DetailData } from "./types";
import { MOCK_LEADERBOARD, MOCK_DETAILS } from "./mock";

export async function fetchLeaderboard(): Promise<LeaderboardData | null> {
  try {
    const res = await fetch("/api/leaderboard");
    if (!res.ok) throw new Error();
    return await res.json();
  } catch {
    return MOCK_LEADERBOARD;
  }
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
