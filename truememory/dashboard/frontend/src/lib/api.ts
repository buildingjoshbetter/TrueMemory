import type {
  HealthResponse,
  MemoriesResponse,
  Memory,
  DashboardStats,
  TierInfo,
  UpdateInfo,
  SessionsResponse,
  Session,
  TranscriptMessage,
  Entity,
  EntityGraph,
  FactsResponse,
  GrowthPoint,
  CategoryCount,
} from "./types";

const BASE = "/api";

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const headers: Record<string, string> = { ...((init?.headers as Record<string, string>) || {}) };
  if (init?.body && typeof init.body === "string") {
    headers["Content-Type"] = "application/json";
  }
  const res = await fetch(`${BASE}${path}`, { ...init, headers });
  if (!res.ok) throw new ApiError(res.status, await res.text());
  return res.json();
}

function qs(params: Record<string, string | number | boolean | undefined | null>): string {
  const entries = Object.entries(params).filter(([, v]) => v != null && v !== "");
  return new URLSearchParams(entries.map(([k, v]) => [k, String(v)])).toString();
}

export const api = {
  health: () => fetchJson<HealthResponse>("/health"),

  memories: {
    list: (params: {
      search?: string;
      category?: string;
      sender?: string;
      sort?: string;
      limit?: number;
      offset?: number;
    }) => fetchJson<MemoriesResponse>(`/memories?${qs(params)}`),

    get: (id: number) => fetchJson<Memory>(`/memories/${id}`),

    search: (query: string, limit = 50) =>
      fetchJson<Memory[]>("/memories/search", {
        method: "POST",
        body: JSON.stringify({ query, limit }),
      }),

    update: (id: number, content: string) =>
      fetchJson<Memory>(`/memories/${id}`, {
        method: "PUT",
        body: JSON.stringify({ content }),
      }),

    delete: (id: number) =>
      fetchJson<{ deleted: boolean }>(`/memories/${id}`, { method: "DELETE" }),

    bulkDelete: (ids: number[]) =>
      fetchJson<{ deleted: number; total: number }>("/memories/bulk-delete", {
        method: "POST",
        body: JSON.stringify({ ids }),
      }),

    senders: () => fetchJson<string[]>("/memories/senders"),

    categories: () => fetchJson<string[]>("/memories/categories"),

    stats: () => fetchJson<DashboardStats>("/memories/stats"),
  },

  tier: () => fetchJson<TierInfo>("/tier"),

  checkUpdate: () =>
    fetchJson<UpdateInfo>("/update/check", { method: "POST" }),

  sessions: {
    list: (params: { search?: string; project?: string; limit?: number; offset?: number }) =>
      fetchJson<SessionsResponse>(`/sessions?${qs(params)}`),
    get: (id: string) => fetchJson<Session>(`/sessions/${id}`),
    transcript: (id: string) =>
      fetchJson<{ messages: TranscriptMessage[]; count: number }>(`/sessions/${id}/transcript`),
    projects: () => fetchJson<string[]>("/sessions/projects"),
    reindex: () => fetchJson<{ indexed: number; total: number }>("/sessions/reindex", { method: "POST" }),
  },

  entities: {
    list: () => fetchJson<Entity[]>("/entities"),
    get: (name: string) => fetchJson<{ profile: Record<string, unknown>; recent_memories: Memory[] }>(`/entities/${encodeURIComponent(name)}`),
    graph: () => fetchJson<EntityGraph>("/entities/graph"),
    preferences: (name: string) => fetchJson<Record<string, unknown[]>>(`/entities/${encodeURIComponent(name)}/preferences`),
  },

  facts: {
    list: (params?: { subject?: string; show_superseded?: boolean }) =>
      fetchJson<FactsResponse>(`/facts?${qs(params || {})}`),
    contradictions: () =>
      fetchJson<{ subject: string; fact_a: Record<string, unknown>; fact_b: Record<string, unknown> }[]>("/facts/contradictions"),
  },

  analytics: {
    growth: () => fetchJson<GrowthPoint[]>("/analytics/growth"),
    categories: () => fetchJson<CategoryCount[]>("/analytics/categories"),
    entities: () => fetchJson<{ entity: string; message_count: number }[]>("/analytics/entities"),
    ingest: () => fetchJson<Record<string, unknown>>("/analytics/ingest"),
    timeline: () => fetchJson<Record<string, Record<string, number>>>("/analytics/timeline"),
  },
};
