export interface Memory {
  id: number;
  content: string;
  sender: string;
  recipient: string;
  timestamp: string;
  category: string;
  modality: string;
  score?: number;
  source?: string;
}

export interface HealthResponse {
  version: string;
  tier: string;
  db_path: string;
  db_size_kb: number;
  memory_count: number;
  capabilities: Record<string, boolean>;
}

export interface DashboardStats {
  total: number;
  this_week: number;
  entities: number;
  gate_pass_rate: number | null;
  sparkline: number[];
  categories: Record<string, number>;
  capabilities?: Record<string, boolean>;
}

export interface MemoriesResponse {
  memories: Memory[];
  total: number;
  limit: number;
  offset: number;
}

export interface UpdateInfo {
  current: string;
  latest: string | null;
  update_available: boolean;
  error?: string;
}

export interface TierInfo {
  tier: string;
  has_api_key: boolean;
  api_provider: string;
}

export interface Session {
  session_id: string;
  project_dir: string;
  started_at: string;
  ended_at: string;
  message_count: number;
  user_message_count: number;
  word_count: number;
  summary: string;
  version: string;
  jsonl_path?: string;
}

export interface SessionsResponse {
  sessions: Session[];
  total: number;
  limit: number;
  offset: number;
}

export interface TranscriptMessage {
  type: string;
  content: string;
  timestamp: string;
  uuid: string;
}

export interface Entity {
  entity: string;
  message_count: number;
  traits: string[];
  topics: string[];
  updated_at: string;
}

export interface EntityGraph {
  nodes: { id: string; message_count: number; radius: number }[];
  edges: { source: string; target: string; relationship_type: string; strength: number; dunbar_layer: string }[];
}

export interface Fact {
  id: number;
  subject: string;
  fact: string;
  source_message_id: number | null;
  timestamp: string;
  superseded_by: number | null;
  entity_scope: string;
  valid_from: string;
  valid_to: string;
  is_current: boolean;
}

export interface FactsResponse {
  facts: Fact[];
  total: number;
  subjects: string[];
}

export interface GrowthPoint {
  date: string;
  count: number;
  cumulative: number;
}

export interface CategoryCount {
  category: string;
  count: number;
}

export type ViewId = "overview" | "explorer" | "sessions" | "people" | "facts" | "analytics" | "settings";
