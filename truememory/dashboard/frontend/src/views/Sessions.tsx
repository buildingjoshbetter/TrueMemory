import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { SearchBar } from "../components/SearchBar";
import { GlassCard } from "../components/GlassCard";
import { SkeletonCard } from "../components/SkeletonLoader";
import { api } from "../lib/api";
import { relativeTime, formatNumber } from "../lib/formatters";
import { springDefault } from "../lib/constants";
import type { Session, TranscriptMessage } from "../lib/types";

export function Sessions() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [project, setProject] = useState("");
  const [projects, setProjects] = useState<string[]>([]);
  const [selectedSession, setSelectedSession] = useState<Session | null>(null);
  const [transcript, setTranscript] = useState<TranscriptMessage[]>([]);
  const [loadingTranscript, setLoadingTranscript] = useState(false);
  const [indexing, setIndexing] = useState(false);

  useEffect(() => {
    api.sessions.projects().then(setProjects).catch(() => {});
  }, []);

  useEffect(() => {
    const timer = setTimeout(() => {
      setLoading(true);
      api.sessions.list({ search: search || undefined, project: project || undefined, limit: 100 })
        .then((resp) => { setSessions(resp.sessions); setTotal(resp.total); })
        .finally(() => setLoading(false));
    }, search ? 300 : 0);
    return () => clearTimeout(timer);
  }, [search, project]);

  const handleSelectSession = async (s: Session) => {
    setSelectedSession(s);
    setLoadingTranscript(true);
    try {
      const resp = await api.sessions.transcript(s.session_id);
      setTranscript(resp.messages);
    } catch {
      setTranscript([]);
    } finally {
      setLoadingTranscript(false);
    }
  };

  const handleReindex = async () => {
    setIndexing(true);
    try {
      const result = await api.sessions.reindex();
      setTotal(result.total);
      const resp = await api.sessions.list({ limit: 100 });
      setSessions(resp.sessions);
      setTotal(resp.total);
    } catch (err) {
      console.error("Reindex failed:", err);
    } finally {
      setIndexing(false);
    }
  };

  if (selectedSession) {
    return (
      <TranscriptView
        session={selectedSession}
        messages={transcript}
        loading={loadingTranscript}
        onBack={() => { setSelectedSession(null); setTranscript([]); }}
      />
    );
  }

  return (
    <div className="h-full flex flex-col">
      <div className="flex-shrink-0 px-5 pt-5 pb-3 space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-2xl font-semibold">Sessions</h2>
          <button
            onClick={handleReindex}
            disabled={indexing}
            className="px-3 py-1.5 text-xs rounded-lg bg-accent/15 text-accent hover:bg-accent/25 transition-colors font-medium"
          >
            {indexing ? "Indexing…" : "Reindex"}
          </button>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex-1">
            <SearchBar value={search} onChange={setSearch} placeholder="Search sessions…" />
          </div>
          <select
            value={project}
            onChange={(e) => setProject(e.target.value)}
            className="bg-bg-elevated border border-[rgba(255,255,255,0.06)] rounded-lg px-3 py-2.5 text-[13px] text-text-primary outline-none appearance-none cursor-pointer max-w-[200px]"
          >
            <option value="">All projects</option>
            {projects.map((p) => (
              <option key={p} value={p}>{p.split("/").pop() || p}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-5 pb-5 space-y-2">
        {loading ? (
          Array.from({ length: 5 }).map((_, i) => <SkeletonCard key={i} />)
        ) : sessions.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-20 text-text-tertiary">
            <p className="text-sm mb-2">Indexing sessions...</p>
            <p className="text-xs">Sessions are being indexed in the background. This may take a moment.</p>
          </div>
        ) : (
          sessions.map((s, i) => (
            <motion.div
              key={s.session_id}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: Math.min(i * 0.02, 0.3), ...springDefault }}
            >
              <button
                onClick={() => handleSelectSession(s)}
                className="w-full text-left glass-panel p-4 hover:bg-bg-elevated/80 transition-colors"
              >
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-[13px] text-text-primary">
                    {s.started_at ? new Date(s.started_at).toLocaleDateString("en-US", {
                      month: "short", day: "numeric", year: "numeric",
                    }) : "Unknown date"}
                    {s.started_at && (
                      <span className="text-text-tertiary ml-2">
                        {new Date(s.started_at).toLocaleTimeString("en-US", {
                          hour: "numeric", minute: "2-digit",
                        })}
                      </span>
                    )}
                  </span>
                  <span className="text-[11px] text-text-tertiary">
                    {formatNumber(s.message_count)} msgs
                  </span>
                </div>
                <p className="text-xs text-text-tertiary mb-1.5 truncate">
                  {s.project_dir}
                </p>
                {s.summary && (
                  <p className="text-[13px] text-text-secondary leading-snug line-clamp-2">
                    {s.summary}
                  </p>
                )}
              </button>
            </motion.div>
          ))
        )}
      </div>

      <div className="flex-shrink-0 px-5 py-2.5 border-t border-[rgba(255,255,255,0.04)] text-[11px] text-text-tertiary">
        {formatNumber(total)} sessions indexed
      </div>
    </div>
  );
}

function TranscriptView({
  session,
  messages,
  loading,
  onBack,
}: {
  session: Session;
  messages: TranscriptMessage[];
  loading: boolean;
  onBack: () => void;
}) {
  return (
    <div className="h-full flex flex-col">
      <div className="flex-shrink-0 px-5 pt-5 pb-3 border-b border-[rgba(255,255,255,0.04)]">
        <div className="flex items-center gap-3 mb-2">
          <button onClick={onBack} className="text-accent hover:text-accent-hover text-sm">
            ← Back
          </button>
          <span className="text-[13px] text-text-secondary">
            {session.started_at && new Date(session.started_at).toLocaleDateString("en-US", {
              weekday: "short", month: "short", day: "numeric", year: "numeric",
            })}
          </span>
          <span className="text-[11px] text-text-tertiary ml-auto">
            {formatNumber(session.message_count)} messages
          </span>
        </div>
        <p className="text-xs text-text-tertiary font-mono">{session.project_dir}</p>
      </div>

      <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
        {loading ? (
          Array.from({ length: 4 }).map((_, i) => <SkeletonCard key={i} />)
        ) : messages.length === 0 ? (
          <p className="text-text-tertiary text-sm text-center py-10">No transcript data available</p>
        ) : (
          messages.map((msg, i) => (
            <motion.div
              key={msg.uuid || i}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: Math.min(i * 0.01, 0.5), ...springDefault }}
              className={`${msg.type === "user" ? "ml-0 mr-12" : "ml-6 mr-0"}`}
            >
              <div className="flex items-center gap-2 mb-1">
                <span className={`text-[11px] font-semibold uppercase tracking-wider ${
                  msg.type === "user" ? "text-info" : "text-accent"
                }`}>
                  {msg.type === "user" ? "You" : "Claude"}
                </span>
                {msg.timestamp && (
                  <span className="text-[10px] text-text-tertiary">
                    {new Date(msg.timestamp).toLocaleTimeString("en-US", {
                      hour: "numeric", minute: "2-digit",
                    })}
                  </span>
                )}
              </div>
              <div className={`rounded-xl px-4 py-3 text-[13px] leading-relaxed ${
                msg.type === "user"
                  ? "bg-info/10 text-text-primary"
                  : "bg-bg-elevated text-text-primary/90"
              }`}>
                <p className="whitespace-pre-wrap break-words">{msg.content || "(empty)"}</p>
              </div>
            </motion.div>
          ))
        )}
      </div>
    </div>
  );
}
