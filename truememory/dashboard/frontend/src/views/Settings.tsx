import { useState } from "react";
import { GlassCard } from "../components/GlassCard";
import { ConfirmSheet } from "../components/ConfirmSheet";
import { api } from "../lib/api";
import { formatBytes } from "../lib/formatters";
import type { HealthResponse, UpdateInfo } from "../lib/types";

interface Props {
  health: HealthResponse | null;
}

export function Settings({ health }: Props) {
  const [updateInfo, setUpdateInfo] = useState<UpdateInfo | null>(null);
  const [checking, setChecking] = useState(false);
  const [confirmDeleteAll, setConfirmDeleteAll] = useState(false);

  const handleCheckUpdate = async () => {
    setChecking(true);
    try {
      const info = await api.checkUpdate();
      setUpdateInfo(info);
    } finally {
      setChecking(false);
    }
  };

  const handleDeleteAll = async () => {
    await fetch("/api/memories", { method: "DELETE" }).catch(() => {});
    setConfirmDeleteAll(false);
    window.location.reload();
  };

  const tierColors: Record<string, string> = {
    edge: "#30D158",
    base: "#64D2FF",
    pro: "#BF5AF2",
  };
  const tier = health?.tier || "edge";

  return (
    <div className="h-full overflow-y-auto p-6 space-y-5">
      <h2 className="text-2xl font-semibold mb-5">Settings</h2>

      <div className="grid grid-cols-2 gap-4">
        <GlassCard>
          <h3 className="text-sm font-semibold mb-4">Tier</h3>
          <div className="flex items-center gap-3">
            <span
              className="text-sm font-bold uppercase px-2.5 py-1 rounded-lg"
              style={{
                backgroundColor: `${tierColors[tier]}20`,
                color: tierColors[tier],
              }}
            >
              {tier}
            </span>
            <span className="text-sm text-text-secondary">
              {tier === "edge" && "Lightweight local model (Model2Vec 8M)"}
              {tier === "base" && "Full semantic search (Qwen3 600M)"}
              {tier === "pro" && "Full semantic + HyDE query expansion"}
            </span>
          </div>
        </GlassCard>

        <GlassCard>
          <h3 className="text-sm font-semibold mb-4">Database</h3>
          <div className="space-y-2.5 text-[13px]">
            <Row label="Path">
              <span className="font-mono text-text-tertiary text-xs break-all">
                {health?.db_path || "—"}
              </span>
            </Row>
            <Row label="Size">
              <span>{health ? formatBytes(health.db_size_kb) : "—"}</span>
            </Row>
            <Row label="Memories">
              <span>{health?.memory_count?.toLocaleString() || "—"}</span>
            </Row>
          </div>
        </GlassCard>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <GlassCard>
          <h3 className="text-sm font-semibold mb-4">Version</h3>
          <div className="flex items-center gap-4">
            <span className="text-[13px] font-mono">{health?.version || "—"}</span>
            <button
              onClick={handleCheckUpdate}
              disabled={checking}
              className="px-3 py-1.5 text-xs rounded-lg bg-accent/15 text-accent hover:bg-accent/25 transition-colors font-medium"
            >
              {checking ? "Checking…" : "Check for Updates"}
            </button>
          </div>
          {updateInfo && (
            <div className="mt-3 text-[13px]">
              {updateInfo.update_available ? (
                <p className="text-success">
                  Update available: <span className="font-mono font-bold">{updateInfo.latest}</span>
                  <br />
                  <span className="text-text-tertiary text-xs">
                    Run in terminal: pip install --upgrade truememory
                  </span>
                </p>
              ) : updateInfo.error ? (
                <p className="text-text-tertiary">{updateInfo.error}</p>
              ) : (
                <p className="text-text-secondary">You're on the latest version.</p>
              )}
            </div>
          )}
        </GlassCard>

        <GlassCard>
          <h3 className="text-[13px] font-medium text-text-secondary mb-3">Session Index</h3>
          <p className="text-[13px] text-text-secondary">
            Sessions are automatically indexed on server startup.
          </p>
          <p className="text-[12px] text-text-tertiary mt-1">
            Use the Sessions tab to browse indexed conversations.
          </p>
        </GlassCard>
      </div>

      <GlassCard>
        <h3 className="text-sm font-semibold mb-2">About</h3>
        <p className="text-[13px] text-text-secondary">
          TrueMemory — SOTA agent memory at zero infrastructure cost.
        </p>
        <p className="text-[12px] text-text-tertiary mt-1">
          Built by Josh Adler
        </p>
      </GlassCard>

      <GlassCard className="border-error/20">
        <h3 className="text-sm font-semibold text-error mb-3">Danger Zone</h3>
        <p className="text-[13px] text-text-secondary mb-3">
          Permanently delete all memories from the database. This cannot be undone.
        </p>
        <button
          onClick={() => setConfirmDeleteAll(true)}
          className="px-4 py-2 text-xs rounded-lg bg-error/10 text-error hover:bg-error/20 transition-colors font-medium"
        >
          Delete All Memories
        </button>
      </GlassCard>

      <ConfirmSheet
        open={confirmDeleteAll}
        title="Delete All Memories"
        message="This will permanently delete every memory in your database. This action cannot be undone."
        confirmLabel="Delete Everything"
        onConfirm={handleDeleteAll}
        onCancel={() => setConfirmDeleteAll(false)}
      />
    </div>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-4">
      <span className="text-text-tertiary flex-shrink-0">{label}</span>
      <div className="text-right">{children}</div>
    </div>
  );
}
