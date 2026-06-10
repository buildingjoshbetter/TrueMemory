import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { StatCard } from "../components/StatCard";
import { GlassCard } from "../components/GlassCard";
import { CategoryBadge } from "../components/CategoryBadge";
import { SkeletonStat } from "../components/SkeletonLoader";
import { api } from "../lib/api";
import { relativeTime, truncate, formatBytes, stripCategoryPrefix } from "../lib/formatters";
import type { DashboardStats, Memory, HealthResponse } from "../lib/types";
import { springDefault } from "../lib/constants";

interface Props {
  health: HealthResponse | null;
  onNavigateExplorer: () => void;
}

export function Overview({ health, onNavigateExplorer }: Props) {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [recent, setRecent] = useState<Memory[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      api.memories.stats(),
      api.memories.list({ sort: "newest", limit: 20 }),
    ]).then(([s, r]) => {
      setStats(s);
      setRecent(r.memories);
      setLoading(false);
    });
  }, []);

  return (
    <div className="h-full overflow-y-auto p-6 space-y-5">
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={springDefault}
      >
        <h2 className="text-2xl font-semibold mb-5">Overview</h2>
      </motion.div>

      <div className="grid grid-cols-4 gap-3">
        {loading || !stats ? (
          <>
            <SkeletonStat />
            <SkeletonStat />
            <SkeletonStat />
            <SkeletonStat />
          </>
        ) : (
          <>
            <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0, ...springDefault }}>
              <StatCard value={stats.total} label="Total Memories" sparklineData={stats.sparkline} color="#6C5CE7" />
            </motion.div>
            <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05, ...springDefault }}>
              <StatCard value={stats.this_week} label="This Week" sparklineData={stats.sparkline.slice(-7)} color="#64D2FF" />
            </motion.div>
            <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1, ...springDefault }}>
              <StatCard value={stats.entities} label="Entities Tracked" color="#30D158" />
            </motion.div>
            <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15, ...springDefault }}>
              <StatCard
                value={stats.gate_pass_rate != null ? `${Math.round(stats.gate_pass_rate * 100)}%` : "N/A"}
                label="Gate Pass Rate"
                color="#FFD60A"
              />
            </motion.div>
          </>
        )}
      </div>

      <div className="grid grid-cols-3 gap-3">
        <div className="col-span-2">
          <GlassCard padding={false}>
            <div className="flex items-center justify-between px-5 pt-4 pb-3">
              <h3 className="text-sm font-semibold">Recent Memories</h3>
              <button
                onClick={onNavigateExplorer}
                className="text-xs text-accent hover:text-accent-hover transition-colors"
              >
                View all →
              </button>
            </div>
            <div className="max-h-[400px] overflow-y-auto">
              {recent.map((m) => (
                <div
                  key={m.id}
                  className="px-5 py-3 border-t border-[rgba(255,255,255,0.04)] hover:bg-bg-elevated/50 transition-colors cursor-pointer"
                  onClick={onNavigateExplorer}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <CategoryBadge category={m.category} />
                    <span className="text-[11px] text-text-tertiary ml-auto">
                      {relativeTime(m.timestamp)}
                    </span>
                  </div>
                  <p className="text-[13px] text-text-primary/80 leading-snug">
                    {truncate(stripCategoryPrefix(m.content), 120)}
                  </p>
                </div>
              ))}
              {recent.length === 0 && !loading && (
                <p className="px-5 py-8 text-sm text-text-tertiary text-center">
                  No memories yet
                </p>
              )}
            </div>
          </GlassCard>
        </div>

        <GlassCard>
          <h3 className="text-sm font-semibold mb-4">System Health</h3>
          <div className="space-y-3 text-[13px]">
            <HealthRow label="Tier" value={health?.tier?.toUpperCase() || "—"} />
            <HealthRow label="DB Size" value={health ? formatBytes(health.db_size_kb) : "—"} />
            <HealthRow label="Version" value={health?.version || "—"} />
            {(() => {
              const caps = (health?.capabilities && Object.keys(health.capabilities).length > 0)
                ? health.capabilities
                : (stats?.capabilities || {});
              const entries = Object.entries(caps).filter(([, v]) => v);
              if (entries.length === 0) return null;
              return (
                <div className="border-t border-[rgba(255,255,255,0.04)] pt-3 mt-3">
                  <p className="text-[11px] text-text-tertiary mb-2">Capabilities</p>
                  <div className="flex flex-wrap gap-1.5">
                    {entries.map(([k]) => (
                      <span
                        key={k}
                        className="text-[10px] px-1.5 py-0.5 rounded bg-success/10 text-success"
                      >
                        {k}
                      </span>
                    ))}
                  </div>
                </div>
              );
            })()}
          </div>

          {stats?.categories && (
            <div className="mt-5 pt-4 border-t border-[rgba(255,255,255,0.04)]">
              <p className="text-[11px] text-text-tertiary mb-2">Categories</p>
              <div className="space-y-1.5">
                {Object.entries(stats.categories)
                  .slice(0, 6)
                  .map(([cat, count]) => (
                    <div key={cat} className="flex items-center justify-between">
                      <CategoryBadge category={cat} />
                      <span className="text-[11px] text-text-tertiary font-mono">{count}</span>
                    </div>
                  ))}
              </div>
            </div>
          )}
        </GlassCard>
      </div>
    </div>
  );
}

function HealthRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <span className="text-text-tertiary">{label}</span>
      <span className="font-medium">{value}</span>
    </div>
  );
}
