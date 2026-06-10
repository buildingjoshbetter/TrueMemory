import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { GlassCard } from "../components/GlassCard";
import { Sparkline } from "../components/Sparkline";
import { SkeletonStat } from "../components/SkeletonLoader";
import { api } from "../lib/api";
import { formatNumber } from "../lib/formatters";
import { springDefault, getCategoryColor } from "../lib/constants";
import type { GrowthPoint, CategoryCount } from "../lib/types";

export function Analytics() {
  const [growth, setGrowth] = useState<GrowthPoint[]>([]);
  const [categories, setCategories] = useState<CategoryCount[]>([]);
  const [topEntities, setTopEntities] = useState<{ entity: string; message_count: number }[]>([]);
  const [ingest, setIngest] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      try { setGrowth(await api.analytics.growth()); } catch {}
      try { setCategories(await api.analytics.categories()); } catch {}
      try { setTopEntities(await api.analytics.entities()); } catch {}
      try { setIngest(await api.analytics.ingest()); } catch {}
      setLoading(false);
    };
    load();
  }, []);

  if (loading) {
    return (
      <div className="h-full p-6 space-y-5">
        <h2 className="text-2xl font-semibold">Analytics</h2>
        <div className="grid grid-cols-3 gap-4"><SkeletonStat /><SkeletonStat /><SkeletonStat /></div>
      </div>
    );
  }

  const total = growth.length > 0 ? growth[growth.length - 1].cumulative : 0;
  const last7 = ingest?.["7d"] as number ?? 0;
  const last30 = ingest?.["30d"] as number ?? 0;
  const dailyRate = (ingest?.daily_rate as { date: string; count: number }[]) || [];
  const realCategories = categories.filter((c) => c.category !== "(uncategorized)");
  const realEntities = topEntities.filter((e) => !e.entity.startsWith("__test") && e.entity !== "test" && e.entity !== "test_user");

  return (
    <div className="h-full overflow-y-auto p-6 space-y-5">
      <motion.h2
        className="text-2xl font-semibold"
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={springDefault}
      >
        Analytics
      </motion.h2>

      {/* Key metrics */}
      <div className="grid grid-cols-3 gap-4">
        <Metric value={formatNumber(total)} label="Total Memories" delay={0} />
        <Metric value={formatNumber(last30)} label="Last 30 Days" delay={0.04} />
        <Metric value={formatNumber(last7)} label="Last 7 Days" delay={0.08} />
      </div>

      {/* Growth + Daily Rate */}
      <div className="grid grid-cols-2 gap-4">
        <FadeIn delay={0.12}>
          <GlassCard>
            <h3 className="text-sm font-semibold mb-1">Memory Growth</h3>
            <p className="text-[11px] text-text-tertiary mb-3">Cumulative memories over time</p>
            <div className="h-36">
              <GrowthChart data={growth} />
            </div>
          </GlassCard>
        </FadeIn>
        <FadeIn delay={0.16}>
          <GlassCard>
            <h3 className="text-sm font-semibold mb-1">Daily Ingest</h3>
            <p className="text-[11px] text-text-tertiary mb-3">Memories created per day (30 days)</p>
            <div className="h-36">
              <Sparkline data={dailyRate.map((d) => d.count)} color="#64D2FF" height={144} />
            </div>
          </GlassCard>
        </FadeIn>
      </div>

      {/* Categories + Entities */}
      <div className="grid grid-cols-2 gap-4">
        <FadeIn delay={0.2}>
          <GlassCard>
            <h3 className="text-sm font-semibold mb-4">Category Distribution</h3>
            <div className="space-y-2.5">
              {realCategories.slice(0, 8).map((c) => {
                const maxCount = realCategories[0]?.count || 1;
                const pct = (c.count / maxCount) * 100;
                const { text: color } = getCategoryColor(c.category.toLowerCase());
                return (
                  <div key={c.category} className="group">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-[12px] text-text-secondary">{c.category}</span>
                      <span className="text-[11px] text-text-tertiary font-mono">{formatNumber(c.count)}</span>
                    </div>
                    <div className="h-1.5 bg-bg-base rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${pct}%` }}
                        transition={{ delay: 0.3, duration: 0.5, ease: "easeOut" }}
                        className="h-full rounded-full"
                        style={{ backgroundColor: color }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </GlassCard>
        </FadeIn>

        <FadeIn delay={0.24}>
          <GlassCard>
            <h3 className="text-sm font-semibold mb-4">Top Entities</h3>
            {realEntities.length === 0 ? (
              <p className="text-sm text-text-tertiary">No entities tracked</p>
            ) : (
              <div className="space-y-2.5">
                {realEntities.slice(0, 8).map((e) => {
                  const maxCount = realEntities[0]?.message_count || 1;
                  const pct = (e.message_count / maxCount) * 100;
                  return (
                    <div key={e.entity} className="group">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-[12px] text-text-secondary">{e.entity}</span>
                        <span className="text-[11px] text-text-tertiary font-mono">{formatNumber(e.message_count)}</span>
                      </div>
                      <div className="h-1.5 bg-bg-base rounded-full overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${pct}%` }}
                          transition={{ delay: 0.35, duration: 0.5, ease: "easeOut" }}
                          className="h-full rounded-full bg-accent/60"
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </GlassCard>
        </FadeIn>
      </div>
    </div>
  );
}

function Metric({ value, label, delay }: { value: string; label: string; delay: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, ...springDefault }}
    >
      <GlassCard className="text-center">
        <span className="text-[32px] font-bold tracking-tight block">{value}</span>
        <span className="text-xs text-text-secondary mt-1 block">{label}</span>
      </GlassCard>
    </motion.div>
  );
}

function FadeIn({ children, delay }: { children: React.ReactNode; delay: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, ...springDefault }}
    >
      {children}
    </motion.div>
  );
}

function GrowthChart({ data }: { data: GrowthPoint[] }) {
  if (!data.length) return <p className="text-text-tertiary text-sm">No data</p>;

  const max = Math.max(...data.map((d) => d.cumulative));
  const w = 400;
  const h = 144;
  const pad = 4;

  const points = data.map((d, i) => {
    const x = (i / Math.max(data.length - 1, 1)) * w;
    const y = h - pad - ((d.cumulative / max) * (h - pad * 2));
    return `${x},${y}`;
  }).join(" ");

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-full" preserveAspectRatio="none">
      <defs>
        <linearGradient id="growth-fill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#6C5CE7" stopOpacity={0.2} />
          <stop offset="100%" stopColor="#6C5CE7" stopOpacity={0} />
        </linearGradient>
      </defs>
      <polygon points={`0,${h} ${points} ${w},${h}`} fill="url(#growth-fill)" />
      <polyline points={points} fill="none" stroke="#6C5CE7" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
