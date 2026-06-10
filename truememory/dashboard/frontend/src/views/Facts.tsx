import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { SearchBar } from "../components/SearchBar";
import { GlassCard } from "../components/GlassCard";
import { SkeletonCard } from "../components/SkeletonLoader";
import { api } from "../lib/api";
import { relativeTime, stripCategoryPrefix } from "../lib/formatters";
import { springDefault } from "../lib/constants";
import type { Fact } from "../lib/types";

export function Facts() {
  const [facts, setFacts] = useState<Fact[]>([]);
  const [subjects, setSubjects] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [subjectFilter, setSubjectFilter] = useState("");
  const [showSuperseded, setShowSuperseded] = useState(false);
  const [contradictions, setContradictions] = useState<
    { subject: string; fact_a: Record<string, unknown>; fact_b: Record<string, unknown> }[]
  >([]);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      api.facts.list({ subject: subjectFilter || undefined, show_superseded: showSuperseded }),
      api.facts.contradictions(),
    ])
      .then(([resp, contras]) => {
        setFacts(resp.facts);
        setSubjects(resp.subjects);
        setContradictions(contras);
      })
      .finally(() => setLoading(false));
  }, [subjectFilter, showSuperseded]);

  const filteredFacts = search
    ? facts.filter((f) => f.fact.toLowerCase().includes(search.toLowerCase()) || f.subject.toLowerCase().includes(search.toLowerCase()))
    : facts;

  const grouped = new Map<string, Fact[]>();
  for (const f of filteredFacts) {
    const list = grouped.get(f.subject) || [];
    list.push(f);
    grouped.set(f.subject, list);
  }

  if (loading) {
    return (
      <div className="h-full p-6 space-y-4">
        <h2 className="text-2xl font-semibold">Facts & Contradictions</h2>
        <SkeletonCard />
        <SkeletonCard />
        <SkeletonCard />
      </div>
    );
  }

  if (facts.length === 0) {
    return (
      <div className="h-full flex flex-col">
        <div className="flex-shrink-0 px-5 pt-5 pb-3">
          <h2 className="text-2xl font-semibold">Facts & Contradictions</h2>
        </div>
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center text-text-tertiary max-w-md">
            <p className="text-lg mb-3">No facts tracked yet</p>
            <p className="text-sm leading-relaxed">
              The fact timeline tracks evolving truths — like when someone moves cities or
              changes jobs. As TrueMemory's L5 consolidation processes more conversations,
              facts and contradictions will appear here.
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      <div className="flex-shrink-0 px-5 pt-5 pb-3 space-y-3">
        <h2 className="text-2xl font-semibold">Facts & Contradictions</h2>
        <div className="flex items-center gap-3">
          <div className="flex-1">
            <SearchBar value={search} onChange={setSearch} placeholder="Search facts…" />
          </div>
          <select
            value={subjectFilter}
            onChange={(e) => setSubjectFilter(e.target.value)}
            className="bg-bg-elevated border border-[rgba(255,255,255,0.06)] rounded-lg px-3 py-2.5 text-[13px] text-text-primary outline-none appearance-none cursor-pointer"
          >
            <option value="">All subjects</option>
            {subjects.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
          <label className="flex items-center gap-2 text-[12px] text-text-secondary cursor-pointer">
            <input
              type="checkbox"
              checked={showSuperseded}
              onChange={(e) => setShowSuperseded(e.target.checked)}
              className="accent-accent"
            />
            Show superseded
          </label>
        </div>
      </div>

      {contradictions.length > 0 && (
        <div className="px-5 pb-3">
          <GlassCard className="border-warning/20">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-warning text-sm">⚠</span>
              <h3 className="text-sm font-semibold text-warning">Contradictions Detected</h3>
            </div>
            <div className="space-y-2">
              {contradictions.map((c, i) => (
                <div key={i} className="text-[12px] text-text-secondary">
                  <span className="text-text-tertiary">{c.subject}:</span>{" "}
                  "{String(c.fact_a.fact)}" vs "{String(c.fact_b.fact)}"
                </div>
              ))}
            </div>
          </GlassCard>
        </div>
      )}

      <div className="flex-1 overflow-y-auto px-5 pb-5 space-y-5">
        {[...grouped.entries()].map(([subject, factList], gi) => (
          <motion.div
            key={subject}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: Math.min(gi * 0.05, 0.3), ...springDefault }}
          >
            <h3 className="text-sm font-semibold text-text-primary mb-2">{subject}</h3>
            <div className="space-y-1.5 ml-2">
              {factList.map((f) => (
                <div
                  key={f.id}
                  className={`flex items-start gap-2.5 ${f.is_current ? "" : "opacity-40"}`}
                >
                  <span className={`mt-1.5 w-2 h-2 rounded-full flex-shrink-0 ${
                    f.is_current ? "bg-accent" : "bg-text-tertiary"
                  }`} />
                  <div className="flex-1 min-w-0">
                    <p className={`text-[13px] ${f.is_current ? "text-text-primary" : "text-text-tertiary line-through"}`}>
                      {stripCategoryPrefix(f.fact)}
                    </p>
                    <div className="flex items-center gap-3 mt-0.5">
                      {f.timestamp && (
                        <span className="text-[10px] text-text-tertiary">{relativeTime(f.timestamp)}</span>
                      )}
                      <span className={`text-[10px] ${f.is_current ? "text-success" : "text-text-tertiary"}`}>
                        {f.is_current ? "current" : "superseded"}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
