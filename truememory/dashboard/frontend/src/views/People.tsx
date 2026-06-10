import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import * as d3 from "d3";
import { GlassCard } from "../components/GlassCard";
import { CategoryBadge } from "../components/CategoryBadge";
import { SkeletonCard } from "../components/SkeletonLoader";
import { api } from "../lib/api";
import { springDefault } from "../lib/constants";
import type { Entity, EntityGraph } from "../lib/types";

export function People() {
  const [entities, setEntities] = useState<Entity[]>([]);
  const [graph, setGraph] = useState<EntityGraph | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [profile, setProfile] = useState<Record<string, unknown> | null>(null);
  const [recentMemories, setRecentMemories] = useState<{ id: number; content: string; timestamp: string; category: string }[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      try {
        const e = await api.entities.list();
        setEntities(e);
      } catch (err) {
        console.error("Failed to load entities:", err);
      }
      try {
        const g = await api.entities.graph();
        setGraph(g);
      } catch (err) {
        console.error("Failed to load graph:", err);
      }
      setLoading(false);
    };
    load();
  }, []);

  const handleSelect = async (name: string) => {
    setSelected(name);
    try {
      const data = await api.entities.get(name);
      setProfile(data.profile);
      setRecentMemories(data.recent_memories as typeof recentMemories);
    } catch {
      setProfile(null);
      setRecentMemories([]);
    }
  };

  if (loading) {
    return (
      <div className="h-full p-6 space-y-4">
        <h2 className="text-xl font-semibold">People</h2>
        <SkeletonCard />
        <SkeletonCard />
      </div>
    );
  }

  if (entities.length === 0) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center text-text-tertiary">
          <p className="text-lg mb-2">No entity profiles yet</p>
          <p className="text-sm">Entity profiles are built as TrueMemory processes conversations.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex">
      <div className="flex-1 flex flex-col min-w-0">
        <div className="flex-shrink-0 px-5 pt-5 pb-3">
          <h2 className="text-xl font-semibold">People</h2>
        </div>

        <div className="flex-1 min-h-0 relative">
          {graph && <ForceGraph graph={graph} selected={selected} onSelect={handleSelect} />}
        </div>

        <div className="flex-shrink-0 px-5 py-2.5 border-t border-[rgba(255,255,255,0.04)] text-[11px] text-text-tertiary">
          {entities.length} entities · {graph?.edges.length || 0} relationships
        </div>
      </div>

      <AnimatePresence>
        {selected && profile && (
          <motion.div
            initial={{ x: 380, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 380, opacity: 0 }}
            transition={springDefault}
            className="w-[380px] flex-shrink-0 border-l border-[rgba(255,255,255,0.06)] bg-bg-elevated h-full overflow-y-auto"
          >
            <div className="p-5">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold">{selected}</h3>
                <button onClick={() => setSelected(null)} className="text-text-tertiary hover:text-text-secondary text-lg">✕</button>
              </div>

              <ProfileSection profile={profile} />

              {recentMemories.length > 0 && (
                <div className="mt-5 pt-4 border-t border-[rgba(255,255,255,0.04)]">
                  <h4 className="text-xs text-text-tertiary mb-3">Recent Memories</h4>
                  <div className="space-y-2">
                    {recentMemories.map((m) => (
                      <div key={m.id} className="p-2.5 bg-bg-base rounded-lg">
                        <div className="flex items-center gap-2 mb-1">
                          <CategoryBadge category={m.category} />
                        </div>
                        <p className="text-[12px] text-text-primary/80 line-clamp-2">{m.content}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function ProfileSection({ profile }: { profile: Record<string, unknown> }) {
  const traits = profile.traits as string[] | Record<string, unknown> | undefined;
  const topics = profile.topics as string[] | undefined;
  const commStyle = profile.communication_style as Record<string, unknown> | undefined;
  const msgCount = profile.message_count as number | undefined;

  const traitList = Array.isArray(traits) ? traits : traits && typeof traits === "object" ? Object.keys(traits) : [];
  const topicList = Array.isArray(topics) ? topics : [];

  return (
    <div className="space-y-4">
      {msgCount != null && (
        <div className="text-sm text-text-secondary">{msgCount} memories</div>
      )}

      {traitList.length > 0 && (
        <div>
          <h4 className="text-xs text-text-tertiary mb-2">Traits</h4>
          <div className="flex flex-wrap gap-1.5">
            {traitList.map((t, i) => (
              <span key={i} className="text-[11px] px-2 py-0.5 rounded-full bg-accent/10 text-accent">
                {String(t)}
              </span>
            ))}
          </div>
        </div>
      )}

      {topicList.length > 0 && (
        <div>
          <h4 className="text-xs text-text-tertiary mb-2">Topics</h4>
          <div className="flex flex-wrap gap-1.5">
            {topicList.map((t, i) => (
              <span key={i} className="text-[11px] px-2 py-0.5 rounded-full bg-info/10 text-info">
                {t}
              </span>
            ))}
          </div>
        </div>
      )}

      {commStyle && (
        <div>
          <h4 className="text-xs text-text-tertiary mb-2">Communication Style</h4>
          <div className="space-y-1.5 text-[12px]">
            {Object.entries(commStyle).map(([k, v]) => (
              <div key={k} className="flex justify-between">
                <span className="text-text-tertiary">{k.replace(/_/g, " ")}</span>
                <span className="text-text-primary">{String(v)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ForceGraph({
  graph,
  selected,
  onSelect,
}: {
  graph: EntityGraph;
  selected: string | null;
  onSelect: (name: string) => void;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const selectedRef = useRef(selected);
  const onSelectRef = useRef(onSelect);
  selectedRef.current = selected;
  onSelectRef.current = onSelect;
  const circlesRef = useRef<d3.Selection<SVGCircleElement, any, SVGGElement, unknown> | null>(null);

  useEffect(() => {
    if (circlesRef.current) {
      circlesRef.current
        .attr("fill", (d: any) => d.id === selected ? "#6C5CE7" : "rgba(108, 92, 231, 0.25)")
        .attr("stroke", (d: any) => d.id === selected ? "#7C6CF7" : "rgba(255,255,255,0.12)")
        .attr("stroke-width", (d: any) => d.id === selected ? 2.5 : 1);
    }
  }, [selected]);

  useEffect(() => {
    if (!svgRef.current || !graph.nodes.length) return;

    const svg = d3.select(svgRef.current);
    const width = svgRef.current.clientWidth;
    const height = svgRef.current.clientHeight;

    svg.selectAll("*").remove();

    const g = svg.append("g");

    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.3, 3])
      .on("zoom", (event) => g.attr("transform", event.transform));
    svg.call(zoom);

    const nodeMap = new Map(graph.nodes.map((n) => [n.id, n]));
    const validEdges = graph.edges.filter(
      (e) => nodeMap.has(e.source as string) && nodeMap.has(e.target as string)
    );

    const simulation = d3.forceSimulation(graph.nodes as d3.SimulationNodeDatum[])
      .force("link", d3.forceLink(validEdges).id((d: any) => d.id).distance(140).strength(0.4))
      .force("charge", d3.forceManyBody().strength(-300))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius((d: any) => d.radius + 8));

    g.append("g")
      .selectAll("line")
      .data(validEdges)
      .join("line")
      .attr("stroke", "rgba(255,255,255,0.06)")
      .attr("stroke-width", (d: any) => Math.max(0.5, d.strength * 2));

    const node = g.append("g")
      .selectAll("g")
      .data(graph.nodes)
      .join("g")
      .style("cursor", "pointer")
      .on("click", (_, d: any) => onSelectRef.current(d.id))
      .call(
        d3.drag<SVGGElement, any>()
          .on("start", (event, d) => { if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
          .on("drag", (event, d) => { d.fx = event.x; d.fy = event.y; })
          .on("end", (event, d) => { if (!event.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }) as any
      );

    circlesRef.current = node.append("circle")
      .attr("r", (d: any) => d.radius)
      .attr("fill", "rgba(108, 92, 231, 0.25)")
      .attr("stroke", "rgba(255,255,255,0.12)")
      .attr("stroke-width", 1);

    node.append("text")
      .text((d: any) => d.id)
      .attr("text-anchor", "middle")
      .attr("dy", (d: any) => d.radius + 16)
      .attr("fill", "#EBEBF5")
      .attr("font-size", "12px")
      .attr("font-weight", "500")
      .attr("font-family", "-apple-system, BlinkMacSystemFont, system-ui");

    simulation.on("tick", () => {
      g.selectAll("line")
        .attr("x1", (d: any) => d.source.x)
        .attr("y1", (d: any) => d.source.y)
        .attr("x2", (d: any) => d.target.x)
        .attr("y2", (d: any) => d.target.y);
      node.attr("transform", (d: any) => `translate(${d.x},${d.y})`);
    });

    return () => { simulation.stop(); };
  }, [graph]);

  return (
    <svg ref={svgRef} className="w-full h-full" style={{ background: "transparent" }} />
  );
}
