import { useState, useEffect, useRef } from "react";
import { AnimatePresence } from "framer-motion";
import { useVirtualizer } from "@tanstack/react-virtual";
import { SearchBar } from "../components/SearchBar";
import { MemoryCard } from "../components/MemoryCard";
import { InspectorPanel } from "../components/InspectorPanel";
import { SkeletonCard } from "../components/SkeletonLoader";
import { useMemories } from "../hooks/useMemories";
import { api } from "../lib/api";
import { formatNumber, stripCategoryPrefix } from "../lib/formatters";
import type { Memory } from "../lib/types";

export function Explorer() {
  const { memories, total, loading, filters, updateFilter, loadMore, refetch } = useMemories();
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [senders, setSenders] = useState<string[]>([]);
  const [categories, setCategories] = useState<string[]>([]);
  const parentRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.memories.senders().then(setSenders);
    api.memories.categories().then(setCategories);
  }, []);

  const selectedMemory = memories.find((m) => m.id === selectedId) || null;

  const virtualizer = useVirtualizer({
    count: memories.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 76,
    overscan: 10,
  });

  const hasMore = memories.length < total;

  useEffect(() => {
    const el = parentRef.current;
    if (!el) return;
    const handleScroll = () => {
      if (el.scrollHeight - el.scrollTop - el.clientHeight < 200 && hasMore && !loading) {
        loadMore();
      }
    };
    el.addEventListener("scroll", handleScroll);
    return () => el.removeEventListener("scroll", handleScroll);
  }, [hasMore, loading, loadMore]);

  const handleUpdated = (updated: Memory) => {
    refetch();
    setSelectedId(updated.id);
  };

  const handleDeleted = () => {
    setSelectedId(null);
    refetch();
  };

  return (
    <div className="h-full flex flex-col">
      <div className="flex-shrink-0 px-5 pt-5 pb-3 space-y-3">
        <div className="flex items-center gap-3">
          <div className="flex-1">
            <SearchBar
              value={filters.search}
              onChange={(v) => updateFilter("search", v)}
            />
          </div>
          <FilterSelect
            value={filters.category}
            onChange={(v) => updateFilter("category", v)}
            options={categories}
            placeholder="Category"
          />
          <FilterSelect
            value={filters.sender}
            onChange={(v) => updateFilter("sender", v)}
            options={senders}
            placeholder="Sender"
          />
          {!filters.search && (
            <FilterSelect
              value={filters.sort}
              onChange={(v) => updateFilter("sort", v)}
              options={["newest", "oldest"]}
              placeholder="Sort"
            />
          )}
        </div>
      </div>

      <div className="flex-1 flex min-h-0">
        <div className="flex-1 flex flex-col min-w-0">
          {loading && memories.length === 0 ? (
            <div className="p-4 space-y-2">
              <SkeletonCard />
              <SkeletonCard />
              <SkeletonCard />
              <SkeletonCard />
              <SkeletonCard />
            </div>
          ) : memories.length === 0 ? (
            <div className="flex-1 flex items-center justify-center">
              <p className="text-text-tertiary text-sm">
                {filters.search ? "No results found" : "No memories"}
              </p>
            </div>
          ) : (
            <div ref={parentRef} className="flex-1 overflow-y-auto">
              <div
                style={{
                  height: `${virtualizer.getTotalSize()}px`,
                  width: "100%",
                  position: "relative",
                }}
              >
                {virtualizer.getVirtualItems().map((virtualItem) => {
                  const memory = memories[virtualItem.index];
                  return (
                    <div
                      key={memory.id}
                      style={{
                        position: "absolute",
                        top: 0,
                        left: 0,
                        width: "100%",
                        height: `${virtualItem.size}px`,
                        transform: `translateY(${virtualItem.start}px)`,
                      }}
                    >
                      <MemoryCard
                        memory={memory}
                        selected={selectedId === memory.id}
                        onClick={() => setSelectedId(selectedId === memory.id ? null : memory.id)}
                      />
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {hasMore && !loading && (
            <div className="flex-shrink-0 flex justify-center py-3">
              <button
                onClick={loadMore}
                className="px-4 py-1.5 text-xs rounded-lg bg-accent/15 text-accent hover:bg-accent/25 transition-colors"
              >
                Load More
              </button>
            </div>
          )}

          <div className="flex-shrink-0 px-5 py-2.5 border-t border-[rgba(255,255,255,0.04)] text-[11px] text-text-tertiary">
            {formatNumber(total)} memories
            {selectedId != null && " · 1 selected"}
          </div>
        </div>

        <AnimatePresence>
          {selectedMemory && (
            <InspectorPanel
              key={selectedMemory.id}
              memory={selectedMemory}
              onClose={() => setSelectedId(null)}
              onDeleted={handleDeleted}
              onUpdated={handleUpdated}
            />
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

function FilterSelect({
  value,
  onChange,
  options,
  placeholder,
}: {
  value: string;
  onChange: (v: string) => void;
  options: string[];
  placeholder: string;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="bg-bg-elevated border border-[rgba(255,255,255,0.06)] rounded-lg px-3 py-2.5 text-[13px] text-text-primary outline-none appearance-none cursor-pointer min-w-[100px]"
    >
      <option value="">{placeholder}</option>
      {options.map((opt) => (
        <option key={opt} value={opt}>
          {opt.charAt(0).toUpperCase() + opt.slice(1)}
        </option>
      ))}
    </select>
  );
}
