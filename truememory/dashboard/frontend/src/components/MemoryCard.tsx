import React from "react";
import { CategoryBadge } from "./CategoryBadge";
import { relativeTime, truncate, stripCategoryPrefix } from "../lib/formatters";
import type { Memory } from "../lib/types";

interface Props {
  memory: Memory;
  selected: boolean;
  onClick: () => void;
}

export const MemoryCard = React.memo(function MemoryCard({ memory, selected, onClick }: Props) {
  return (
    <button
      onClick={onClick}
      className={`w-full text-left px-4 py-3 border-l-2 transition-colors ${
        selected
          ? "bg-accent-muted border-l-accent"
          : "border-l-transparent hover:bg-bg-elevated"
      }`}
    >
      <div className="flex items-center gap-2 mb-1.5">
        <CategoryBadge category={memory.category} />
        {memory.sender && (
          <span className="text-[11px] text-text-tertiary">{memory.sender}</span>
        )}
        <span className="text-[11px] text-text-tertiary ml-auto whitespace-nowrap">
          {relativeTime(memory.timestamp)}
        </span>
        {memory.score != null && (
          <span className="text-[10px] text-accent font-mono">{memory.score.toFixed(2)}</span>
        )}
      </div>
      <p className="text-[13px] text-text-primary/90 leading-snug line-clamp-2">
        {truncate(stripCategoryPrefix(memory.content), 200)}
      </p>
    </button>
  );
});
