import { useState } from "react";
import { motion } from "framer-motion";
import { springDefault } from "../lib/constants";
import { CategoryBadge } from "./CategoryBadge";
import { ConfirmSheet } from "./ConfirmSheet";
import { relativeTime, stripCategoryPrefix } from "../lib/formatters";
import { api } from "../lib/api";
import type { Memory } from "../lib/types";

interface Props {
  memory: Memory;
  onClose: () => void;
  onDeleted: () => void;
  onUpdated: (m: Memory) => void;
}

export function InspectorPanel({ memory, onClose, onDeleted, onUpdated }: Props) {
  const [editing, setEditing] = useState(false);
  const [editContent, setEditContent] = useState(memory.content);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [saving, setSaving] = useState(false);

  const handleSave = async () => {
    setSaving(true);
    try {
      const updated = await api.memories.update(memory.id, editContent);
      onUpdated(updated);
      setEditing(false);
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    await api.memories.delete(memory.id);
    setConfirmDelete(false);
    onDeleted();
  };

  return (
    <>
      <motion.div
        initial={{ x: 380, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        exit={{ x: 380, opacity: 0 }}
        transition={springDefault}
        className="w-[380px] flex-shrink-0 border-l border-[rgba(255,255,255,0.06)] bg-bg-elevated h-full overflow-y-auto"
      >
        <div className="p-5">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-text-secondary">Memory #{memory.id}</h3>
            <button
              onClick={onClose}
              className="text-text-tertiary hover:text-text-secondary text-lg leading-none"
            >
              ✕
            </button>
          </div>

          {editing ? (
            <div className="mb-4">
              <textarea
                value={editContent}
                onChange={(e) => setEditContent(e.target.value)}
                className="w-full h-40 bg-bg-base border border-[rgba(255,255,255,0.08)] rounded-lg p-3 text-[13px] font-mono text-text-primary resize-none outline-none focus:border-accent/40"
              />
              <div className="flex gap-2 mt-2">
                <button
                  onClick={handleSave}
                  disabled={saving}
                  className="px-3 py-1.5 text-xs rounded-lg bg-accent/20 text-accent hover:bg-accent/30 transition-colors font-medium"
                >
                  {saving ? "Saving…" : "Save"}
                </button>
                <button
                  onClick={() => { setEditing(false); setEditContent(memory.content); }}
                  className="px-3 py-1.5 text-xs rounded-lg bg-bg-grouped text-text-secondary hover:bg-bg-tertiary transition-colors"
                >
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            <div className="mb-4 p-3 bg-bg-base rounded-lg">
              <p className="text-[13px] font-mono text-text-primary/90 leading-relaxed whitespace-pre-wrap break-words">
                {stripCategoryPrefix(memory.content)}
              </p>
            </div>
          )}

          <div className="space-y-3 mb-6">
            <MetaRow label="Category">
              <CategoryBadge category={memory.category} />
            </MetaRow>
            <MetaRow label="Sender">
              <span className="text-sm">{memory.sender || "—"}</span>
            </MetaRow>
            <MetaRow label="Timestamp">
              <span className="text-sm">{memory.timestamp ? `${relativeTime(memory.timestamp)}` : "—"}</span>
            </MetaRow>
            {memory.timestamp && (
              <MetaRow label="Date">
                <span className="text-xs text-text-tertiary font-mono">
                  {new Date(memory.timestamp).toLocaleString()}
                </span>
              </MetaRow>
            )}
            {memory.score != null && (
              <MetaRow label="Score">
                <span className="text-sm font-mono text-accent">{memory.score.toFixed(4)}</span>
              </MetaRow>
            )}
            {memory.source && (
              <MetaRow label="Source">
                <span className="text-xs font-mono text-text-tertiary">{memory.source}</span>
              </MetaRow>
            )}
          </div>

          <div className="flex gap-2">
            {!editing && (
              <button
                onClick={() => { setEditing(true); setEditContent(memory.content); }}
                className="px-3 py-1.5 text-xs rounded-lg bg-bg-grouped text-text-primary hover:bg-bg-tertiary transition-colors"
              >
                Edit
              </button>
            )}
            <button
              onClick={() => setConfirmDelete(true)}
              className="px-3 py-1.5 text-xs rounded-lg bg-error/10 text-error hover:bg-error/20 transition-colors"
            >
              Delete
            </button>
          </div>
        </div>
      </motion.div>

      <ConfirmSheet
        open={confirmDelete}
        title="Delete Memory"
        message="This action cannot be undone. This memory will be permanently removed."
        confirmLabel="Delete"
        onConfirm={handleDelete}
        onCancel={() => setConfirmDelete(false)}
      />
    </>
  );
}

function MetaRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-start justify-between">
      <span className="text-xs text-text-tertiary">{label}</span>
      <div className="text-right">{children}</div>
    </div>
  );
}
