import { useState, useEffect, useRef, useCallback } from "react";
import { api } from "../lib/api";
import type { Memory } from "../lib/types";

interface Filters {
  search: string;
  category: string;
  sender: string;
  sort: string;
}

export function useMemories() {
  const [memories, setMemories] = useState<Memory[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [filters, setFilters] = useState<Filters>({
    search: "",
    category: "",
    sender: "",
    sort: "newest",
  });
  const [offset, setOffset] = useState(0);
  const limit = 100;
  const debounceRef = useRef<ReturnType<typeof setTimeout>>();

  const fetchMemories = useCallback(async (f: Filters, off: number) => {
    try {
      setLoading(true);
      if (f.search) {
        const results = await api.memories.search(f.search, limit);
        let filtered = results;
        if (f.category) filtered = filtered.filter((m) => m.category === f.category);
        if (f.sender) filtered = filtered.filter((m) => m.sender === f.sender);
        setMemories(filtered);
        setTotal(filtered.length);
      } else {
        const resp = await api.memories.list({
          category: f.category || undefined,
          sender: f.sender || undefined,
          sort: f.sort,
          limit,
          offset: off,
        });
        if (off > 0) {
          setMemories(prev => [...prev, ...resp.memories]);
        } else {
          setMemories(resp.memories);
        }
        setTotal(resp.total);
      }
    } catch {
      // keep existing data on error
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      fetchMemories(filters, offset);
    }, filters.search ? 300 : 0);
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current); };
  }, [filters, offset, fetchMemories]);

  const updateFilter = useCallback((key: keyof Filters, value: string) => {
    setOffset(0);
    setFilters((prev) => ({ ...prev, [key]: value }));
  }, []);

  const loadMore = useCallback(() => {
    setOffset((prev) => prev + limit);
  }, []);

  const refetch = useCallback(() => {
    fetchMemories(filters, offset);
  }, [filters, offset, fetchMemories]);

  return { memories, total, loading, filters, updateFilter, loadMore, refetch, offset, limit };
}
