import { GlassCard } from "./GlassCard";
import { Sparkline } from "./Sparkline";
import { formatNumber } from "../lib/formatters";

interface Props {
  value: number | string;
  label: string;
  sparklineData?: number[];
  color?: string;
}

export function StatCard({ value, label, sparklineData, color = "#6C5CE7" }: Props) {
  return (
    <GlassCard className="flex flex-col gap-2 min-w-0">
      <span className="text-[28px] font-bold leading-none tracking-tight">
        {typeof value === "number" ? formatNumber(value) : value}
      </span>
      <span className="text-xs text-text-secondary">{label}</span>
      {sparklineData && sparklineData.length > 0 && (
        <Sparkline data={sparklineData} color={color} height={28} />
      )}
    </GlassCard>
  );
}
