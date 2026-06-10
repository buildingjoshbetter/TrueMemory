export function SkeletonCard() {
  return (
    <div className="glass-panel p-4 animate-pulse">
      <div className="flex items-center gap-2 mb-3">
        <div className="h-5 w-16 rounded-full bg-bg-tertiary" />
        <div className="h-3 w-12 rounded bg-bg-tertiary ml-auto" />
      </div>
      <div className="space-y-2">
        <div className="h-3 w-full rounded bg-bg-tertiary" />
        <div className="h-3 w-3/4 rounded bg-bg-tertiary" />
      </div>
    </div>
  );
}

export function SkeletonStat() {
  return (
    <div className="glass-panel p-5 animate-pulse">
      <div className="h-8 w-20 rounded bg-bg-tertiary mb-2" />
      <div className="h-3 w-24 rounded bg-bg-tertiary mb-3" />
      <div className="h-7 w-full rounded bg-bg-tertiary" />
    </div>
  );
}
