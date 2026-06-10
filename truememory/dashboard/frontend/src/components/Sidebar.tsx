import type { ViewId, HealthResponse } from "../lib/types";

interface Props {
  currentView: ViewId;
  onNavigate: (view: ViewId) => void;
  health: HealthResponse | null;
}

const navItems: { id: ViewId; label: string; icon: React.ReactNode }[] = [
  {
    id: "overview",
    label: "Overview",
    icon: (
      <svg className="w-[18px] h-[18px]" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.8}>
        <rect x="3" y="3" width="7" height="7" rx="1.5" />
        <rect x="14" y="3" width="7" height="7" rx="1.5" />
        <rect x="3" y="14" width="7" height="7" rx="1.5" />
        <rect x="14" y="14" width="7" height="7" rx="1.5" />
      </svg>
    ),
  },
  {
    id: "explorer",
    label: "Explorer",
    icon: (
      <svg className="w-[18px] h-[18px]" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.8}>
        <circle cx="11" cy="11" r="8" />
        <path d="m21 21-4.35-4.35" strokeLinecap="round" />
      </svg>
    ),
  },
];

const proItems: { id: ViewId; label: string; icon: React.ReactNode }[] = [
  {
    id: "sessions",
    label: "Sessions",
    icon: (
      <svg className="w-[18px] h-[18px]" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.8}>
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
      </svg>
    ),
  },
  {
    id: "people",
    label: "People",
    icon: (
      <svg className="w-[18px] h-[18px]" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.8}>
        <circle cx="9" cy="7" r="4" />
        <path d="M3 21v-2a4 4 0 0 1 4-4h4a4 4 0 0 1 4 4v2" />
        <circle cx="17" cy="7" r="3" />
        <path d="M21 21v-2a3 3 0 0 0-2-2.83" />
      </svg>
    ),
  },
  {
    id: "facts",
    label: "Facts",
    icon: (
      <svg className="w-[18px] h-[18px]" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.8}>
        <path d="M12 2L2 7l10 5 10-5-10-5z" />
        <path d="M2 17l10 5 10-5" />
        <path d="M2 12l10 5 10-5" />
      </svg>
    ),
  },
  {
    id: "analytics",
    label: "Analytics",
    icon: (
      <svg className="w-[18px] h-[18px]" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.8}>
        <path d="M3 3v18h18" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M7 16l4-8 4 4 4-8" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
];

const settingsItem: { id: ViewId; label: string; icon: React.ReactNode } = {
  id: "settings",
  label: "Settings",
  icon: (
    <svg className="w-[18px] h-[18px]" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.8}>
      <circle cx="12" cy="12" r="3" />
      <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" strokeLinecap="round" />
    </svg>
  ),
};

export function Sidebar({ currentView, onNavigate, health }: Props) {
  const tierColors: Record<string, string> = {
    edge: "#30D158",
    base: "#64D2FF",
    pro: "#BF5AF2",
  };
  const tier = health?.tier || "edge";

  const renderNavButton = (item: { id: ViewId; label: string; icon: React.ReactNode }) => (
    <button
      key={item.id}
      onClick={() => onNavigate(item.id)}
      className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-[13px] transition-colors ${
        currentView === item.id
          ? "bg-accent-muted text-text-primary"
          : "text-text-secondary hover:bg-bg-elevated hover:text-text-primary"
      }`}
    >
      <span className={currentView === item.id ? "text-accent" : ""}>{item.icon}</span>
      {item.label}
    </button>
  );

  return (
    <div className="w-[220px] flex-shrink-0 glass-sidebar flex flex-col h-full select-none">
      <div className="px-5 pt-6 pb-4">
        <h1 className="text-[15px] font-semibold tracking-tight">TrueMemory</h1>
      </div>

      <nav className="flex-1 px-3">
        <div className="space-y-0.5">
          {navItems.map(renderNavButton)}
        </div>

        <div className="mt-6 pt-4 border-t border-[rgba(255,255,255,0.04)]">
          <div className="space-y-0.5">
            {proItems.map(renderNavButton)}
          </div>
        </div>

        <div className="mt-6 pt-4 border-t border-[rgba(255,255,255,0.04)]">
          <div className="space-y-0.5">
            {renderNavButton(settingsItem)}
          </div>
        </div>
      </nav>

      <div className="px-4 py-4 border-t border-[rgba(255,255,255,0.04)]">
        <div className="flex items-center gap-2 mb-1">
          <span
            className="text-[10px] font-bold uppercase px-1.5 py-0.5 rounded"
            style={{
              backgroundColor: `${tierColors[tier] || "#8E8E93"}20`,
              color: tierColors[tier] || "#8E8E93",
            }}
          >
            {tier}
          </span>
          <span className="text-[11px] text-text-tertiary">
            v{health?.version || "…"}
          </span>
        </div>
        <p className="text-[11px] text-text-tertiary">
          {health ? `${health.memory_count.toLocaleString()} memories` : "Loading…"}
        </p>
      </div>
    </div>
  );
}
