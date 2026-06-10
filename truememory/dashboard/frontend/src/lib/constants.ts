export const springDefault = { type: "spring" as const, stiffness: 400, damping: 30 };
export const springGentle = { type: "spring" as const, stiffness: 300, damping: 35 };
export const springSnappy = { type: "spring" as const, stiffness: 500, damping: 28 };

export const CATEGORY_COLORS: Record<string, { bg: string; text: string }> = {
  technical: { bg: "rgba(100, 210, 255, 0.15)", text: "#64D2FF" },
  preference: { bg: "rgba(108, 92, 231, 0.15)", text: "#A78BFA" },
  decision: { bg: "rgba(255, 214, 10, 0.15)", text: "#FFD60A" },
  personal: { bg: "rgba(48, 209, 88, 0.15)", text: "#30D158" },
  correction: { bg: "rgba(255, 69, 58, 0.15)", text: "#FF453A" },
  temporal: { bg: "rgba(100, 210, 255, 0.12)", text: "#5AC8FA" },
  relationship: { bg: "rgba(191, 90, 242, 0.15)", text: "#BF5AF2" },
  architecture: { bg: "rgba(100, 160, 255, 0.15)", text: "#64A0FF" },
  implementation: { bg: "rgba(48, 176, 199, 0.15)", text: "#30B0C7" },
};

export function getCategoryColor(category: string) {
  return CATEGORY_COLORS[category] || { bg: "rgba(142, 142, 147, 0.12)", text: "#8E8E93" };
}
