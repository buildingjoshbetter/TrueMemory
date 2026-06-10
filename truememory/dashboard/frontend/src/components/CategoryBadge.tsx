import { getCategoryColor } from "../lib/constants";

interface Props {
  category: string;
}

export function CategoryBadge({ category }: Props) {
  if (!category) return null;
  const { bg, text } = getCategoryColor(category);
  return (
    <span
      className="inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-medium whitespace-nowrap"
      style={{ backgroundColor: bg, color: text }}
    >
      {category}
    </span>
  );
}
