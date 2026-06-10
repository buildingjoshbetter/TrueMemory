import { motion } from "framer-motion";
import { springDefault } from "../lib/constants";

interface Props {
  children: React.ReactNode;
  className?: string;
  padding?: boolean;
}

export function GlassCard({ children, className = "", padding = true }: Props) {
  return (
    <motion.div
      layout
      transition={springDefault}
      className={`glass-panel ${padding ? "p-5" : ""} ${className}`}
    >
      {children}
    </motion.div>
  );
}
