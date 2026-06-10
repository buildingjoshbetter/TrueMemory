import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Sidebar } from "./components/Sidebar";
import { Overview } from "./views/Overview";
import { Explorer } from "./views/Explorer";
import { Sessions } from "./views/Sessions";
import { People } from "./views/People";
import { Facts } from "./views/Facts";
import { Analytics } from "./views/Analytics";
import { Settings } from "./views/Settings";
import { useHealth } from "./hooks/useHealth";
import { springDefault } from "./lib/constants";
import type { ViewId } from "./lib/types";

export default function App() {
  const [view, setView] = useState<ViewId>("overview");
  const { data: health } = useHealth();

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-bg-base font-system">
      <Sidebar currentView={view} onNavigate={setView} health={health} />

      <main className="flex-1 min-w-0 h-full overflow-hidden">
        <AnimatePresence mode="wait">
          <motion.div
            key={view}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={springDefault}
            className="h-full"
          >
            {view === "overview" && (
              <Overview health={health} onNavigateExplorer={() => setView("explorer")} />
            )}
            {view === "explorer" && <Explorer />}
            {view === "sessions" && <Sessions />}
            {view === "people" && <People />}
            {view === "facts" && <Facts />}
            {view === "analytics" && <Analytics />}
            {view === "settings" && <Settings health={health} />}
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}
