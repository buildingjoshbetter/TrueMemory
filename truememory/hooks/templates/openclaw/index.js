/**
 * TrueMemory plugin for OpenClaw.
 *
 * Hooks into the agent lifecycle to recall memories before each run
 * and trigger extraction after each run. Uses the TrueMemory MCP server
 * for memory operations.
 */
const { execSync, spawn } = require("child_process");
const path = require("path");

const PYTHON_PATH = process.env.TRUEMEMORY_PYTHON || "python3";
const HOOKS_DIR = process.env.TRUEMEMORY_HOOKS_DIR || "";

function getHooksDir() {
  if (HOOKS_DIR) return HOOKS_DIR;
  try {
    const out = execSync(
      `${PYTHON_PATH} -c "from pathlib import Path; import truememory; print(Path(truememory.__file__).parent / 'ingest' / 'hooks')"`,
      { encoding: "utf-8", timeout: 10000 }
    ).trim();
    return out;
  } catch {
    return "";
  }
}

module.exports = {
  register(api) {
    const hooksDir = getHooksDir();
    if (!hooksDir) {
      console.error("[truememory] Could not locate hook scripts");
      return;
    }

    api.on("before_agent_run", async (ctx) => {
      try {
        const input = JSON.stringify({
          session_id: ctx.sessionId || "openclaw",
          cwd: process.cwd(),
          transcript_path: ctx.transcriptPath || "",
        });
        const result = execSync(
          `echo '${input.replace(/'/g, "\\'")}' | ${PYTHON_PATH} ${path.join(hooksDir, "session_start.py")}`,
          { encoding: "utf-8", timeout: 10000 }
        ).trim();
        if (result) {
          const parsed = JSON.parse(result);
          if (parsed.additionalContext) {
            ctx.additionalContext = (ctx.additionalContext || "") + "\n" + parsed.additionalContext;
          }
        }
      } catch (err) {
        // Never block the agent run
      }
    });

    api.on("agent_end", async (ctx) => {
      try {
        const input = JSON.stringify({
          session_id: ctx.sessionId || "openclaw",
          transcript_path: ctx.transcriptPath || "",
        });
        const child = spawn(PYTHON_PATH, [path.join(hooksDir, "stop.py")], {
          stdio: ["pipe", "ignore", "ignore"],
          detached: true,
        });
        child.stdin.write(input);
        child.stdin.end();
        child.unref();
      } catch (err) {
        // Never block agent shutdown
      }
    });
  },
};
