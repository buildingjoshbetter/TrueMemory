/**
 * TrueMemory plugin for OpenClaw.
 *
 * Hooks into the agent lifecycle to recall memories before each run
 * and trigger extraction after each run. Uses the TrueMemory MCP server
 * for memory operations.
 *
 * OpenClaw plugin API:
 *   module.exports.activate(ctx) — called once when the plugin loads.
 *   ctx.onSessionStart(cb)       — before agent starts processing.
 *   ctx.onSessionEnd(cb)         — after agent finishes.
 *   ctx.onToolCall(cb)           — after each tool invocation.
 *   ctx.onCompress(cb)           — before context compaction.
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
  activate(ctx) {
    const hooksDir = getHooksDir();
    if (!hooksDir) {
      console.error("[truememory] Could not locate hook scripts");
      return;
    }

    ctx.onSessionStart(async (session) => {
      try {
        const input = JSON.stringify({
          session_id: session.id || "openclaw",
          cwd: process.cwd(),
          transcript_path: session.transcriptPath || "",
        });
        const result = execSync(
          `echo '${input.replace(/'/g, "\\'")}' | ${PYTHON_PATH} ${path.join(hooksDir, "session_start.py")}`,
          { encoding: "utf-8", timeout: 10000 }
        ).trim();
        if (result) {
          const parsed = JSON.parse(result);
          if (parsed.additionalContext) {
            session.additionalContext = (session.additionalContext || "") + "\n" + parsed.additionalContext;
          }
        }
      } catch (err) {
        // Never block the agent run
      }
    });

    ctx.onSessionEnd(async (session) => {
      try {
        const input = JSON.stringify({
          session_id: session.id || "openclaw",
          transcript_path: session.transcriptPath || "",
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

    ctx.onToolCall(async (session) => {
      try {
        const input = JSON.stringify({
          session_id: session.id || "openclaw",
          cwd: process.cwd(),
          user_prompt: session.lastUserPrompt || "",
        });
        execSync(
          `echo '${input.replace(/'/g, "\\'")}' | ${PYTHON_PATH} ${path.join(hooksDir, "user_prompt_submit.py")}`,
          { encoding: "utf-8", timeout: 5000 }
        );
      } catch (err) {
        // Never block tool call processing
      }
    });

    ctx.onCompress(async (session) => {
      try {
        const input = JSON.stringify({
          session_id: session.id || "openclaw",
          transcript_path: session.transcriptPath || "",
        });
        execSync(
          `echo '${input.replace(/'/g, "\\'")}' | ${PYTHON_PATH} ${path.join(hooksDir, "compact.py")}`,
          { encoding: "utf-8", timeout: 5000 }
        );
      } catch (err) {
        // Never block compression
      }
    });
  },
};
