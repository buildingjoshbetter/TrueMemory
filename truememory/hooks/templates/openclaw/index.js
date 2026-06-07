/**
 * TrueMemory plugin for OpenClaw.
 *
 * Hooks into the agent lifecycle to recall memories before each run
 * and trigger extraction after each run. Uses the TrueMemory MCP server
 * for memory operations.
 *
 * OpenClaw plugin API:
 *   export default { id, name, register(api) }
 *   api.on("session_start", handler)    — before agent starts processing
 *   api.on("session_end", handler)      — after agent finishes
 *   api.on("before_tool_call", handler) — before each tool invocation
 *   api.on("before_compaction", handler) — before context compaction
 */
import { spawnSync, spawn } from "child_process";
import { join } from "path";

const PYTHON_PATH = process.env.TRUEMEMORY_PYTHON || "python3";
const HOOKS_DIR = process.env.TRUEMEMORY_HOOKS_DIR || "";

function getHooksDir() {
  if (HOOKS_DIR) return HOOKS_DIR;
  try {
    const result = spawnSync(
      PYTHON_PATH,
      ["-c", "from pathlib import Path; import truememory; print(Path(truememory.__file__).parent / 'ingest' / 'hooks')"],
      { encoding: "utf-8", timeout: 10000 }
    );
    return (result.stdout || "").trim();
  } catch {
    return "";
  }
}

function runHookSync(hooksDir, script, input, timeoutMs) {
  try {
    const result = spawnSync(
      PYTHON_PATH,
      [join(hooksDir, script)],
      { input, encoding: "utf-8", timeout: timeoutMs }
    );
    return (result.stdout || "").trim();
  } catch {
    return "";
  }
}

export default {
  id: "truememory",
  name: "TrueMemory",
  description: "TrueMemory persistent memory integration for OpenClaw",

  register(api) {
    const hooksDir = getHooksDir();
    if (!hooksDir) {
      console.error("[truememory] Could not locate hook scripts");
      return;
    }

    let lastProcessedPrompt = null;
    let toolCallsSinceLastPrompt = 0;

    api.on("session_start", async (event) => {
      lastProcessedPrompt = null;
      toolCallsSinceLastPrompt = 0;
      try {
        const input = JSON.stringify({
          session_id: event.sessionId || "openclaw",
          cwd: process.cwd(),
          transcript_path: event.transcriptPath || "",
        });
        const result = runHookSync(hooksDir, "session_start.py", input, 10000);
        if (result) {
          const parsed = JSON.parse(result);
          if (parsed.additionalContext) {
            event.additionalContext = (event.additionalContext || "") + "\n" + parsed.additionalContext;
          }
        }
      } catch (err) {
        // Never block the agent run
      }
    });

    api.on("session_end", async (event) => {
      try {
        const input = JSON.stringify({
          session_id: event.sessionId || "openclaw",
          transcript_path: event.transcriptPath || "",
        });
        const child = spawn(PYTHON_PATH, [join(hooksDir, "stop.py")], {
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

    api.on("before_tool_call", async (event) => {
      const prompt = event.lastUserPrompt ?? event.userPrompt ?? null;

      if (prompt !== null) {
        if (!prompt || prompt === lastProcessedPrompt) return;
        lastProcessedPrompt = prompt;
      } else {
        // Field not present on this event type — use a counter to run
        // only on the first tool call after each prompt (heuristic).
        toolCallsSinceLastPrompt++;
        if (toolCallsSinceLastPrompt > 1) return;
      }

      try {
        const input = JSON.stringify({
          session_id: event.sessionId || "openclaw",
          cwd: process.cwd(),
          user_prompt: prompt || "",
        });
        runHookSync(hooksDir, "user_prompt_submit.py", input, 5000);
      } catch (err) {
        // Never block tool call processing
      }
    });

    api.on("before_compaction", async (event) => {
      try {
        const input = JSON.stringify({
          session_id: event.sessionId || "openclaw",
          transcript_path: event.transcriptPath || "",
        });
        runHookSync(hooksDir, "compact.py", input, 5000);
      } catch (err) {
        // Never block compression
      }
    });
  },
};
