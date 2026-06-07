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
import { execSync, spawn } from "child_process";
import { join } from "path";

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

    // Track the last user prompt we processed to avoid running
    // user_prompt_submit.py multiple times for the same prompt.
    // OpenClaw fires before_tool_call on every tool invocation, but
    // user_prompt_submit.py is designed for once-per-user-message semantics.
    let lastProcessedPrompt = null;

    api.on("session_start", async (event) => {
      lastProcessedPrompt = null;
      try {
        const input = JSON.stringify({
          session_id: event.sessionId || "openclaw",
          cwd: process.cwd(),
          transcript_path: event.transcriptPath || "",
        });
        const result = execSync(
          `echo '${input.replace(/'/g, "\\'")}' | ${PYTHON_PATH} ${join(hooksDir, "session_start.py")}`,
          { encoding: "utf-8", timeout: 10000 }
        ).trim();
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
      // Deduplicate: only run user_prompt_submit.py once per unique user
      // prompt. before_tool_call fires on every tool invocation, but the
      // hook buffers messages and triggers recall — both should happen
      // once per user message, not once per tool call.
      //
      // If event.lastUserPrompt is not available on this event type,
      // fall back to running on every call rather than silently skipping.
      const prompt = event.lastUserPrompt ?? event.userPrompt ?? null;
      if (prompt === null) {
        // Field not present — run unconditionally rather than becoming a no-op
      } else if (!prompt || prompt === lastProcessedPrompt) {
        return;
      }
      if (prompt) lastProcessedPrompt = prompt;

      try {
        const input = JSON.stringify({
          session_id: event.sessionId || "openclaw",
          cwd: process.cwd(),
          user_prompt: prompt,
        });
        execSync(
          `echo '${input.replace(/'/g, "\\'")}' | ${PYTHON_PATH} ${join(hooksDir, "user_prompt_submit.py")}`,
          { encoding: "utf-8", timeout: 5000 }
        );
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
        execSync(
          `echo '${input.replace(/'/g, "\\'")}' | ${PYTHON_PATH} ${join(hooksDir, "compact.py")}`,
          { encoding: "utf-8", timeout: 5000 }
        );
      } catch (err) {
        // Never block compression
      }
    });
  },
};
