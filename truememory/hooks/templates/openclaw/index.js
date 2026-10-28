/** Native OpenClaw recall transport. Run IDs correlate work, not admitted turns. */
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";

const INSTALLED_CONFIG = {"pythonPath": "python3", "userId": "", "dbPath": ""};
const WORKER_TIMEOUT_MS = 4000;
const MAX_PROMPT_BYTES = 128 * 1024;
const MAX_OUTPUT_BYTES = 64 * 1024;
const MAX_WORKERS = 4;

function validIdentity(value) {
  return typeof value === "string" && value.trim().length > 0 &&
    value.length <= 512 && !/[\x00-\x1f\x7f]/.test(value);
}

function assertInvocationCurrent(ctx) {
  if (ctx.hookInvocation !== undefined) {
    if (typeof ctx.hookInvocation?.assertActive !== "function") throw new Error("Invalid invocation");
    ctx.hookInvocation.assertActive();
  }
  if (ctx.assertMemoryAudienceCurrent !== undefined) {
    if (typeof ctx.assertMemoryAudienceCurrent !== "function") throw new Error("Invalid audience");
    ctx.assertMemoryAudienceCurrent();
  }
}

function recallWorker(config, input, workers) {
  return new Promise((resolve) => {
    let child;
    let timer;
    let settled = false;
    let outputBytes = 0;
    const chunks = [];
    const finish = (result) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve(result);
    };
    const rejectWorker = () => {
      finish(undefined);
      try {
        if (child) child.kill("SIGKILL");
      } catch {
        return;
      }
    };
    try {
      child = spawn(config.pythonPath, [
        "-m", "truememory.ingest.hooks.openclaw", "--event", "before_prompt_build",
        "--user", config.userId, "--db", config.dbPath,
      ], { stdio: ["pipe", "pipe", "ignore"], shell: false });
      workers.add(child);
      timer = setTimeout(rejectWorker, WORKER_TIMEOUT_MS);
      child.on("error", rejectWorker);
      child.stdin.on("error", rejectWorker);
      child.stdout.on("error", rejectWorker);
      child.stdout.on("data", (chunk) => {
        if (settled) return;
        const bytes = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
        outputBytes += bytes.length;
        if (outputBytes > MAX_OUTPUT_BYTES) {
          rejectWorker();
          return;
        }
        chunks.push(bytes);
      });
      child.on("close", (code, signal) => {
        workers.delete(child);
        if (settled) return;
        if (code !== 0 || signal) {
          finish(undefined);
          return;
        }
        try {
          const result = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(Buffer.concat(chunks)));
          if (!result || Array.isArray(result) || Object.keys(result).length !== 1 ||
              typeof result.context !== "string") {
            finish(undefined);
            return;
          }
          finish(result.context);
        } catch {
          finish(undefined);
        }
      });
      child.stdin.end(JSON.stringify(input));
    } catch {
      rejectWorker();
    }
  });
}

export default {
  id: "truememory",
  name: "TrueMemory",
  description: "TrueMemory recall-only integration for OpenClaw",

  register(api) {
    const supplied = api.pluginConfig ?? {};
    const config = {
      pythonPath: supplied.pythonPath ?? INSTALLED_CONFIG.pythonPath,
      userId: supplied.userId ?? INSTALLED_CONFIG.userId,
      dbPath: supplied.dbPath ?? INSTALLED_CONFIG.dbPath,
    };
    if (!config.pythonPath || Object.values(config).some((value) =>
      typeof value !== "string" || value.length > 8192 || value.includes("\0"))) {
      api.logger?.warn("TrueMemory recall configuration is invalid");
      return;
    }
    const pending = new Map();
    const workers = new Set();
    api.on("before_prompt_build", async (event, ctx) => {
      try {
        if (!event || !ctx || ctx.trigger !== "user" ||
            !validIdentity(ctx.sessionId) || !validIdentity(ctx.runId)) return;
        if ([ctx.agentId, ctx.sessionKey].some((value) =>
          value !== undefined && !validIdentity(value))) return;
        if (ctx.inputProvenance !== undefined && ctx.inputProvenance?.kind !== "external_user") return;
        const prompt = Object.hasOwn(event, "currentUserMessage") ? event.currentUserMessage : event.prompt;
        const hasAdmissionId = Object.hasOwn(event, "currentUserMessageId");
        const admissionId = hasAdmissionId ? event.currentUserMessageId : null;
        if (hasAdmissionId && !validIdentity(admissionId)) return;
        if (typeof prompt !== "string" || prompt.trim().length < 3 ||
            Buffer.byteLength(prompt, "utf8") > MAX_PROMPT_BYTES || prompt.includes("\0")) return;
        assertInvocationCurrent(ctx);
        const sessionId = "openclaw_" + createHash("sha256")
          .update(JSON.stringify([ctx.agentId ?? "", ctx.sessionKey ?? "", ctx.sessionId]))
          .digest("hex").slice(0, 48);
        const runId = createHash("sha256").update(ctx.runId).digest("hex");
        const key = createHash("sha256").update(JSON.stringify([sessionId, runId, admissionId, prompt])).digest("hex");
        let work = pending.get(key);
        if (!work) {
          if (workers.size >= MAX_WORKERS) return;
          work = recallWorker(config, { session_id: sessionId, run_id: runId, prompt }, workers);
          pending.set(key, work);
          work.finally(() => {
            if (pending.get(key) === work) pending.delete(key);
          });
        }
        const context = await work;
        assertInvocationCurrent(ctx);
        if (typeof context === "string" && context.trim()) return { prependContext: context };
      } catch {
        return;
      }
    });
  },
};
