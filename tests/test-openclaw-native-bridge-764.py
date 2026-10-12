"""Actual-source native recall controls with inert collaborators and no package load.

Python controls run under pytest or stdlib unittest. Explicit --js-controls
requires a working Bun executable and fails if it is absent; it never skips.
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import os
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "truememory/ingest/hooks/openclaw.py"
ADAPTER = ROOT / "truememory/hooks/adapters/openclaw.py"
TEMPLATE = ROOT / "truememory/hooks/templates/openclaw/index.js"
SESSION = "openclaw_" + "a" * 48
RUN = "b" * 64


class NativeImports(ast.NodeTransformer):
    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST | None:
        if (node.module or "").startswith("truememory"):
            return None
        return node


def actual_source(path: Path, injected: dict[str, object]) -> dict[str, object]:
    tree = NativeImports().visit(ast.parse(path.read_text(encoding="utf-8")))
    ast.fix_missing_locations(tree)
    namespace = {"__name__": "synthetic_openclaw764", "__file__": str(path), **injected}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


class OpenClawNativeBridgeTests(unittest.TestCase):
    def invoke(self, payload: object, result: object = "synthetic recall", *, raw: bool = False,
               extracting: bool = False) -> tuple[list[tuple[object, ...]], str]:
        calls: list[tuple[object, ...]] = []

        def recall(*args: object, **kwargs: object) -> object:
            calls.append((*args, kwargs))
            return result

        namespace = actual_source(WRAPPER, {"_try_auto_recall": recall})
        output = io.StringIO()
        text = payload if raw else json.dumps(payload)
        with patch.object(sys, "argv", ["openclaw", "--event", "before_prompt_build",
                                        "--user", "synthetic-user", "--db", "/synthetic/db"]), \
                patch.object(sys, "stdin", io.StringIO(text)), patch.object(sys, "stdout", output), \
                patch.dict(os.environ, {"TRUEMEMORY_EXTRACTION": "1" if extracting else ""}):
            namespace["main"]()
        return calls, output.getvalue()

    def payload(self, prompt: object = "What do you remember about synthetic tea?") -> dict[str, object]:
        return {"session_id": SESSION, "run_id": RUN, "prompt": prompt}

    def test_native_request_calls_unchanged_helper_with_installed_identity(self) -> None:
        calls, output = self.invoke(self.payload())
        self.assertEqual(calls, [(self.payload()["prompt"], "synthetic-user", "/synthetic/db",
                                 SESSION, {"debounced": False})])
        self.assertEqual(json.loads(output), {"context": "synthetic recall"})

    def test_missing_malformed_identity_and_current_text_never_recall(self) -> None:
        invalid = [None, [], {}, self.payload(None), self.payload([]), self.payload({}),
                   self.payload(""), self.payload("ab"), self.payload("  "), self.payload("abc\0")]
        for key in ("session_id", "run_id"):
            for value in (None, "", "unknown", [], "x" * 512):
                invalid.append({**self.payload(), key: value})
            invalid.append({k: v for k, v in self.payload().items() if k != key})
        for payload in invalid:
            with self.subTest(payload=payload):
                calls, output = self.invoke(payload)
                self.assertEqual(calls, [])
                self.assertEqual(output, "")

    def test_native_paths_history_and_claimed_host_are_rejected(self) -> None:
        for key, value in (("transcript_path", "/synthetic/forged.jsonl"),
                           ("messages", [{"role": "user", "content": "historical"}]),
                           ("host", "claude"), ("cwd", "/synthetic"),
                           ("user_id", "stdin-user"), ("db_path", "/stdin/db")):
            calls, output = self.invoke({**self.payload(), key: value})
            self.assertEqual(calls, [])
            self.assertEqual(output, "")

    def test_bounded_input_and_utf8_do_not_truncate_accepted_text(self) -> None:
        for prompt in ("x" * (128 * 1024 + 1), "é" * (128 * 1024), "\ud800"):
            calls, output = self.invoke(self.payload(prompt))
            self.assertEqual(calls, [])
            self.assertEqual(output, "")
        prompt = "synthetic\nquoted \" tea é"
        calls, _ = self.invoke(self.payload(prompt))
        self.assertEqual(calls[0][0], prompt)

    def test_malformed_json_and_non_json_input_are_inert(self) -> None:
        for text in ("", "{", "null", "[]", "{} trailing", "x" * (132 * 1024 + 1)):
            calls, output = self.invoke(text, raw=True)
            self.assertEqual(calls, [])
            self.assertEqual(output, "")

    def test_no_recall_has_exact_native_empty_context(self) -> None:
        calls, output = self.invoke(self.payload(), result=None)
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(output), {"context": ""})
        self.assertNotIn("additionalContext", output)

    def test_nonstring_helper_output_never_crosses_transport(self) -> None:
        for result in ([], {}, 42):
            calls, output = self.invoke(self.payload(), result=result)
            self.assertEqual(len(calls), 1)
            self.assertEqual(output, "")

    def test_extraction_recursion_is_inert(self) -> None:
        calls, output = self.invoke(self.payload(), extracting=True)
        self.assertEqual(calls, [])
        self.assertEqual(output, "")

    def test_wrapper_has_no_capture_lifecycle_or_shared_main(self) -> None:
        tree = ast.parse(WRAPPER.read_text(encoding="utf-8"))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.assertTrue({"_try_capture_email", "buffer_message", "_prune_old_buffers",
                         "_try_per_exchange_store", "_increment_prompt_count", "emit_context",
                         "_run_background_ingestion"}.isdisjoint(names))
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                   and (node.module or "").startswith("truememory")]
        self.assertEqual(imports, ["truememory.ingest.hooks.user_prompt_submit"])

    def test_installer_serializes_user_db_and_interpreter_without_execution(self) -> None:
        written: dict[str, str] = {}
        namespace = actual_source(ADAPTER, {"CLIAdapter": object,
                                            "atomic_write_text": lambda p, t: written.update({str(p): t})})
        namespace["_PLUGINS_DIR"] = Path("/synthetic/.openclaw/extensions")
        config_path = Path("/synthetic/.openclaw/openclaw.json")
        namespace["_CONFIG_PATH"] = config_path
        adapter = namespace["OpenClawAdapter"]()
        python = '/synthetic/quote"/python\\exe'
        user, db = 'synthetic-"user', '/synthetic/db "with spaces"'
        original_exists = Path.exists
        with patch.object(Path, "mkdir"), patch.object(Path, "exists", lambda p:
                False if p == config_path else original_exists(p)):
            adapter.install_hooks(python, user, db)
            first = written.copy()
            adapter.install_hooks(python, user, db)
        self.assertEqual(first, written)
        js = written["/synthetic/.openclaw/extensions/truememory/index.js"]
        installed = re.search(r"^const INSTALLED_CONFIG = (.*);$", js, re.MULTILINE)
        self.assertIsNotNone(installed)
        self.assertEqual(json.loads(installed.group(1)), {"pythonPath": python, "userId": user, "dbPath": db})
        self.assertEqual(set(Path(path).name for path in written),
                         {"index.js", "package.json", "openclaw.plugin.json", "openclaw.json"})
        self.assertFalse(adapter.has_session_start)

    def permission_install(self, seed: object, *, raw: bool = False) -> dict[str, str]:
        written: dict[str, str] = {}
        namespace = actual_source(ADAPTER, {"CLIAdapter": object,
                                            "atomic_write_text": lambda p, t: written.update({str(p): t})})
        config_path = Path("/synthetic/.openclaw/openclaw.json")
        namespace["_CONFIG_PATH"] = config_path
        namespace["_PLUGINS_DIR"] = Path("/synthetic/.openclaw/extensions")
        original_read = Path.read_text
        original_exists = Path.exists
        text = seed if raw else json.dumps(seed)
        with patch.object(Path, "mkdir"), patch.object(Path, "exists", lambda p:
                True if p == config_path else original_exists(p)), \
                patch.object(Path, "read_text", lambda p, *args, **kwargs:
                             text if p == config_path else original_read(p, *args, **kwargs)):
            namespace["OpenClawAdapter"]().install_hooks("/synthetic/python")
        return written

    def test_permission_grants_only_absent_owned_conversation_access(self) -> None:
        seed = {"mcp": {"servers": {"other": {"command": "foreign"}}}, "unrelated": 42,
                "plugins": {"enabled": False, "allow": ["other"], "deny": ["truememory"],
                            "entries": {"other": {"enabled": False, "hooks": {"allowConversationAccess": False}},
                                        "truememory": {"enabled": False, "config": {"foreign": "keep"},
                                                       "hooks": {"allowPromptInjection": False}}}}}
        expected = json.loads(json.dumps(seed))
        expected["plugins"]["entries"]["truememory"]["hooks"]["allowConversationAccess"] = True
        written = self.permission_install(seed)
        self.assertEqual(json.loads(written["/synthetic/.openclaw/openclaw.json"]), expected)
        self.assertEqual(seed["plugins"]["deny"], ["truememory"])
        self.assertFalse(seed["plugins"]["entries"]["truememory"]["hooks"]["allowPromptInjection"])

    def test_permission_preserves_explicit_denials_and_malformed_configs(self) -> None:
        for value in (False, True, None, "false"):
            seed = {"plugins": {"entries": {"truememory": {"hooks": {"allowConversationAccess": value}}}}}
            written = self.permission_install(seed)
            self.assertNotIn("/synthetic/.openclaw/openclaw.json", written)
            self.assertIn("/synthetic/.openclaw/extensions/truememory/index.js", written)
        for seed in ([], {"plugins": False}, {"plugins": {"entries": []}},
                     {"plugins": {"entries": {"truememory": False}}},
                     {"plugins": {"entries": {"truememory": {"hooks": False}}}}):
            self.assertNotIn("/synthetic/.openclaw/openclaw.json", self.permission_install(seed))
        self.assertNotIn("/synthetic/.openclaw/openclaw.json", self.permission_install("{", raw=True))


def js_controls(bun: str) -> None:
    source = TEMPLATE.read_text(encoding="utf-8")
    source = source.replace('import { spawn } from "node:child_process";', "")
    source = source.replace("export default {", "const plugin = {")
    harness = r'''
const checks = [];
function check(name, condition) {
  if (!condition) throw new Error(name);
  checks.push(name);
}
let plan = [];
let calls = [];
let children = [];
function spawn(command, args, options) {
  const step = plan.shift() ?? {};
  const child = new EventEmitter();
  child.stdout = new EventEmitter();
  child.stdin = new EventEmitter();
  child.kill = (signal) => {
    child.killed = signal;
    if (!step.suppressClose) queueMicrotask(() => child.emit("close", null, signal));
    return true;
  };
  child.stdin.end = (input) => {
    calls.push({command, args, options, input: JSON.parse(input)});
    if (step.hang) return;
    const deliver = () => {
      if (step.error) { child.emit("error", new Error("synthetic")); return; }
      for (const chunk of step.chunks ?? [JSON.stringify({context: "synthetic recall"})]) {
        child.stdout.emit("data", Buffer.from(chunk));
      }
      if (!step.suppressClose) child.emit("close", step.code ?? 0, step.signal ?? null);
    };
    if (step.delayed) child.deliver = deliver;
    else queueMicrotask(deliver);
  };
  children.push(child);
  return child;
}
const nativeSetTimeout = globalThis.setTimeout;
const nativeClearTimeout = globalThis.clearTimeout;
const timers = new Map();
let timerId = 0;
globalThis.setTimeout = (fn, delay) => {
  const id = ++timerId;
  timers.set(id, {fn, delay});
  return id;
};
globalThis.clearTimeout = (id) => timers.delete(id);
const handlers = new Map();
plugin.register({pluginConfig: {pythonPath: "/synthetic/python", userId: "synthetic-user", dbPath: "/synthetic/db"},
                 on: (event, fn) => handlers.set(event, fn)});
check("single native prompt registration", handlers.size === 1 && handlers.has("before_prompt_build"));
const handler = handlers.get("before_prompt_build");
const ctx = (session = "synthetic-session", run = "synthetic-run") => ({sessionId: session, runId: run, trigger: "user"});
const event = {prompt: "What do you remember about synthetic tea?", messages: [{role: "user", content: "historical"}],
               transcriptPath: "/synthetic/forged", user_id: "stdin-user", db_path: "/stdin/db"};
const result = await handler(event, ctx());
check("native prependContext", result?.prependContext === "synthetic recall");
check("fixed native argv identity", JSON.stringify(calls[0].args) === JSON.stringify([
  "-m", "truememory.ingest.hooks.openclaw", "--event", "before_prompt_build", "--user", "synthetic-user", "--db", "/synthetic/db"]));
check("no shell or discovery worker", calls[0].command === "/synthetic/python" && calls[0].options.shell === false);
check("exact bounded native payload", Object.keys(calls[0].input).sort().join() === "prompt,run_id,session_id" &&
  /^openclaw_[0-9a-f]{48}$/.test(calls[0].input.session_id) && /^[0-9a-f]{64}$/.test(calls[0].input.run_id));
const before = calls.length;
await handler({...event, currentUserMessage: ""}, ctx());
await handler({...event, prompt: ""}, ctx());
await handler({...event, prompt: []}, ctx());
await handler({...event, prompt: "x".repeat(128 * 1024 + 1)}, ctx());
for (const bad of [{trigger: "user"}, {...ctx(), sessionId: ""}, {...ctx(), runId: null},
                   {...ctx(), sessionId: []}, {...ctx(), trigger: "heartbeat"},
                   {...ctx(), inputProvenance: {kind: "inter_session"}},
                   {...ctx(), hookInvocation: {assertActive() {throw new Error("stale");}}}]) await handler(event, bad);
check("malformed missing empty and stale never spawn", calls.length === before);
await handler(event, ctx("different-session"));
check("equal text sessions separated", calls.at(-1).input.session_id !== calls[0].input.session_id);
await handler(event, ctx("synthetic-session", "different-run"));
check("equal text runs separated", calls.at(-1).input.run_id !== calls[0].input.run_id);
const rebuildBefore = calls.length;
await handler(event, ctx());
await handler(event, ctx());
check("sequential rebuild re-evaluates without completed suppression", calls.length === rebuildBefore + 2);
plan.push({delayed: true});
const concurrentBefore = calls.length;
const first = handler(event, ctx("concurrent"));
const second = handler(event, ctx("concurrent"));
check("only in-flight identical work coalesces", calls.length === concurrentBefore + 1);
children.at(-1).deliver();
const pair = await Promise.all([first, second]);
check("both concurrent invocations return supported context", pair.every((value) => value?.prependContext === "synthetic recall"));
for (const admission of ["", "   ", null, [], {}, 42, "x".repeat(513), "synthetic\0id", "synthetic\nid"]) {
  const admissionBefore = calls.length;
  check("explicit invalid admission ID rejected " + JSON.stringify(admission).slice(0, 80),
    await handler({...event, currentUserMessageId: admission}, ctx("admission")) === undefined &&
    calls.length === admissionBefore);
}
plan.push({delayed: true}, {delayed: true});
const distinctAdmissionBefore = calls.length;
const distinctFirst = handler({...event, currentUserMessageId: "synthetic-admission-1"}, ctx("admission"));
const distinctSecond = handler({...event, currentUserMessageId: "synthetic-admission-2"}, ctx("admission"));
check("distinct explicit admissions do not coalesce", calls.length === distinctAdmissionBefore + 2);
for (const child of children.slice(-2)) child.deliver();
check("distinct explicit admissions both return context", (await Promise.all([distinctFirst, distinctSecond]))
  .every((value) => value?.prependContext === "synthetic recall"));
plan.push({delayed: true});
const sameAdmissionBefore = calls.length;
const sameFirst = handler({...event, currentUserMessageId: "synthetic-admission-3"}, ctx("admission"));
const sameSecond = handler({...event, currentUserMessageId: "synthetic-admission-3"}, ctx("admission"));
check("identical in-flight explicit admission coalesces", calls.length === sameAdmissionBefore + 1);
children.at(-1).deliver();
check("identical explicit admission returns context to both", (await Promise.all([sameFirst, sameSecond]))
  .every((value) => value?.prependContext === "synthetic recall"));
for (const step of [{code: 7}, {signal: "SIGTERM"}, {chunks: ["{"]}, {chunks: ["null"]},
                    {chunks: ["{}"]}, {chunks: ['{"context":42}']},
                    {chunks: ['{"context":"synthetic","unexpected":true}']},
                    {chunks: [Buffer.from('{"context":"'), Buffer.from([255]), Buffer.from('"}')]},
                    {chunks: ["x".repeat(64 * 1024 + 1)]}, {error: true}]) {
  plan.push(step);
  check("failed output rejected " + JSON.stringify(step).slice(0, 100), await handler(event, ctx("failure")) === undefined);
  check("failed work remains retryable", (await handler(event, ctx("failure")))?.prependContext === "synthetic recall");
}
plan.push({chunks: ['{"context":""}']});
check("empty recall is not fabricated context", await handler(event, ctx("empty")) === undefined);
let active = true;
plan.push({delayed: true});
const expired = handler(event, {...ctx("expired"), hookInvocation: {assertActive() {
  if (!active) throw new Error("expired");
}}});
active = false;
children.at(-1).deliver();
check("invocation lifetime checked after await", await expired === undefined);
plan.push({hang: true});
const timeout = handler(event, ctx("timeout"));
check("worker deadline bounded to four seconds", [...timers.values()].every((timer) => timer.delay === 4000));
for (const timer of [...timers.values()]) timer.fn();
check("timeout returns without accepting output", await timeout === undefined);
check("timeout kills owned child", children.at(-1).killed === "SIGKILL");
await Promise.resolve();
check("timed out work remains retryable", (await handler(event, ctx("timeout")))?.prependContext === "synthetic recall");
for (let i = 0; i < 4; i++) plan.push({delayed: true});
const boundedBefore = calls.length;
const bounded = Array.from({length: 4}, (_, i) => handler(event, ctx("budget", "run" + i)));
check("fifth concurrent worker denied", await handler(event, ctx("budget", "run4")) === undefined && calls.length === boundedBefore + 4);
for (const child of children.slice(-4)) child.deliver();
check("four admitted workers succeed", (await Promise.all(bounded)).every((value) => value?.prependContext === "synthetic recall"));
check("capacity restored after close", (await handler(event, ctx("budget", "run4")))?.prependContext === "synthetic recall");
globalThis.setTimeout = nativeSetTimeout;
globalThis.clearTimeout = nativeClearTimeout;
process.stdout.write(JSON.stringify({checks: checks.length, passed: checks}));
'''
    script = 'import { EventEmitter } from "node:events";\n' + source + harness
    result = subprocess.run([bun, "--eval", script], capture_output=True, text=True, timeout=30, check=True)
    report = json.loads(result.stdout)
    if report["checks"] < 53:
        raise AssertionError(report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--js-controls", action="store_true")
    parser.add_argument("--bun")
    args, remaining = parser.parse_known_args()
    if args.js_controls:
        bun = args.bun or shutil.which("bun")
        if not bun:
            parser.error("--js-controls requires --bun PATH or Bun on PATH")
        js_controls(bun)
    else:
        unittest.main(argv=[sys.argv[0], *remaining])
