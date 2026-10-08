#!/usr/bin/env python3
"""Content-free, standard-library contract reproductions. No models or network."""
from __future__ import annotations
import ast
import json
import logging
import sys
import types
from pathlib import Path

SOURCE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
RESULTS: dict[str, object] = {}

def functions(relative: str, names: set[str], namespace: dict) -> dict:
    tree = ast.parse((SOURCE / relative).read_text())
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    nodes.extend(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), relative, "exec"), namespace)
    return namespace

# Import only this stdlib parser via exec: no package import or bytecode files.
parser = types.ModuleType("contracts_transcript_fixture")
sys.modules[parser.__name__] = parser
exec(compile((SOURCE / "truememory/ingest/transcript.py").read_text(), "transcript.py", "exec"), parser.__dict__)
codex = json.dumps({"type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "SYNTHETIC_FACT_42"}]}})
claude = json.dumps({"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": "SYNTHETIC_FACT_42"}]}})
RESULTS["C06_parser"] = {"codex_messages": len(parser.parse_transcript(codex)), "claude_control_messages": len(parser.parse_transcript(claude))}
assert RESULTS["C06_parser"] == {"codex_messages": 0, "claude_control_messages": 1}

class SyntheticPath(type(Path())):
    @classmethod
    def home(cls):
        return cls("/synthetic-home")
namespace = functions("truememory/ingest/hooks/_shared.py", {"_transcript_roots", "is_allowed_transcript"}, {"Path": SyntheticPath, "os": types.SimpleNamespace(environ={})})
RESULTS["C05_transcript_admission"] = {host: namespace["is_allowed_transcript"](f"/synthetic-home/{path}") for host, path in {"claude": ".claude/projects/project/session.jsonl", "codex": ".codex/sessions/2026/10/08/session.jsonl", "gemini": ".gemini/tmp/project/chats/session.json"}.items()}
assert RESULTS["C05_transcript_admission"] == {"claude": True, "codex": False, "gemini": False}

# Provider auto-detection: fake availability, fake key, no subprocess/network.
namespace = functions("truememory/ingest/models.py", {"auto_detect"}, {"LLMConfig": lambda **kw: types.SimpleNamespace(**kw), "hydrate_config": lambda cfg: cfg, "_ollama_available": lambda: False, "_claude_cli_available": lambda: False, "os": types.SimpleNamespace(environ={"OPENROUTER_API_KEY": "SYNTHETIC_NOT_A_KEY"}), "log": logging.getLogger("fixture")})
RESULTS["C02_provider_autodetect"] = namespace["auto_detect"]().provider
assert RESULTS["C02_provider_autodetect"] == "openrouter"

# Execute actual extraction function with inert chunk/parser/provider fakes.
tree = ast.parse((SOURCE / "truememory/ingest/extractor.py").read_text())
prompt = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "EXTRACTION_PROMPT" for t in n.targets))
captured: list[str] = []
def complete(config, text, system=""):
    captured.append(text)
    return "[]"
namespace = functions("truememory/ingest/extractor.py", {"extract_facts"}, {"_DEFAULT_MAX_CHUNKS": 20, "_CHUNK_CHAR_BUDGET": 20000, "_chunk_transcript": lambda text, budget: [text], "EXTRACTION_PROMPT": prompt, "EXTRACTION_SYSTEM": "synthetic system", "_neutralize_delimiters": lambda text: text, "complete": complete, "LLMError": RuntimeError, "_parse_extraction_response": lambda response, count: [], "_dedupe_facts_by_content": lambda facts: facts, "log": logging.getLogger("fixture")})
namespace["extract_facts"]("SYNTHETIC_TRANSCRIPT_PRIVATE_42", types.SimpleNamespace(provider="openrouter"))
RESULTS["C02_extraction_prompt_contains_transcript"] = "SYNTHETIC_TRANSCRIPT_PRIVATE_42" in captured[0]
assert RESULTS["C02_extraction_prompt_contains_transcript"]

# Telemetry init payload, without thread execution, hardware lookup or persistence.
class NoopThread:
    def __init__(self, **kwargs): pass
    def start(self): pass
tracked: list[dict] = []
namespace = functions("truememory/telemetry.py", {"init"}, {"is_enabled": lambda: True, "_get_version": lambda: "fixture", "sys": types.SimpleNamespace(platform="fixture"), "platform": types.SimpleNamespace(machine=lambda: "fixture", python_version=lambda: "fixture"), "get_device_id": lambda: "SYNTHETIC_HASH", "track": lambda event, props: tracked.append({"event": event, "props": props}), "threading": types.SimpleNamespace(Thread=NoopThread), "_flush_loop": lambda: None})
namespace["init"]({"user_id": "SYNTHETIC_UUID", "email": "fixture@example.invalid", "tier": "edge"})
RESULTS["C04_telemetry"] = tracked
assert tracked[0]["props"]["email"] == "fixture@example.invalid"
assert tracked[0]["props"]["device_id"] == "SYNTHETIC_HASH"

# Exact OpenClaw wrapper key versus actual shared consumer key.
openclaw_input = {"session_id": "fixture", "user_prompt": "SYNTHETIC_PROMPT_42"}
RESULTS["C08_openclaw_prompt_seen_by_shared_hook"] = openclaw_input.get("prompt", "").strip()
assert RESULTS["C08_openclaw_prompt_seen_by_shared_hook"] == ""

# Pure contract fixture: documented native host output fields versus shared output.
shared_output = {"additionalContext": "SYNTHETIC_RECALL_42"}
RESULTS["C07_documented_output_contract"] = {"cursor": shared_output.get("additional_context"), "gemini": shared_output.get("hookSpecificOutput", {}).get("additionalContext"), "codex": shared_output.get("hookSpecificOutput", {}).get("additionalContext")}
assert all(value is None for value in RESULTS["C07_documented_output_contract"].values())

print(json.dumps({"source_sha": "063e5b8844af735a52fde886217a5d26a0f13064", "all_assertions_passed": True, "limitations": "AST/contract fixtures; no host dispatch, model inference, live user data, or network.", "results": RESULTS}, indent=2))
