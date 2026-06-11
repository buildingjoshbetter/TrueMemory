"""
Build the LONG_HORIZON_SYNTHETIC evaluation dataset.

Phase 7.2 of the MEMORIST research session. This is the instrument we
build because no public benchmark scores selective ingestion across
realistic multi-week Claude-Code-style sessions (per TrueMemory paper §6.4
/ §8 — "no public benchmark scores a system that chooses what to ingest").

Design constraints from MEMORIST_SPEC.md:
- ~3,000–5,000 messages across 30–50 simulated sessions
- Spanning 4 weeks of wall-clock time
- Mix proportions:
    40% task/code/tool-call discussion
    20% substantive user context (preferences, decisions, personal facts)
    30% pleasantries / acknowledgments / short responses
    10% off-topic chit-chat
- Ground-truth annotated — each session has `should_remember` + `should_forget`
- Retrieval Qs at varied time gaps — 50+ queries each tagged with
  "asked after session X" / "gold answer should come from session Y"
- Adversarial probes: contradictions, updates, near-duplicates,
  emotional intensity, repetition / rehearsal, noise

Architecture:
1. **Personas are defined in code** — they are the ground truth, NOT
   LLM-generated. Adversarial probes (contradictions, updates) are
   planted deterministically. The LLM only generates the *narrative
   surface* of each session, not the underlying facts.
2. **Sessions are conditioned on** (persona + topic + previously-planted
   facts that should appear) and the LLM is instructed to weave the
   planted content into a believable Claude-Code dialogue.
3. **Retrieval queries are constructed deterministically** from the
   planted facts — the gold answer is whatever was planted, not whatever
   the LLM happened to generate.

Cost budget: ≤$5 of total $50 cap. With Anthropic Haiku 4.5 at
~$0.001 per ~1k input tokens, generating 50 sessions × ~2k tokens each
= 100k tokens → ~$0.10. Very comfortable headroom.

Output: `benchmarks/gate_eval/datasets/long_horizon_synthetic.json`
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "benchmarks" / "gate_eval" / "datasets" / "long_horizon_synthetic.json"

# ---------------------------------------------------------------------------
# Persona definitions — the ground truth
# ---------------------------------------------------------------------------

@dataclass
class PlantedFact:
    """A fact that's been told to the system and is gold-truth queryable."""
    fact: str
    session_id: int                 # Which session it was first attested in
    category: str                   # personal | preference | decision | technical | temporal | emotional | trivial
    should_remember: bool = True    # Whether a good memory system should retain this
    superseded_by: int | None = None  # Session_id where a contradicting / updating fact was attested
    rehearsals: list[int] = field(default_factory=list)  # Sessions that re-mention it
    emotional_intensity: float = 0.0  # 0.0 = neutral, 1.0 = high


@dataclass
class Persona:
    name: str
    role: str
    location: str
    project: str
    stack: str
    preferences: list[str]
    family: str
    allergies: str
    recent_events: list[str]
    pets: str = ""


@dataclass
class Session:
    session_id: int
    persona_name: str
    day_offset: int                # 0..27 (4 weeks)
    topic: str
    mix_goal: dict                 # category → message count
    planted_facts: list[int]       # indices into the persona's planted_fact_pool
    transcript: list[dict] = field(default_factory=list)
    n_messages: int = 0


@dataclass
class RetrievalQuery:
    query_id: int
    asked_after_session: int
    gold_from_session: int
    time_gap_days: int
    gold_answer: str
    question: str
    probe_type: str  # baseline | contradiction | update | near_duplicate | emotional | rehearsal | noise


# Hand-authored persona pool. Adversarial-probe-friendly: each persona has
# at least one fact that gets contradicted (planted superseded), one that
# gets reinforced via repetition (rehearsals), and one with emotional weight.

PERSONA_TEMPLATES: list[Persona] = [
    Persona(
        name="Alex Park",
        role="Senior Rust developer",
        location="Boston",
        project="forge (CLI)",
        stack="SvelteKit + Postgres",
        preferences=["uv over pip", "vim", "espresso, no sugar"],
        family="married, daughter Emma age 7",
        allergies="peanuts (anaphylactic)",
        recent_events=["accepted Meta offer in week 2", "moved from Boston to NYC in week 3"],
        pets="cat named Ada",
    ),
    Persona(
        name="Mei Chen",
        role="ML engineer",
        location="Seattle",
        project="vibecheck (sentiment)",
        stack="PyTorch + FastAPI + Modal",
        preferences=["zsh", "fish over bash", "decaf only"],
        family="single, dog Mochi (corgi)",
        allergies="lactose intolerant",
        recent_events=["promoted to senior in week 1", "diagnosed with adult ADHD week 3"],
    ),
    Persona(
        name="Diego Ramos",
        role="DevOps lead",
        location="Madrid",
        project="meridian (k8s platform)",
        stack="Go + Kubernetes + Terraform",
        preferences=["always emacs", "yerba mate", "prefers PRs over Slack discussion"],
        family="married, twins age 4",
        allergies="shellfish",
        recent_events=["moved team to GitLab from GitHub in week 1", "actually switched back to GitHub week 4 (contradiction)"],
    ),
    Persona(
        name="Priya Iyer",
        role="Product manager",
        location="Bangalore",
        project="tessera (mobile app)",
        stack="React Native + Firebase",
        preferences=["Notion not Linear", "bullet journal", "no caffeine after 2pm"],
        family="married, no kids, planning",
        allergies="none",
        recent_events=["got pregnant in week 2 (high-emotion fact)", "miscarried in week 3 (devastating)"],
    ),
    Persona(
        name="Sam Beck",
        role="Solo indie developer",
        location="Austin",
        project="quill (writing app)",
        stack="Tauri + Rust + SQLite",
        preferences=["sublime over vscode", "loose-leaf tea", "writes at 5am"],
        family="single, no pets",
        allergies="gluten (celiac)",
        recent_events=["launched on Product Hunt week 2 (success)", "got 1k paying users week 3 (milestone)"],
    ),
    Persona(
        name="Marcus Webb",
        role="Staff engineer at startup",
        location="San Francisco",
        project="pilot (compliance SaaS)",
        stack="TypeScript + Postgres + Temporal",
        preferences=["pnpm over npm", "Linear for tracking", "espresso macchiato"],
        family="divorced, son Ben age 12 every other weekend",
        allergies="none",
        recent_events=["Series B closed week 2 ($50M)", "lost a key engineer to Stripe week 3"],
    ),
]


# ---------------------------------------------------------------------------
# Adversarial probe planning
# ---------------------------------------------------------------------------

def plan_planted_facts(persona: Persona, n_sessions: int) -> list[PlantedFact]:
    """Plan which facts get planted in which sessions, including adversarial probes.

    Each persona contributes ~10-15 planted facts spanning all probe types.
    Probe types:
      - baseline: simple fact mentioned once, queried later
      - contradiction: fact A in session N, fact ¬A in session M (M>N)
      - update: fact A in session N, evolved fact A' in session M (M>N)
      - near_duplicate: same fact phrased differently in 2+ sessions (should not bloat store)
      - emotional: high-emotion fact (recent_events typically)
      - rehearsal: fact mentioned in 3+ sessions (should be very memorable)
      - noise: trivial chit-chat (should NOT be remembered)
    """
    facts: list[PlantedFact] = []

    # Baseline facts (mentioned once, ground truth)
    facts.append(PlantedFact(
        fact=f"works as a {persona.role}", session_id=1, category="personal",
    ))
    facts.append(PlantedFact(
        fact=f"is working on a project called {persona.project}", session_id=1, category="technical",
    ))
    facts.append(PlantedFact(
        fact=f"uses {persona.stack} as the stack for {persona.project}", session_id=2, category="technical",
    ))
    facts.append(PlantedFact(
        fact=f"family: {persona.family}", session_id=2, category="personal",
    ))
    facts.append(PlantedFact(
        fact=f"allergies: {persona.allergies}", session_id=3, category="personal",
        emotional_intensity=0.6,  # allergies feel important
    ))
    if persona.pets:
        facts.append(PlantedFact(
            fact=f"has {persona.pets}", session_id=2, category="personal",
        ))

    # Preferences (should be remembered, low emotion)
    for i, pref in enumerate(persona.preferences):
        facts.append(PlantedFact(
            fact=f"prefers {pref}", session_id=2 + i, category="preference",
        ))

    # Rehearsal: original location mentioned 3 times
    base_loc_session = 1
    facts.append(PlantedFact(
        fact=f"lives in {persona.location}",
        session_id=base_loc_session,
        category="personal",
        rehearsals=[base_loc_session + 2, base_loc_session + 4],
    ))

    # Plant the persona's recent_events as emotional + temporally-significant facts.
    # These are the highest-information events of the persona's life during this 4-week window.
    for ev_idx, event in enumerate(persona.recent_events):
        # Emotionally intense events scattered across sessions ~5-15
        ev_session = 5 + ev_idx * 4
        if ev_session > n_sessions:
            ev_session = n_sessions - 1
        emo_intensity = 0.8 if any(w in event.lower() for w in [
            "fired", "diagnosed", "promoted", "accepted", "miscarried",
            "pregnant", "lost", "launched", "closed", "anaphylactic",
        ]) else 0.5
        facts.append(PlantedFact(
            fact=event, session_id=ev_session, category="emotional",
            emotional_intensity=emo_intensity,
        ))

    # Contradiction probe: location moves
    if "moved" in " ".join(persona.recent_events).lower():
        # Find the move event's session and mark prior location as superseded
        for f in facts:
            if f.fact.startswith("lives in ") and "moved" in " ".join(persona.recent_events).lower():
                # Find the move session
                move_sess = None
                for f2 in facts:
                    if "moved" in f2.fact.lower():
                        move_sess = f2.session_id
                        break
                if move_sess is not None:
                    f.superseded_by = move_sess

    # Near-duplicate probe: re-attest some facts with paraphrased wording
    # (handled in session generation by the LLM — see below)

    # Trivial / noise facts (should NOT be remembered)
    noise_facts = [
        ("debating whether to get coffee or tea this morning", 4, 0.0),
        ("complaining about the weather", 6, 0.0),
        ("commenting on a meme they saw", 8, 0.0),
        ("idle small talk about a TV show", 11, 0.0),
        ("griping about a meeting that ran long", 13, 0.0),
    ]
    for fact_str, sess, _emo in noise_facts:
        facts.append(PlantedFact(
            fact=fact_str, session_id=sess, category="trivial",
            should_remember=False,
        ))

    return facts


# ---------------------------------------------------------------------------
# Session topic templates — drives the LLM's narrative surface
# ---------------------------------------------------------------------------

SESSION_TOPICS = [
    # Each session topic gets one of these high-level frames; the LLM weaves the planted facts in
    "debugging a CI failure in {project}",
    "designing a new feature for {project}",
    "code review of a PR adding tests to {project}",
    "weekly standup recap with the {project} team",
    "evaluating a new {stack}-related dependency",
    "deciding architecture for {project}'s next milestone",
    "writing documentation for {project}",
    "interviewing a candidate to join the {project} team",
    "1:1 with a teammate about prioritization",
    "post-mortem on a production incident",
    "performance optimization session for {project}",
    "migration planning to a new tool / stack",
    "exploring a competitor's product for ideas",
    "writing a launch post for {project}",
    "sprint retrospective and planning the next two weeks",
]


# ---------------------------------------------------------------------------
# Anthropic API caller
# ---------------------------------------------------------------------------

def call_anthropic(system: str, user: str, model: str = "claude-haiku-4-5",
                    max_tokens: int = 1500, temperature: float = 0.7) -> str:
    """Call Claude via the local `claude` CLI (OAuth, no API-key wrangling).

    The session env had a stale `ANTHROPIC_API_KEY` that returned 401 on both
    direct API calls and CLI invocations that picked it up. Mirror truememory's
    own `_complete_claude_cli` pattern (truememory/ingest/models.py) and unset
    the env var so the CLI uses OAuth keychain auth instead.

    `temperature` is currently ignored — the claude CLI doesn't expose it. We
    rely on Claude's default temperature for the persona-narrative generation.
    """
    import subprocess
    full_prompt = f"{system}\n\n{user}"
    cmd = ["claude", "-p", "--output-format", "json", "--model", model]
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    proc = subprocess.run(
        cmd, input=full_prompt, capture_output=True, text=True, timeout=180, env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"claude CLI exit {proc.returncode}: {proc.stderr.strip()[:300]}")
    data = json.loads(proc.stdout)
    if data.get("is_error"):
        raise RuntimeError(f"claude CLI error: {data.get('result', 'unknown')}")
    result = data.get("result")
    if not isinstance(result, str):
        raise RuntimeError(f"claude CLI missing result string: {data}")
    return result


# ---------------------------------------------------------------------------
# Session generator
# ---------------------------------------------------------------------------

SESSION_SYSTEM_PROMPT = """\
You are simulating a Claude Code session — a back-and-forth dialogue between a USER \
(a software developer) and an ASSISTANT. Generate a realistic transcript that reflects \
the persona, the session topic, and the planted facts the user must mention naturally.

Output JSON only — a list of message objects with shape:
{"role": "user" | "assistant", "content": "<text>"}

Mix the message types per the mix_goal:
- substantive: messages that convey lasting facts (preferences, decisions, personal info, technical context)
- task: code/debugging/tool-call back-and-forth specific to the session topic
- pleasantry: short acknowledgments, greetings, "thanks", "got it"
- offtopic: brief tangents about weather, food, weekend plans

Guidance:
- Plant the listed facts NATURALLY in the user's messages — don't just dump them. \
A good ingestion gate should be able to extract them.
- Mix in some near-duplicates of facts already attested in earlier sessions \
(paraphrased — same fact, different wording). The user might re-mention their stack \
in passing, or refer to their pet by name.
- Pleasantries and offtopic should be SHORT (under 15 words).
- Substantive messages should be MEDIUM-LONG (30-150 words).
- Task messages range widely.
- 8-25 total messages per session.
- DO NOT output any prose outside the JSON array. Output starts with [ and ends with ].
"""


def render_user_prompt(persona: Persona, session: Session, planted_fact_objs: list[PlantedFact]) -> str:
    """Build the per-session user prompt for the generator."""
    fact_lines = []
    for f in planted_fact_objs:
        marker = " [HIGH EMOTION]" if f.emotional_intensity >= 0.7 else ""
        fact_lines.append(f"  - {f.fact}{marker}")

    persona_block = (
        f"Persona: {persona.name}\n"
        f"  role: {persona.role}\n"
        f"  location: {persona.location}\n"
        f"  project: {persona.project} (stack: {persona.stack})\n"
        f"  family: {persona.family}\n"
        f"  preferences: {', '.join(persona.preferences)}\n"
        f"  recent events: {', '.join(persona.recent_events) if persona.recent_events else 'none'}\n"
        f"  pets: {persona.pets or 'none'}\n"
        f"  allergies: {persona.allergies}\n"
    )

    mix_block = (
        f"Mix goal (target message counts):\n"
        f"  substantive: {session.mix_goal.get('substantive', 0)}\n"
        f"  task: {session.mix_goal.get('task', 0)}\n"
        f"  pleasantry: {session.mix_goal.get('pleasantry', 0)}\n"
        f"  offtopic: {session.mix_goal.get('offtopic', 0)}\n"
    )

    facts_block = "Plant these facts naturally in the USER's messages:\n" + (
        "\n".join(fact_lines) if fact_lines else "  (none — surface the persona via the topic only)"
    )

    return (
        f"{persona_block}\n"
        f"Session topic: {session.topic}\n"
        f"Day offset (week 0-3): day {session.day_offset}\n\n"
        f"{mix_block}\n"
        f"{facts_block}\n\n"
        f"Now generate the transcript as a JSON list of message objects."
    )


def generate_session(persona: Persona, session: Session, planted_fact_objs: list[PlantedFact]) -> list[dict]:
    """Generate one session's transcript via Anthropic API. Returns a list of message dicts."""
    user_prompt = render_user_prompt(persona, session, planted_fact_objs)
    raw = call_anthropic(SESSION_SYSTEM_PROMPT, user_prompt)

    # Parse JSON list from the response. Use the same balanced-bracket
    # walker the truememory extractor uses.
    raw = raw.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()

    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise RuntimeError(f"Could not find JSON array in response: {raw[:200]!r}")

    parsed = json.loads(raw[start:end + 1])
    msgs: list[dict] = []
    for m in parsed:
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and m.get("content"):
            msgs.append({"role": m["role"], "content": str(m["content"]).strip()})
    return msgs


# ---------------------------------------------------------------------------
# Retrieval-query construction (deterministic from planted facts)
# ---------------------------------------------------------------------------

QUERY_TEMPLATES = {
    "personal":   ["What's {persona}'s {attr}?", "Where does {persona} live?", "Who is in {persona}'s family?"],
    "preference": ["What does {persona} prefer for {topic}?", "What are {persona}'s tool preferences?"],
    "technical":  ["What stack is {persona} using for {project}?", "What is {persona} working on?"],
    "emotional":  ["Did {persona} have any major life events recently?", "Tell me about {persona}'s recent news"],
    "trivial":    ["Did {persona} say anything about {topic}?"],  # negative probe — should mostly fail
}


def build_retrieval_queries(personas: list[Persona], all_facts: dict[str, list[PlantedFact]],
                             n_sessions_per_persona: int) -> list[RetrievalQuery]:
    """Construct ~50+ retrieval queries with varied time gaps and probe types."""
    queries: list[RetrievalQuery] = []
    qid = 0

    for persona in personas:
        facts = all_facts[persona.name]
        for f in facts:
            if not f.should_remember:
                # Negative probe — query should ideally return nothing useful
                # (or abstention). We still include some to test gate over-eagerness.
                if qid % 5 == 0:  # only ~20% of trivial facts get probed
                    qid += 1
                    queries.append(RetrievalQuery(
                        query_id=qid,
                        asked_after_session=min(f.session_id + 5, n_sessions_per_persona),
                        gold_from_session=f.session_id,
                        time_gap_days=5,  # ~5 days later
                        gold_answer="N/A (trivial fact — should NOT surface)",
                        question=f"Did {persona.name} say anything about {f.fact[:30]}?",
                        probe_type="noise",
                    ))
                continue

            # Choose a time gap: vary across {0, 3, 7, 14, 21, 28+}
            for gap_days in (3, 14, 21):
                ask_session = f.session_id + gap_days // 2  # rough day-to-session mapping (~2 days/session)
                if ask_session > n_sessions_per_persona:
                    continue

                qid += 1

                if f.superseded_by and ask_session > f.superseded_by:
                    # Contradiction probe — gold answer should be the SUPERSEDED fact
                    # (the most recent one), not the original
                    superseding = next((x for x in facts if x.session_id == f.superseded_by), None)
                    gold = superseding.fact if superseding else f.fact
                    probe = "contradiction"
                else:
                    gold = f.fact
                    probe = "rehearsal" if f.rehearsals else (
                        "emotional" if f.emotional_intensity >= 0.5 else "baseline"
                    )

                queries.append(RetrievalQuery(
                    query_id=qid,
                    asked_after_session=ask_session,
                    gold_from_session=f.session_id,
                    time_gap_days=gap_days,
                    gold_answer=gold,
                    question=f"What do you remember about {persona.name}'s {f.category}? Specifically: {f.fact[:60]}",
                    probe_type=probe,
                ))

    return queries


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--personas", type=int, default=6, help="Number of personas (max=len(PERSONA_TEMPLATES))")
    parser.add_argument("--sessions-per-persona", type=int, default=8, help="Sessions per persona over 4 weeks")
    parser.add_argument("--smoke-test", action="store_true", help="Generate 1 persona × 2 sessions only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip API calls — generate the structure with placeholder transcripts. Useful for offline-validating the schema.")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    if args.smoke_test:
        args.personas = 1
        args.sessions_per_persona = 2

    rng = random.Random(args.seed)
    personas = PERSONA_TEMPLATES[: args.personas]
    n_sessions_per_persona = args.sessions_per_persona

    # ---- Plan all planted facts upfront ----
    all_facts: dict[str, list[PlantedFact]] = {}
    for p in personas:
        all_facts[p.name] = plan_planted_facts(p, n_sessions_per_persona)

    # ---- Build session schedules ----
    sessions: list[Session] = []
    sid = 0
    for p in personas:
        topics = rng.sample(SESSION_TOPICS, min(n_sessions_per_persona, len(SESSION_TOPICS)))
        for sess_idx in range(n_sessions_per_persona):
            sid += 1
            day_offset = int(28 * sess_idx / max(n_sessions_per_persona - 1, 1))
            topic = topics[sess_idx % len(topics)].format(project=p.project, stack=p.stack)

            # Find the planted facts whose session_id matches this session
            session_facts = [
                i for i, f in enumerate(all_facts[p.name])
                if f.session_id == sess_idx + 1
            ]

            mix_goal = {
                "substantive": max(1, len(session_facts) + 2),  # at least cover planted facts
                "task": rng.randint(4, 8),
                "pleasantry": rng.randint(3, 6),
                "offtopic": rng.randint(0, 2),
            }
            sessions.append(Session(
                session_id=sid,
                persona_name=p.name,
                day_offset=day_offset,
                topic=topic,
                mix_goal=mix_goal,
                planted_facts=session_facts,
            ))

    # ---- Generate transcripts ----
    print(f"Generating {len(sessions)} sessions across {len(personas)} personas...", file=sys.stderr)
    total_messages = 0
    for sess in sessions:
        persona = next(p for p in personas if p.name == sess.persona_name)
        planted_objs = [all_facts[persona.name][i] for i in sess.planted_facts]

        if args.no_llm:
            # Placeholder: emit a fake transcript that contains the planted facts verbatim
            sess.transcript = []
            for pf in planted_objs:
                sess.transcript.append({"role": "user", "content": f"Just so you know — {pf.fact}."})
                sess.transcript.append({"role": "assistant", "content": "Got it."})
        else:
            try:
                sess.transcript = generate_session(persona, sess, planted_objs)
            except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError, json.JSONDecodeError) as e:
                print(f"  Session {sess.session_id} failed: {e}", file=sys.stderr)
                # Fallback to placeholder transcript so the pipeline still produces a dataset
                sess.transcript = [
                    {"role": "user", "content": f"[GENERATION FAILED] {pf.fact}"} for pf in planted_objs
                ]

        sess.n_messages = len(sess.transcript)
        total_messages += sess.n_messages
        print(f"  session {sess.session_id} ({persona.name}, day {sess.day_offset}): {sess.n_messages} msgs", file=sys.stderr)
        time.sleep(0.5)  # gentle rate limit

    # ---- Build retrieval queries ----
    queries = build_retrieval_queries(personas, all_facts, n_sessions_per_persona)
    print(f"Built {len(queries)} retrieval queries", file=sys.stderr)

    # ---- Serialize ----
    payload = {
        "meta": {
            "seed": args.seed,
            "n_personas": len(personas),
            "n_sessions_per_persona": n_sessions_per_persona,
            "n_total_sessions": len(sessions),
            "n_total_messages": total_messages,
            "n_queries": len(queries),
            "wall_clock_days": 28,
            "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "phase": "memorist_phase_7_2",
            "smoke_test": args.smoke_test,
            "no_llm": args.no_llm,
        },
        "personas": [asdict(p) for p in personas],
        "planted_facts": {name: [asdict(f) for f in facts] for name, facts in all_facts.items()},
        "sessions": [asdict(s) for s in sessions],
        "retrieval_queries": [asdict(q) for q in queries],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {args.out}", file=sys.stderr)
    print(f"  total messages: {total_messages}", file=sys.stderr)
    print(f"  total queries: {len(queries)}", file=sys.stderr)


if __name__ == "__main__":
    main()
