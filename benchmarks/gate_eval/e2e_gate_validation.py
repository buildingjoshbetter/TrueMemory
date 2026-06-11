#!/usr/bin/env python3
"""
End-to-end gate validation with a realistic conversation.

Feeds a hand-crafted conversation through the FULL ingestion pipeline
(LLM extractor → encoding gate → store) and verifies the gate makes
sensible decisions. This tests the actual production path, not the
benchmark harness.

If no LLM API key is available, falls back to the regex extractor.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# A realistic conversation with clear signal and noise
CONVERSATION = """User: hey what's up
Assistant: Not much, how are you?
User: good good. so I have some news
Assistant: Oh yeah? What's going on?
User: I got the job at Anthropic!!! Starting as a senior engineer in two weeks
Assistant: That's amazing, congratulations!
User: thanks!! yeah I'm so excited. the salary is $285k base plus equity
Assistant: Wow that's great. Are you going to relocate?
User: yeah we're moving to San Francisco from Portland. already found an apartment in the Mission district
Assistant: Nice! How's Sarah feeling about the move?
User: she's actually really excited. she got a remote position at Stripe so she can work from anywhere
Assistant: That's perfect timing
User: totally. oh and we're engaged!! I proposed last weekend at Multnomah Falls
Assistant: Oh my god congratulations!! When's the wedding?
User: we're thinking June next year. nothing booked yet though
Assistant: That's so exciting. How's the apartment search going?
User: found a 2br on Valencia Street. $4,200 a month which is insane but whatever lol
Assistant: Yeah SF prices are wild
User: haha yeah. anyway how's your week been?
Assistant: Pretty good, just working on some projects
User: cool cool. ok I gotta run, talk later!
Assistant: Congrats again on everything! Talk soon."""


# What the gate SHOULD store (signal):
EXPECTED_SIGNAL = [
    "got the job at Anthropic",
    "senior engineer",
    "$285k",
    "moving to San Francisco from Portland",
    "Mission district",
    "Sarah",
    "remote position at Stripe",
    "engaged",
    "proposed",
    "Multnomah Falls",
    "June next year",
    "Valencia Street",
    "$4,200",
]

# What the gate SHOULD reject (noise):
EXPECTED_NOISE = [
    "hey what's up",
    "good good",
    "thanks",
    "totally",
    "haha yeah",
    "cool cool",
    "ok I gotta run",
]


def main():
    print("End-to-End Gate Validation")
    print("=" * 70)

    # Write conversation to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(CONVERSATION)
        transcript_path = f.name

    try:
        # Try full pipeline with LLM extractor
        try:
            from truememory.ingest.pipeline import IngestionPipeline
            from truememory.memory import Memory

            # Use in-memory database
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "test.db"
                memory = Memory(str(db_path))

                pipeline = IngestionPipeline(
                    memory=memory,
                    gate_threshold=0.26,
                    gate_enabled=True,
                )

                print("\nRunning full pipeline (extractor → gate → store)...")
                result = pipeline.ingest_transcript(transcript_path, session_id="validation")

                print(f"\n{'='*70}")
                print("PIPELINE RESULTS")
                print(f"{'='*70}")

                if hasattr(result, 'trace') and result.trace:
                    for entry in result.trace:
                        fact = entry.get("fact", {})
                        gate = entry.get("gate", {})
                        content = fact.get("content", "")[:60]
                        category = fact.get("category", "?")
                        passed = gate.get("passed", False)
                        score = gate.get("score", 0)
                        verdict = "ENCODE" if passed else "SKIP"
                        print(f"  [{verdict}] {score:.2f} ({category}) {content}")
                elif isinstance(result, dict):
                    stored = result.get("stored", [])
                    skipped = result.get("skipped", [])
                    print(f"  Stored: {len(stored)} facts")
                    print(f"  Skipped: {len(skipped)} facts")
                    for fact in stored:
                        content = fact if isinstance(fact, str) else fact.get("content", "")
                        print(f"  [ENCODE] {content[:70]}")
                    for fact in skipped:
                        content = fact if isinstance(fact, str) else fact.get("content", "")
                        print(f"  [SKIP]   {content[:70]}")

                # Check what's actually stored
                print(f"\n{'='*70}")
                print("STORED MEMORIES")
                print(f"{'='*70}")
                try:
                    results = memory.search("Anthropic job salary engaged", limit=20)
                    for r in results:
                        print(f"  {r.get('content', '')[:80]}")
                    print(f"\n  Total stored: {len(results)}")
                except Exception as e:
                    print(f"  Search failed: {e}")

                memory.close()

        except Exception as e:
            print(f"\nFull pipeline not available: {e}")
            print("Running gate-only validation on raw messages...\n")
            _run_gate_raw_messages()

    finally:
        Path(transcript_path).unlink()


def _run_gate_only(transcript_path):
    """Test just the gate with the regex extractor's output."""
    from truememory.ingest.encoding_gate import EncodingGate
    from truememory.ingest.extractor import extract_facts_regex

    with open(transcript_path) as f:
        transcript = f.read()

    # Use regex extractor (no LLM needed)
    facts = extract_facts_regex(transcript)
    print(f"Regex extractor found {len(facts)} facts:\n")

    if not facts:
        print("  No facts extracted by regex — testing gate on raw messages instead\n")
        _run_gate_raw_messages()
        return

    class SimpleMemory:
        def __init__(self):
            self.items = []
        def search(self, query, **kw):
            return [{"content": c, "score": 0.3} for c in self.items[-5:]]
        def search_vectors(self, query, **kw):
            return self.search(query)

    memory = SimpleMemory()
    gate = EncodingGate(memory=memory, threshold=0.26)

    encoded = []
    skipped = []

    for fact in facts:
        decision = gate.evaluate(fact.content, fact.category)
        entry = {
            "content": fact.content,
            "category": fact.category,
            "score": decision.encoding_score,
            "novelty": decision.novelty,
            "salience": decision.salience,
            "pe": decision.prediction_error,
            "verdict": "ENCODE" if decision.should_encode else "SKIP",
        }

        if decision.should_encode:
            encoded.append(entry)
            memory.items.append(fact.content)
        else:
            skipped.append(entry)

        print(f"  [{entry['verdict']}] {entry['score']:.2f} "
              f"n={entry['novelty']:.2f} s={entry['salience']:.2f} pe={entry['pe']:.2f} "
              f"({entry['category']}) {entry['content'][:55]}")

    print(f"\n{'='*70}")
    print(f"SUMMARY: {len(encoded)} encoded, {len(skipped)} skipped "
          f"({len(encoded)/(len(encoded)+len(skipped))*100:.0f}% encode rate)")

    # Check signal coverage
    print(f"\n{'='*70}")
    print("SIGNAL COVERAGE CHECK")
    print(f"{'='*70}")
    stored_text = " ".join(e["content"].lower() for e in encoded)
    for signal in EXPECTED_SIGNAL:
        found = signal.lower() in stored_text
        marker = "✓" if found else "✗"
        print(f"  {marker} {signal}")

    hit = sum(1 for s in EXPECTED_SIGNAL if s.lower() in stored_text)
    print(f"\n  Coverage: {hit}/{len(EXPECTED_SIGNAL)} ({hit/len(EXPECTED_SIGNAL)*100:.0f}%)")


def _run_gate_raw_messages():
    """Test gate directly on raw conversation messages."""
    from truememory.ingest.encoding_gate import EncodingGate

    messages = [
        ("hey what's up", ""),
        ("good good. so I have some news", ""),
        ("I got the job at Anthropic!!! Starting as a senior engineer in two weeks", "personal"),
        ("thanks!! yeah I'm so excited. the salary is $285k base plus equity", "personal"),
        ("yeah we're moving to San Francisco from Portland. already found an apartment in the Mission district", "personal"),
        ("she's actually really excited. she got a remote position at Stripe so she can work from anywhere", "personal"),
        ("totally. oh and we're engaged!! I proposed last weekend at Multnomah Falls", "relationship"),
        ("we're thinking June next year. nothing booked yet though", "temporal"),
        ("found a 2br on Valencia Street. $4,200 a month which is insane but whatever lol", "personal"),
        ("haha yeah. anyway how's your week been?", ""),
        ("cool cool. ok I gotta run, talk later!", ""),
    ]

    class SimpleMemory:
        def __init__(self):
            self.items = []
        def search(self, query, **kw):
            return [{"content": c, "score": 0.3} for c in self.items[-5:]]
        def search_vectors(self, query, **kw):
            return self.search(query)

    memory = SimpleMemory()
    gate = EncodingGate(memory=memory, threshold=0.26)

    encoded = []
    skipped = []

    print(f"{'Verdict':<8} {'Score':>5} {'N':>5} {'S':>5} {'PE':>5}  {'Cat':<12} Message")
    print("-" * 100)

    for content, category in messages:
        decision = gate.evaluate(content, category)
        verdict = "ENCODE" if decision.should_encode else "SKIP"

        if decision.should_encode:
            encoded.append(content)
            memory.items.append(content)
        else:
            skipped.append(content)

        print(f"{verdict:<8} {decision.encoding_score:>5.2f} {decision.novelty:>5.2f} "
              f"{decision.salience:>5.2f} {decision.prediction_error:>5.2f}  "
              f"{category or 'none':<12} {content[:50]}")

    print(f"\n{'='*70}")
    print(f"SUMMARY: {len(encoded)} encoded, {len(skipped)} skipped")
    print(f"\nEncoded:")
    for e in encoded:
        print(f"  ✓ {e[:70]}")
    print(f"\nSkipped:")
    for s in skipped:
        print(f"  ✗ {s[:70]}")

    # Sanity checks
    print(f"\n{'='*70}")
    print("SANITY CHECKS")
    print(f"{'='*70}")

    checks = [
        ("'hey what's up' should be SKIP", "hey what's up" in skipped),
        ("'cool cool' should be SKIP", any("cool cool" in s for s in skipped)),
        ("'haha yeah' should be SKIP", any("haha yeah" in s for s in skipped)),
        ("Anthropic job should be ENCODE", any("anthropic" in e.lower() for e in encoded)),
        ("$285k should be ENCODE", any("285" in e for e in encoded)),
        ("Engaged/proposed should be ENCODE", any("engaged" in e.lower() or "proposed" in e.lower() for e in encoded)),
        ("San Francisco move should be ENCODE", any("san francisco" in e.lower() for e in encoded)),
    ]

    all_pass = True
    for desc, result in checks:
        marker = "PASS" if result else "FAIL"
        if not result:
            all_pass = False
        print(f"  [{marker}] {desc}")

    print(f"\n  {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")


if __name__ == "__main__":
    main()
