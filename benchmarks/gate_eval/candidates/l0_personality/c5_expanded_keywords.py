"""C5 — expanded keyword clusters (same mechanism as C1, wider coverage).

Tests the hypothesis "the gap is just coverage, not the mechanism."

Expansion:
  * topic clusters: 10 → 50 (woodworking, classical_music, soccer, K-pop,
    anime, ESL teaching, model_trains, WW2, etc.)
  * trait indicators: 10 → 25 (curious, introverted, playful, …)
  * formality markers: original anglophone set + cross-register additions
    (letter-style, Gen-Z slang, textspeak emoji-heavy)

Scoring at retrieval time also does persona-aware filtering — a correction
to C1's design flaw surfaced by the smoke test, to make the coverage
comparison fair (C5 should be measured with persona awareness so that
the delta C5-vs-C1 isolates "coverage" rather than also capturing
"persona-scoping missing in C1").
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.gate_eval.candidates.l0_personality._base import (
    L0Candidate,
    Profile,
    RerankResult,
)


# ───────────────────────── expanded topic clusters (50) ─────────────────────────
TOPIC_CLUSTERS = {
    # tech + cs
    "software_engineering": {"code", "debug", "compile", "deploy", "api", "bug", "refactor",
                             "repo", "git", "commit", "pr", "ci"},
    "rust_programming": {"rust", "cargo", "tokio", "borrow", "msrv", "no_std", "crate"},
    "python_ecosystem": {"python", "pip", "uv", "pyproject", "venv", "conda"},
    "javascript_web": {"javascript", "typescript", "node", "npm", "react", "vue"},
    "infra_devops": {"kubernetes", "docker", "aws", "terraform", "helm", "prometheus"},
    "databases": {"postgres", "sqlite", "redis", "mongo", "bigquery", "snowflake"},
    # fitness + sport
    "triathlon": {"swim", "bike", "run", "brick", "ironman", "half", "wildflower", "transition"},
    "soccer": {"soccer", "football", "match", "bundesliga", "goal", "assist", "midfielder", "coach"},
    "classical_music": {"chopin", "beethoven", "ravel", "sonata", "nocturne", "piano",
                        "cadenza", "score", "movement", "op."},
    "bouldering_climbing": {"boulder", "climb", "v4", "v5", "route", "crimp", "project", "gym"},
    "yoga_pilates": {"yoga", "asana", "savasana", "mat", "downward"},
    "golf": {"golf", "tee", "par", "fairway", "iron", "wedge"},
    "running_athletics": {"marathon", "pace", "threshold", "track", "interval"},
    # food + drink
    "vegan_vegetarian": {"vegan", "vegetarian", "tempeh", "tofu", "seitan", "lentil", "chickpea",
                          "oat milk"},
    "cooking_home": {"cook", "recipe", "simmer", "saute", "dinner", "lunch", "breakfast"},
    "french_cuisine": {"bouillabaisse", "cassoulet", "ratatouille", "tarte", "confit"},
    "japanese_cuisine": {"sushi", "omakase", "kaiseki", "onigiri", "ramen", "soba", "izakaya",
                          "karaage", "konbini"},
    "italian_cuisine": {"pasta", "pizza", "risotto", "tiramisu", "espresso"},
    "cuban_latin_food": {"ropa vieja", "picadillo", "moros", "bustelo"},
    "coffee_tea": {"coffee", "latte", "espresso", "matcha", "tea"},
    "wine_spirits": {"wine", "rhone", "burgundy", "pommard", "champagne", "whiskey", "scotch",
                      "cocktail", "bourbon"},
    "beer_brewery": {"beer", "pilsner", "ipa", "pale ale", "brewery"},
    # relationships + life events
    "romantic_partner": {"girlfriend", "boyfriend", "wife", "husband", "partner", "dating",
                          "anniversary"},
    "family_core": {"mom", "dad", "mother", "father", "sister", "brother", "sibling"},
    "kids_parenting": {"kids", "son", "daughter", "child", "baby", "toddler"},
    "grief_loss": {"passed", "died", "funeral", "widower", "widow", "late"},
    # work + career
    "law_practice": {"indemnification", "mac", "closing", "earnout", "purchase agreement",
                       "covenant", "counsel", "associate"},
    "medicine_health": {"doctor", "cardiologist", "pediatrician", "vet", "nurse", "clinic",
                         "hospital", "surgeon", "patient"},
    "teaching_education": {"student", "class", "lesson", "school", "teacher", "esl",
                            "curriculum", "homework", "quiz", "test", "grade"},
    "engineering_manufacturing": {"turbine", "manufacturing", "factory", "machine shop"},
    "finance_invest": {"salary", "savings", "mortgage", "401k", "stock", "investment"},
    # hobbies + crafts
    "woodworking": {"woodwork", "lathe", "sanding", "wood", "joinery"},
    "model_railroading": {"locomotive", "kit-bash", "ho", "scale", "lionel", "coach", "consist",
                            "roundhouse"},
    "photography": {"canon", "nikon", "lens", "aperture", "shutter", "ae-1"},
    "gardening": {"garden", "perennial", "peony", "bed", "compost"},
    "diy_home": {"drill", "plumbing", "renovation", "sheetrock"},
    "anime_manga": {"anime", "manga", "shonen", "fanfic", "oshi", "idol group", "waifu"},
    "kpop_jpop": {"k-pop", "kpop", "newjeans", "bts", "blackpink", "j-pop", "city pop"},
    "videogames": {"game", "gaming", "xbox", "playstation", "nintendo", "steam", "roblox",
                     "minecraft"},
    "visual_novel_dev": {"ren'py", "renpy", "vn", "visual novel", "chapter", "episode"},
    # reading + media
    "scifi_literary": {"le guin", "iain banks", "stanislaw lem", "asimov", "ted chiang",
                         "culture novel"},
    "ww2_history": {"ww2", "bismarck", "convoy", "u-boat", "halsey", "enigma", "panzer"},
    "military_history": {"navy", "destroyer", "frigate", "submarine", "regiment"},
    # social + community
    "religion_church": {"church", "congregational", "pastor", "deacon", "sermon", "prayer",
                          "scripture", "bible"},
    "lgbtq_community": {"lgbtq", "queer", "pride", "ally", "trans", "nonbinary"},
    "activism_volunteer": {"volunteer", "helpline", "pro bono", "clinic", "immigrant"},
    # travel + place
    "europe_travel": {"berlin", "paris", "provence", "burgundy", "rome"},
    "japan_travel": {"tokyo", "kyoto", "tohoku", "osaka", "hanami", "matsuri", "sakura"},
    "us_domestic_travel": {"napa", "vermont", "san francisco", "atlanta", "miami"},
    # mind + body states
    "anxiety_stress": {"worried", "anxious", "stress", "overwhelmed", "panic", "nervous"},
    "sleep_routine": {"sleep", "bed", "morning", "up at", "wake", "bedtime", "routine"},
}


# ───────────────────────── expanded trait indicators (25) ────────────────────────
TRAIT_INDICATORS = {
    "ambitious": {"goal", "growth", "raise", "scale", "launch"},
    "anxious": {"worried", "anxious", "stress", "panic", "what if"},
    "caring": {"love", "care", "miss", "proud", "support"},
    "analytical": {"data", "metrics", "measure", "analysis", "benchmark"},
    "social": {"drinks", "party", "hangout", "dinner with", "catch up"},
    "health_conscious": {"gym", "workout", "run", "meditation", "diet", "fitness"},
    "technical": {"code", "api", "database", "algorithm", "architecture"},
    "family_oriented": {"mom", "dad", "family", "parents", "home"},
    "entrepreneurial": {"startup", "founder", "pitch", "investor", "equity"},
    "loyal": {"always", "forever", "promise", "count on", "got your back"},
    # New in C5:
    "introverted": {"quiet", "alone", "recharge", "tired of people", "small group"},
    "extraverted": {"party", "group", "crowd", "stage", "big event"},
    "curious": {"why", "how does", "wonder", "read about", "explore"},
    "conscientious": {"schedule", "on time", "checklist", "deadline", "prepared"},
    "playful": {"haha", "lol", "joke", "fun", "teasing"},
    "empathetic": {"hear you", "understand", "totally valid", "i get"},
    "contrarian": {"actually", "disagree", "counterpoint", "not convinced"},
    "spiritual_religious": {"pray", "blessing", "sermon", "god", "faith"},
    "artistic": {"paint", "draw", "write", "compose", "creative"},
    "formal_professional": {"kindly", "please advise", "pursuant", "regards"},
    "casual_breezy": {"lol", "lowkey", "bet", "fr", "bruh"},
    "detail_oriented": {"notice", "exact", "precise", "specific"},
    "bookish": {"book", "chapter", "novel", "reading", "library"},
    "cinephile": {"film", "movie", "director", "screenwriter", "shot"},
    "outdoorsy": {"hike", "trail", "camp", "mountain", "river"},
}


CASUAL_MARKERS = {
    "lol", "haha", "omg", "gonna", "wanna", "gotta", "yeah", "yep", "nah",
    "bruh", "dude", "tbh", "idk", "imo", "btw", "ngl", "fr", "rn", "lmao",
    # Gen-Z additions:
    "bestie", "slay", "no cap", "bet", "ate", "mid", "fr fr", "ok bestie",
    # Textspeak / online
    "lowkey", "highkey", "iykyk", "honestly??", "pls", "thx",
}
FORMAL_MARKERS = {
    "shall", "furthermore", "thus", "indeed", "kindly", "please advise",
    "regards", "sincerely", "dear", "pursuant", "notwithstanding",
    "i should think", "on balance", "as it happens", "in my experience",
    "that said",
}
EMOJI_RE = re.compile(r"["
                       r"\U0001f600-\U0001f64f"
                       r"\U0001f300-\U0001f5ff"
                       r"\U0001f680-\U0001f6ff"
                       r"\U0001f1e0-\U0001f1ff"
                       r"\U0001f900-\U0001f9ff"
                       r"\U0001fa00-\U0001faff"
                       r"☀-➿"
                       r"]+", re.UNICODE)


def detect_emoji(text: str) -> bool:
    return bool(EMOJI_RE.search(text))


def assess_formality(text: str) -> str:
    low = text.lower()
    casual = sum(1 for m in CASUAL_MARKERS if m in low)
    formal = sum(1 for m in FORMAL_MARKERS if m in low)
    all_lower = text == text.lower()
    has_period = text.rstrip().endswith(".")
    starts_cap = text[:1].isupper() if text else False
    long_msg = len(text) > 200
    if casual > formal and (all_lower or detect_emoji(text)):
        return "casual"
    if casual > 0 and formal == 0:
        return "casual"
    if formal > casual and starts_cap and has_period:
        return "formal"
    if long_msg and starts_cap and has_period and formal >= casual:
        return "formal"
    return "mixed"


def extract_topics(messages: list[dict], min_hits: int = 2) -> list[str]:
    counts: dict[str, int] = defaultdict(int)
    for m in messages:
        low = m["text"].lower()
        for topic, keywords in TOPIC_CLUSTERS.items():
            for kw in keywords:
                if kw in low:
                    counts[topic] += 1
    return [t for t, c in sorted(counts.items(), key=lambda x: -x[1])
            if c >= min_hits]


def extract_traits(messages: list[dict]) -> list[str]:
    counts: dict[str, int] = defaultdict(int)
    for m in messages:
        low = m["text"].lower()
        for trait, inds in TRAIT_INDICATORS.items():
            for ind in inds:
                if ind in low:
                    counts[trait] += 1
    threshold = max(2, len(messages) // 20)  # less strict than C1
    return [t for t, c in sorted(counts.items(), key=lambda x: -x[1])
            if c >= threshold]


class C5ExpandedKeywords(L0Candidate):
    name = "c5_expanded_keywords"
    tier = "edge"
    consumes_at_retrieval = True

    def build_profile(self, persona_id: str,
                      messages: list[dict]) -> Profile:
        msgs = [{"text": m["text"]} for m in messages]
        if not msgs:
            return Profile(persona_id=persona_id, data={}, bytes_estimate=2,
                           readable_summary="(empty)")

        # Aggregate stats
        avg_length = sum(len(m["text"]) for m in msgs) / len(msgs)
        emoji_rate = sum(1 for m in msgs if detect_emoji(m["text"])) / len(msgs)
        emoji_level = ("heavy" if emoji_rate > 0.5 else
                        "frequent" if emoji_rate > 0.25 else
                        "minimal" if emoji_rate > 0.05 else "none")
        formality_counts = {"casual": 0, "formal": 0, "mixed": 0}
        for m in msgs:
            formality_counts[assess_formality(m["text"])] += 1
        dominant_formality = max(formality_counts, key=formality_counts.get)

        topics = extract_topics(msgs)
        traits = extract_traits(msgs)

        data = {
            "message_count": len(msgs),
            "communication_style": {
                "avg_length": round(avg_length, 1),
                "emoji_level": emoji_level,
                "formality": dominant_formality,
            },
            "topics": topics[:20],   # cap
            "traits": traits[:15],
        }
        raw = json.dumps(data, ensure_ascii=False)
        summary = (f"{persona_id} · {dominant_formality} · emoji={emoji_level} · "
                   f"topics={topics[:4]} · traits={traits[:4]}")
        return Profile(
            persona_id=persona_id, data=data,
            bytes_estimate=len(raw.encode("utf-8")),
            readable_summary=summary,
        )

    def score_for_personalization(self, query: str,
                                  active_profile: Profile,
                                  candidate_messages: list[dict]
                                  ) -> list[RerankResult]:
        data = active_profile.data
        persona_topics = set(data.get("topics", []))
        persona_traits = set(data.get("traits", []))
        persona_formality = (data.get("communication_style", {})
                              .get("formality", "mixed"))
        persona_emoji_level = (data.get("communication_style", {})
                                .get("emoji_level", "none"))

        results = []
        for msg in candidate_messages:
            text = msg["text"]
            same_user = msg.get("source_persona_id", "") == active_profile.persona_id

            low = text.lower()

            # Topic cluster hit from persona's topic list
            topic_score = 0.0
            for topic in persona_topics:
                for kw in TOPIC_CLUSTERS.get(topic, ()):
                    if kw in low:
                        topic_score += 1.0
                        break

            # Trait hit
            trait_score = 0.0
            for trait in persona_traits:
                for ind in TRAIT_INDICATORS.get(trait, ()):
                    if ind in low:
                        trait_score += 1.0
                        break

            # Formality alignment
            msg_formality = assess_formality(text)
            formality_score = 1.0 if msg_formality == persona_formality else 0.0

            # Emoji alignment
            msg_has_emoji = detect_emoji(text)
            expects_emoji = persona_emoji_level in ("heavy", "frequent")
            emoji_score = 1.0 if msg_has_emoji == expects_emoji else 0.0

            # Persona-scoping bias (this is the fix for C1's design flaw).
            user_score = 5.0 if same_user else 0.0

            total = (user_score
                     + 0.5 * topic_score
                     + 0.3 * trait_score
                     + 0.2 * formality_score
                     + 0.1 * emoji_score)
            results.append(RerankResult(
                message_text=text,
                source_persona_id=msg.get("source_persona_id", ""),
                score=total,
                metadata={
                    "topic_score": topic_score, "trait_score": trait_score,
                    "formality_score": formality_score, "emoji_score": emoji_score,
                },
            ))
        results.sort(key=lambda r: r.score, reverse=True)
        return results
