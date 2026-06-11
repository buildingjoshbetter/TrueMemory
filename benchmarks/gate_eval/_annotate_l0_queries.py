"""Augment l0_queries.json with gold_keywords — positive lexical signals.

A message is relevant to a query if it contains ≥1 gold keyword (case-insensitive,
substring). Combined with the existing must_not_contain filter this gives a
deterministic, relevance-aware scoring function for A/B queries.

Annotations are hand-authored based on each query's gold_answer_gist + the
persona spec. No LLM call. Idempotent.
"""

from __future__ import annotations

import json
from pathlib import Path

DATASETS = Path(__file__).resolve().parent / "datasets"

# Hand-authored gold_keywords per query_id. Keys are query_id; value is a
# list of lexical tokens (substrings). Empty list = "persona-filter only;
# any message from the gold persona is considered relevant."
GOLD_KEYWORDS = {
    # --- P1 Alex (Rust dev, vegan, Berlin) ---
    "P1_A_001": ["lentil", "falafel", "tempeh", "vegan", "saueraut", "sauerkraut", "bean", "eat", "cook", "dinner"],
    "P1_A_002": ["bouldering", "boulder", "climb", "v4", "v5", "walk", "factory", "marzipan", "woodwork", "weekend"],
    "P1_A_003": ["le guin", "lem", "banks", "book", "read", "reading", "novel"],
    "P1_A_004": ["rust", "cargo", "tokio", "postgres", "debian", "neovim", "tmux", "msrv", "unsafe", "no_std", "deploy"],
    "P1_A_005": ["marzipan", "cat", "vet", "shelter", "rescue"],
    "P1_B_001": ["vegan", "meat", "steak"],
    "P1_B_002": ["boulder", "quiet", "walk", "factory", "cook", "small", "gym"],
    "P1_A_006": ["union", "bundesliga", "berlin", "match"],
    "P1_A_007": ["sam", "chicago", "ingrid", "brother", "mother", "father", "mom", "museum", "textile"],
    "P1_B_003": ["night", "1am", "evening", "late", "bed"],
    "P1_A_008": ["pilsner", "beer", "bar"],
    "P1_A_009": ["walnut", "allerg"],
    "P1_B_004": ["open-plan", "coworking", "no", "hate", "prefer"],
    "P1_A_010": ["marzipan", "cat", "shelter", "adopt"],
    # --- P2 Jamie ---
    "P2_A_001": ["french", "japanese", "omakase", "kaiseki", "bouillabaisse", "restaurant", "tasting", "carlos"],
    "P2_A_002": ["half", "ironman", "triathlon", "wildflower", "swim", "bike", "run", "brick"],
    "P2_A_003": ["chopin", "nocturne", "op", "62", "piano", "practice"],
    "P2_A_004": ["indemnification", "mac", "deal", "purchase agreement", "closing", "earnout", "seller", "buyer"],
    "P2_A_005": ["burgundy", "rhône", "rhone", "wine", "pommard", "pair", "laurent-perrier"],
    "P2_B_001": ["no", "chain", "avoid", "chipotle", "prefer"],
    "P2_B_002": ["5:30", "espresso", "swim", "run", "morning", "up at"],
    "P2_A_006": ["elena", "thomas", "carlos", "stanford", "design"],
    "P2_A_007": ["pro bono", "clinic", "immigrant", "saturday", "volunteer"],
    "P2_B_003": ["swim", "training", "early", "8am", "conflict"],
    "P2_A_008": ["penicillin", "allerg"],
    "P2_A_009": ["scotch", "single-malt", "whisky", "whiskey"],
    "P2_B_004": ["no", "piano", "triathlon", "chopin", "prefer"],
    "P2_A_010": ["paris", "provence", "kyoto", "tohoku", "travel", "trip"],
    # --- P3 Taylor ---
    "P3_A_001": ["konbini", "onigiri", "sando", "egg", "rice", "tamago", "pescatarian"],
    "P3_A_002": ["ren'py", "renpy", "vn", "chapter", "episode", "draft", "ep"],
    "P3_A_003": ["mia", "hanami", "izakaya", "yoyogi", "food tour"],
    "P3_A_004": ["pescatarian", "fish", "egg", "no beef", "karaage", "natto"],
    "P3_A_005": ["oshi", "idol", "group", "solo"],
    "P3_B_001": ["no", "beef", "pescatarian", "prefer", "tempura", "soba"],
    "P3_B_002": ["matcha", "starbucks", "barley", "coffee"],
    "P3_A_006": ["mom", "ruth", "vancouver", "kenji", "dad", "canon", "ae-1", "camera"],
    "P3_A_007": ["student", "class", "cram", "esl", "lesson", "kids"],
    "P3_B_003": ["no", "festival", "matsuri", "queer", "prefer"],
    "P3_A_008": ["tatsuro", "yamashita", "anri", "j-pop", "city pop", "idol"],
    "P3_A_009": ["lgbtq", "helpline", "youth", "volunteer"],
    "P3_B_004": ["no", "online", "newsletter", "vn", "bluesky"],
    "P3_A_010": ["bike", "commute", "yoga", "walk"],
    # --- P4 Morgan ---
    "P4_A_001": ["kit-bash", "kit bash", "coach", "ho", "scale", "branch line", "1952"],
    "P4_A_002": ["holland", "rise of germany", "naval", "bismarck", "chapter"],
    "P4_A_003": ["matthew", "july", "grandson", "visit", "lionel", "4-8-4"],
    "P4_A_004": ["heritage", "railway", "board", "fundraising", "chair"],
    "P4_A_005": ["church", "congregational", "pastor", "breakfast", "deacon", "bible", "sermon"],
    "P4_B_001": ["no", "raw", "traditional", "avoid", "prefer"],
    "P4_B_002": ["pot roast", "oyster", "sunday roast", "red wine", "dinner"],
    "P4_A_006": ["david", "pediatrician", "chapel hill", "matthew", "north carolina"],
    "P4_A_007": ["eleanor", "widower", "wife"],
    "P4_A_008": ["baxter", "walk", "vermont", "brattleboro", "river"],
    "P4_A_009": ["halsey", "navy", "radar", "pacific", "destroyer"],
    "P4_A_010": ["ham radio", "cw", "morse", "40 meters"],
    "P4_B_003": ["no", "outside", "not interested"],
    "P4_B_004": ["maple", "walnut", "apple crisp", "eleanor", "dessert"],
    # --- P5 Riley ---
    "P5_A_001": ["ap bio", "chem", "english", "spanish", "homework", "essay", "quiz"],
    "P5_A_002": ["sophie", "maya", "drama", "dating", "ex"],
    "P5_A_003": ["practice", "soccer", "coach ramirez", "ramirez", "jv", "sprint", "game"],
    "P5_A_004": ["abuela", "ropa vieja", "picadillo", "cuban", "bean", "rice"],
    "P5_A_005": ["taquito", "kitten", "cat", "sock"],
    "P5_B_001": ["no", "not", "chipotle", "cuban", "prefer"],
    "P5_B_002": ["pink drink", "celsius", "starbucks", "bustelo", "morning"],
    "P5_A_006": ["retinol", "skincare", "breakout", "routine"],
    "P5_A_007": ["newjeans", "le sserafim", "bts", "k-pop", "kpop", "comeback"],
    "P5_B_003": ["no", "mushroom", "hate", "picky"],
    "P5_A_008": ["sofía", "sofia", "mom", "mateo", "abuela", "miami", "brother"],
    "P5_A_009": ["sports medicine", "doctor", "anatomy", "pre-med", "medicine"],
    "P5_A_010": ["gpa", "bio", "chem", "anxious", "grade"],
    "P5_B_004": ["soccer", "boba", "tiktok", "friend", "saturday"],
}


def main() -> int:
    path = DATASETS / "l0_queries.json"
    data = json.loads(path.read_text())
    annotated = 0
    skipped_non_ab = 0
    for q in data["queries"]:
        if q["query_type"] in ("A", "B"):
            if q["query_id"] in GOLD_KEYWORDS:
                q["gold_keywords"] = GOLD_KEYWORDS[q["query_id"]]
                annotated += 1
            else:
                q["gold_keywords"] = []  # empty → persona-filter only
        else:
            skipped_non_ab += 1
    data["meta"]["annotated_at"] = "2026-04-23"
    data["meta"]["annotation_note"] = (
        "gold_keywords added for A/B queries. A candidate's top-1 result is "
        "relevance-correct iff: (a) from gold persona, (b) contains no "
        "must_not_contain words, (c) contains ≥1 gold_keyword (or "
        "gold_keywords is empty, meaning persona-filter is the only signal)."
    )
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Annotated {annotated} A/B queries with gold_keywords.")
    print(f"Non-A/B queries (no annotation needed): {skipped_non_ab}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
