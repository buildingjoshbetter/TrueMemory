"""
TrueMemory L0 — Personality Engram
=================================

Extracts personality profiles from message patterns at ingest time.  Every
competitor scored <=3/10 on personality queries because generic retrieval
treats ``"What kind of person is Jordan?"`` as a keyword lookup — which
returns nothing useful.

This module pre-computes entity profiles (communication style, topics,
relationships, traits, preferences) so that personality questions can be
answered from structured data rather than raw message search alone.

Personality categories handled:
    - General character (``"What kind of person is X?"``)
    - Food and drink preferences
    - Communication style differences per recipient
    - Fears and insecurities
    - Routines and habits

All functions operate on a ``sqlite3.Connection`` produced by
:func:`truememory.storage.create_db`.  No external dependencies beyond the
Python standard library.
"""

import contextlib
import json
import re
import sqlite3
import warnings
from collections import defaultdict
from datetime import datetime, timezone

from truememory.fts_search import _build_safe_fts_query


# ---------------------------------------------------------------------------
# Keyword dictionaries for preference extraction
# ---------------------------------------------------------------------------

_FOOD_KEYWORDS = {
    "eat", "eating", "ate", "lunch", "dinner", "breakfast", "brunch",
    "restaurant", "restaurants", "food", "cook", "cooking", "cooked",
    "pizza", "sushi", "tacos", "burger", "burgers", "steak", "chicken",
    "salad", "ramen", "pasta", "sandwich", "bbq", "brisket", "ribs",
    "coffee", "latte", "espresso", "tea", "beer", "wine", "cocktail",
    "drinks", "bar", "brewery", "pub", "oat milk", "oats", "eggs",
    "smoothie", "vegan", "vegetarian", "omakase", "charcuterie",
    "appetizer", "dessert", "cuisine", "dish", "meal", "snack",
    "recipe", "diner", "cafe",
}

_ACTIVITY_KEYWORDS = {
    "gym", "run", "running", "ran", "workout", "exercise", "lift",
    "lifting", "crossfit", "yoga", "meditation", "meditate",
    "swim", "swimming", "hike", "hiking", "bike", "biking", "cycling",
    "watch", "watched", "watching", "show", "movie", "film", "read",
    "reading", "book", "podcast", "music", "concert", "festival",
    "game", "gaming", "play", "played", "golf", "tennis", "soccer",
    "basketball", "football", "baseball", "marathon", "race", "trail",
    "climb", "climbing", "surf", "surfing", "ski", "skiing",
}

_FEAR_KEYWORDS = {
    "worried", "worries", "worry", "worrying", "scared", "scary",
    "afraid", "anxious", "anxiety", "nervous", "panic", "panicking",
    "stress", "stressed", "stressful", "fear", "fears", "terrified",
    "what if", "freaking out", "freak out", "can't sleep",
    "overwhelmed", "overwhelming", "burnout", "burn out",
    "imposter", "imposter syndrome", "doubt", "doubts", "uncertain",
    "uncertainty", "insecure", "insecurity", "chest tightness",
    "worst case", "nightmare", "dread",
}

_ROUTINE_KEYWORDS = {
    "morning", "mornings", "every day", "daily", "routine", "routines",
    "always", "usually", "every morning", "every night", "every week",
    "weekly", "habit", "habits", "schedule", "ritual",
    "wake up", "woke up", "alarm", "6am", "7am", "bedtime",
    "night routine", "morning routine", "workout routine",
    "every monday", "every friday",
}

_VALUE_KEYWORDS = {
    "important", "priority", "priorities", "value", "values",
    "believe", "believes", "belief", "principle", "principles",
    "decision", "decided", "choose", "chose", "choice",
    "commit", "committed", "commitment", "boundary", "boundaries",
    "hard stop", "non-negotiable", "matter", "matters",
    "care about", "stand for", "mission",
}

# Personality-aspect detection for search queries
PERSONALITY_ASPECTS = {
    "food": {
        "keywords": {"eat", "food", "drink", "coffee", "restaurant",
                     "cuisine", "breakfast", "lunch", "dinner", "taco",
                     "sushi", "cook", "diet", "meal", "snack"},
        "fts_terms": ["eat", "food", "restaurant", "coffee", "lunch",
                      "dinner", "breakfast", "cook", "drink"],
    },
    "communication": {
        "keywords": {"communicate", "talk", "text", "message", "style",
                     "differently", "tone", "speak", "say", "respond"},
        "fts_terms": ["message", "text", "talk", "hey", "lol"],
    },
    "fears": {
        "keywords": {"fear", "afraid", "scared", "worry", "anxious",
                     "anxiety", "insecurity", "insecure", "nervous",
                     "stress", "concern"},
        "fts_terms": ["worried", "scared", "anxious", "stress", "fear",
                      "panic", "overwhelmed", "burnout"],
    },
    "routines": {
        "keywords": {"routine", "morning", "daily", "habit", "schedule",
                     "every day", "wake", "night", "ritual"},
        "fts_terms": ["morning", "routine", "daily", "wake", "alarm",
                      "every day", "gym", "meditate"],
    },
    "personality": {
        "keywords": {"kind of person", "personality", "character", "trait",
                     "describe", "like as a person", "what is.*like",
                     "who is"},
        "fts_terms": [],  # personality uses the profile, not FTS
    },
    "activities": {
        "keywords": {"hobby", "hobbies", "fun", "free time", "weekend",
                     "activity", "activities", "enjoy", "watch", "play",
                     "exercise", "sport"},
        "fts_terms": ["gym", "run", "watch", "read", "play", "hike",
                      "concert", "festival", "marathon"],
    },
    "relationships": {
        "keywords": {"relationship", "friend", "friends", "family",
                     "closest", "talk to", "hang out", "trust"},
        "fts_terms": [],  # relationships use the profile
    },
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_all_messages(conn: sqlite3.Connection) -> list[dict]:
    """Fetch all messages ordered by timestamp."""
    rows = conn.execute(
        "SELECT id, content, sender, recipient, timestamp, category, modality "
        "FROM messages ORDER BY timestamp"
    ).fetchall()
    return [
        {
            "id": r[0], "content": r[1], "sender": r[2],
            "recipient": r[3], "timestamp": r[4],
            "category": r[5], "modality": r[6],
        }
        for r in rows
    ]


def _get_messages_by_sender(conn: sqlite3.Connection, sender: str) -> list[dict]:
    """Fetch all messages for a specific sender, ordered by timestamp."""
    rows = conn.execute(
        "SELECT id, content, sender, recipient, timestamp, category, modality "
        "FROM messages WHERE LOWER(sender) = LOWER(?) ORDER BY timestamp",
        (sender,),
    ).fetchall()
    return [
        {
            "id": r[0], "content": r[1], "sender": r[2],
            "recipient": r[3], "timestamp": r[4],
            "category": r[5], "modality": r[6],
        }
        for r in rows
    ]


def _content_matches_any(content: str, keywords: set) -> list[str]:
    """Return which keywords appear in the lowered content."""
    lower = content.lower()
    return [kw for kw in keywords if kw in lower]


def _extract_proper_nouns(content: str) -> list[str]:
    """
    Extract capitalized words that look like proper nouns.
    Simple heuristic: words starting with uppercase that are not at sentence
    start and are not common English words.
    """
    common = {
        "I", "The", "A", "An", "Is", "It", "He", "She", "We", "They",
        "My", "Your", "His", "Her", "Our", "This", "That", "What",
        "When", "Where", "Who", "How", "Why", "But", "And", "Or",
        "So", "If", "Not", "No", "Yes", "Can", "Will", "Just",
        "Do", "Did", "Has", "Have", "Had", "Was", "Were", "Are",
        "Been", "Be", "Would", "Could", "Should", "May", "Might",
        "Need", "Want", "Know", "Think", "Got", "Get", "Let",
        "Also", "Still", "Even", "Too", "Very", "Really", "About",
        "Like", "Here", "There", "Now", "Then", "Well", "Hey",
        "Yeah", "Yep", "Ok", "Okay", "Thanks", "Thank", "Sure",
        "Oh", "Wow", "Lol", "Haha", "Omg", "For", "With",
        "From", "Into", "Over", "After", "Before", "Between",
        "During", "Through", "Some", "Any", "All", "Each", "Every",
        "New", "Good", "Great", "First", "Last", "Going", "Looking",
    }
    words = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", content)
    return [w for w in words if w not in common]


_EMOJI_RE = re.compile(
    "[\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"   # symbols & pictographs
    "\U0001f680-\U0001f6ff"   # transport & map
    "\U0001f1e0-\U0001f1ff"   # flags
    "\U00002702-\U000027b0"
    "\U000024c2-\U0001f251"
    "\U0001f900-\U0001f9ff"   # supplemental symbols
    "\U0001fa00-\U0001fa6f"   # chess symbols
    "\U0001fa70-\U0001faff"   # extended-A
    "]+",
    flags=re.UNICODE,
)


def _detect_emoji(content: str) -> bool:
    """Return True if content contains emoji characters."""
    return bool(_EMOJI_RE.search(content))


def _assess_formality(messages: list[dict]) -> str:
    """
    Classify formality as 'casual', 'formal', or 'mixed' based on message
    characteristics.

    Casual indicators: all lowercase, abbreviations, slang, emoji.
    Formal indicators: proper capitalization, longer sentences, no slang.
    """
    if not messages:
        return "mixed"

    casual_count = 0
    formal_count = 0
    casual_markers = {"lol", "haha", "omg", "gonna", "wanna", "gotta",
                      "yeah", "yep", "nah", "bruh", "dude", "tbh",
                      "idk", "imo", "btw", "ngl", "fr", "rn", "lmao"}

    for msg in messages:
        text = msg["content"]
        lower = text.lower()

        # Check casual markers
        has_casual = any(m in lower.split() for m in casual_markers)
        all_lower = text == text.lower()
        has_emoji = _detect_emoji(text)
        short = len(text) < 50

        # Check formal markers
        starts_cap = text[0:1].isupper() if text else False
        has_period = text.rstrip().endswith(".")
        long_msg = len(text) > 200

        if (has_casual or all_lower or has_emoji) and short:
            casual_count += 1
        elif (starts_cap and has_period) or long_msg:
            formal_count += 1

    total = casual_count + formal_count
    if total == 0:
        return "mixed"
    ratio = casual_count / total
    if ratio > 0.65:
        return "casual"
    elif ratio < 0.35:
        return "formal"
    return "mixed"


def _find_typical_greeting(messages: list[dict]) -> str:
    """Find the most common way an entity starts conversations."""
    greeting_patterns = defaultdict(int)
    greeting_words = {"hey", "hi", "hello", "yo", "sup", "what's up",
                      "good morning", "morning", "heyy", "heyyy"}

    for msg in messages:
        first_word = msg["content"].lower().split()[0] if msg["content"].strip() else ""
        first_two = " ".join(msg["content"].lower().split()[:2]) if msg["content"].strip() else ""

        if first_two in greeting_words:
            greeting_patterns[first_two] += 1
        elif first_word in greeting_words:
            greeting_patterns[first_word] += 1

    if not greeting_patterns:
        return ""
    return max(greeting_patterns, key=greeting_patterns.get)


def _extract_topics(messages: list[dict]) -> list[str]:
    """
    Extract frequent topics from a set of messages using keyword frequency
    analysis and proper noun extraction.

    Returns the top topics sorted by frequency.
    """
    import warnings
    warnings.warn(
        "_extract_topics is deprecated as of v0.6.0 per MEMORIST-L0 research. "
        "Keyword-based scoring replaced by char-n-gram style vectors in v0.6.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    topic_counts: dict[str, int] = defaultdict(int)

    # Domain-specific topic clusters
    topic_clusters = {
        "startup/business": {"startup", "company", "founded", "revenue",
                             "funding", "investor", "pitch", "valuation",
                             "mrr", "arr", "series", "seed", "raise"},
        "technology": {"code", "deploy", "database", "api", "backend",
                       "frontend", "server", "aws", "cloud", "repo",
                       "kubernetes", "docker", "github", "python", "go",
                       "rewrite", "migrate", "typescript", "javascript"},
        "health/fitness": {"gym", "run", "running", "workout", "health",
                           "marathon", "sleep", "meditation", "therapy",
                           "doctor", "weight", "resting heart rate"},
        "relationships": {"dating", "anniversary", "valentine", "proposal",
                          "ring", "wedding", "girlfriend", "boyfriend",
                          "partner", "date night", "love"},
        "food/dining": {"restaurant", "coffee", "tacos", "sushi", "dinner",
                        "lunch", "breakfast", "eat", "cook", "bar"},
        "finance": {"salary", "savings", "money", "bank", "mortgage",
                    "house", "rent", "401k", "investment", "budget",
                    "expense", "burn rate"},
        "career": {"job", "quit", "hired", "hiring", "employee",
                   "cofounder", "cto", "ceo", "role", "interview",
                   "offer", "equity", "promotion"},
        "pets": {"dog", "puppy", "vet", "walk",
                 "shelter", "pet"},
        "entertainment": {"show", "movie", "netflix", "hulu", "watch",
                          "concert", "festival", "book",
                          "game"},
        "family": {"mom", "dad", "sister", "brother", "parents",
                   "family", "thanksgiving", "christmas", "birthday"},
    }

    for msg in messages:
        lower = msg["content"].lower()
        words = set(lower.split())

        for topic, keywords in topic_clusters.items():
            overlap = words & keywords
            if overlap:
                topic_counts[topic] += len(overlap)

    # Sort by frequency, return topic names
    sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
    return [t[0] for t in sorted_topics if t[1] >= 2]


def _extract_traits(messages: list[dict]) -> list[str]:
    """
    Infer personality traits from message patterns.

    Uses a trait-indicator mapping: if enough messages match the indicators,
    the trait is assigned.
    """
    import warnings
    warnings.warn(
        "_extract_traits is deprecated as of v0.6.0 per MEMORIST-L0 research. "
        "Keyword-based scoring replaced by char-n-gram style vectors in v0.6.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    trait_indicators = {
        "ambitious": {"goal", "growth", "scale", "expand", "raise",
                      "million", "revenue", "build", "launch", "grind"},
        "anxious": {"worried", "anxious", "stress", "panic", "what if",
                    "scared", "nervous", "can't sleep", "overwhelmed"},
        "caring": {"love", "miss you", "thinking of you", "proud of",
                   "care about", "how are you", "check in", "support"},
        "analytical": {"data", "metrics", "numbers", "analysis", "measure",
                       "calculate", "percentage", "accuracy", "benchmark"},
        "social": {"drinks", "party", "hang out", "get together",
                   "dinner with", "let's meet", "catch up", "plans"},
        "health-conscious": {"gym", "workout", "run", "meditation",
                             "sleep", "diet", "healthy", "fitness",
                             "organic", "supplements"},
        "technical": {"code", "deploy", "api", "database", "server",
                      "algorithm", "architecture", "framework", "debug"},
        "family-oriented": {"mom", "dad", "sister", "family", "parents",
                            "thanksgiving", "christmas", "birthday",
                            "home", "visit"},
        "entrepreneurial": {"startup", "founder", "company", "pitch",
                            "investor", "equity", "valuation", "raise"},
        "loyal": {"always", "forever", "promise", "never leave",
                  "count on", "trust", "got your back", "support"},
    }

    trait_scores: dict[str, int] = defaultdict(int)
    for msg in messages:
        lower = msg["content"].lower()
        for trait, indicators in trait_indicators.items():
            matches = sum(1 for ind in indicators if ind in lower)
            trait_scores[trait] += matches

    # Require a minimum threshold relative to message count
    min_threshold = max(2, len(messages) // 50)
    return [
        trait for trait, score in sorted(
            trait_scores.items(), key=lambda x: x[1], reverse=True
        )
        if score >= min_threshold
    ]


def _fts_search(conn: sqlite3.Connection, fts_query: str,
                limit: int = 20) -> list[dict]:
    """Run an FTS5 search and return result dicts."""
    sql = (
        "SELECT m.id, m.content, m.sender, m.recipient, m.timestamp, "
        "       m.category, m.modality, messages_fts.rank AS score "
        "FROM messages_fts "
        "JOIN messages m ON m.id = messages_fts.rowid "
        "WHERE messages_fts MATCH ? "
        "ORDER BY messages_fts.rank LIMIT ?"
    )
    try:
        rows = conn.execute(sql, (fts_query, limit)).fetchall()
    except sqlite3.OperationalError:
        return []

    return [
        {
            "id": r[0], "content": r[1], "sender": r[2],
            "recipient": r[3], "timestamp": r[4],
            "category": r[5], "modality": r[6], "score": r[7],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _warnings_ctx():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        yield


def build_entity_profiles(conn: sqlite3.Connection) -> dict:
    """
    Analyze all messages and build personality profiles for key entities.

    For each entity (sender), extracts:

    - **message_count**: total messages sent.
    - **topics**: frequent themes (startup, health, food, etc.).
    - **communication_style**: average message length, emoji usage,
      formality level, and typical greeting.
    - **relationships**: who they message most and approximate topic focus
      per recipient.
    - **traits**: personality descriptors inferred from content analysis.

    Results are stored in the ``entity_profiles`` table and also returned
    as a dict keyed by entity name.

    Args:
        conn: Open database connection (from :func:`truememory.storage.create_db`).

    Returns:
        ``{entity: profile_dict}`` for every sender who has at least one
        message in the database.
    """
    all_msgs = _get_all_messages(conn)

    # Group messages by sender
    by_sender: dict[str, list[dict]] = defaultdict(list)
    for msg in all_msgs:
        if msg["sender"]:
            by_sender[msg["sender"]].append(msg)

    profiles: dict[str, dict] = {}

    for sender, messages in by_sender.items():
        # Communication style
        lengths = [len(m["content"]) for m in messages]
        avg_length = sum(lengths) / len(lengths) if lengths else 0.0
        uses_emoji = any(_detect_emoji(m["content"]) for m in messages)
        formality = _assess_formality(messages)
        greeting = _find_typical_greeting(messages)

        comm_style = {
            "avg_length": round(avg_length, 1),
            "uses_emoji": uses_emoji,
            "formality": formality,
            "typical_greeting": greeting,
        }

        # Relationships: who they talk to and rough topic per recipient
        recipient_counts: dict[str, int] = defaultdict(int)
        recipient_topics: dict[str, list[str]] = defaultdict(list)
        for msg in messages:
            recip = msg["recipient"]
            if recip:
                recipient_counts[recip] += 1
                # Quick topic tag for each message
                lower = msg["content"].lower()
                if any(w in lower for w in ("code", "deploy", "api",
                                            "database", "bug", "server")):
                    recipient_topics[recip].append("technical")
                elif any(w in lower for w in ("worried", "anxious",
                                              "scared", "stress")):
                    recipient_topics[recip].append("emotional")
                elif any(w in lower for w in ("dinner", "drinks",
                                              "hang out", "watch")):
                    recipient_topics[recip].append("social")
                elif any(w in lower for w in ("revenue", "investor",
                                              "pricing", "customer")):
                    recipient_topics[recip].append("business")

        relationships = {}
        for recip, count in sorted(recipient_counts.items(),
                                   key=lambda x: x[1], reverse=True):
            topic_freq: dict[str, int] = defaultdict(int)
            for t in recipient_topics.get(recip, []):
                topic_freq[t] += 1
            top_topic = max(topic_freq, key=topic_freq.get) if topic_freq else "general"
            relationships[recip] = {
                "message_count": count,
                "primary_topic": top_topic,
            }

        with _warnings_ctx():
            topics = _extract_topics(messages)
            traits = _extract_traits(messages)

        profile = {
            "message_count": len(messages),
            "topics": topics,
            "communication_style": comm_style,
            "relationships": relationships,
            "traits": traits,
        }
        profiles[sender] = profile

        # Store in database
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """INSERT OR REPLACE INTO entity_profiles
               (entity, message_count, traits, communication_style,
                topics, relationships, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                sender,
                len(messages),
                json.dumps(traits),
                json.dumps(comm_style),
                json.dumps(topics),
                json.dumps(relationships),
                now,
            ),
        )

    conn.commit()
    return profiles


def extract_preferences(conn: sqlite3.Connection,
                        entity: str | None = None) -> dict:
    """
    Extract specific preferences for an entity from their messages.

    Scans every message sent by *entity* and categorizes relevant content
    into preference buckets:

    - **food**: restaurants, cuisines, dishes, drinks mentioned.
    - **activities**: hobbies, exercise, entertainment.
    - **people**: who they are closest to (by message volume + sentiment).
    - **routines**: morning/evening patterns, recurring activities.
    - **fears**: anxiety, worry, concern patterns.
    - **values**: what they prioritize (based on decision language).

    Args:
        conn:   Open database connection.
        entity: Entity name to analyze.

    Returns:
        Dict with keys ``food``, ``activities``, ``people``, ``routines``,
        ``fears``, ``values``.  Each value is a list of extracted snippets
        or structured data.
    """
    if entity is None:
        return {}
    messages = _get_messages_by_sender(conn, entity)

    preferences: dict[str, list] = {
        "food": [],
        "activities": [],
        "people": [],
        "routines": [],
        "fears": [],
        "values": [],
    }

    # Track unique snippets to avoid duplicates
    seen: dict[str, set] = {k: set() for k in preferences}

    for msg in messages:
        content = msg["content"]
        _lower = content.lower()

        # Food
        food_hits = _content_matches_any(content, _FOOD_KEYWORDS)
        if food_hits:
            # Extract the sentence(s) containing food keywords
            snippet = content[:300]
            key = snippet[:80]
            if key not in seen["food"]:
                seen["food"].add(key)
                # Also extract proper nouns (restaurant names, etc.)
                nouns = _extract_proper_nouns(content)
                preferences["food"].append({
                    "text": snippet,
                    "keywords": food_hits[:5],
                    "proper_nouns": nouns,
                    "timestamp": msg["timestamp"],
                })

        # Activities
        activity_hits = _content_matches_any(content, _ACTIVITY_KEYWORDS)
        if activity_hits:
            snippet = content[:300]
            key = snippet[:80]
            if key not in seen["activities"]:
                seen["activities"].add(key)
                preferences["activities"].append({
                    "text": snippet,
                    "keywords": activity_hits[:5],
                    "timestamp": msg["timestamp"],
                })

        # Fears
        fear_hits = _content_matches_any(content, _FEAR_KEYWORDS)
        if fear_hits:
            snippet = content[:300]
            key = snippet[:80]
            if key not in seen["fears"]:
                seen["fears"].add(key)
                preferences["fears"].append({
                    "text": snippet,
                    "keywords": fear_hits[:5],
                    "timestamp": msg["timestamp"],
                })

        # Routines
        routine_hits = _content_matches_any(content, _ROUTINE_KEYWORDS)
        if routine_hits:
            snippet = content[:300]
            key = snippet[:80]
            if key not in seen["routines"]:
                seen["routines"].add(key)
                preferences["routines"].append({
                    "text": snippet,
                    "keywords": routine_hits[:5],
                    "timestamp": msg["timestamp"],
                })

        # Values
        value_hits = _content_matches_any(content, _VALUE_KEYWORDS)
        if value_hits:
            snippet = content[:300]
            key = snippet[:80]
            if key not in seen["values"]:
                seen["values"].add(key)
                preferences["values"].append({
                    "text": snippet,
                    "keywords": value_hits[:5],
                    "timestamp": msg["timestamp"],
                })

    # People: ranked by message volume per recipient
    recipient_counts: dict[str, int] = defaultdict(int)
    for msg in messages:
        if msg["recipient"]:
            recipient_counts[msg["recipient"]] += 1

    preferences["people"] = [
        {"name": recip, "message_count": count}
        for recip, count in sorted(
            recipient_counts.items(), key=lambda x: x[1], reverse=True
        )
    ]

    return preferences


def search_personality(conn: sqlite3.Connection, query: str,
                       limit: int = 10) -> list[dict]:
    """
    Search for personality-relevant information.

    This function supplements the main search engine.  When a query is about
    personality (food, communication style, fears, routines, general
    character), it performs targeted retrieval that generic keyword or vector
    search would miss.

    Strategy (post-MEMORIST-L0):

    1. Detect which personality aspect the query asks about.
    2. Extract the target entity from the query.
    3. Pull candidate messages via FTS5 using aspect-specific search terms.
    4. Score candidates using char-n-gram style vector similarity with
       persona scoping bias (5.0 for same-entity messages).
    5. Enrich results with pre-computed entity profile data.
    6. Return results ranked by score.

    Args:
        conn:  Open database connection.
        query: Natural-language personality question.
        limit: Maximum number of results.

    Returns:
        List of result dicts.  Each dict has ``content``, ``sender``,
        ``timestamp``, ``source`` (``"fts"``, ``"profile"``, or
        ``"style_vec"``), and ``aspect`` (the detected personality
        dimension).
    """
    lower_query = query.lower()
    results: list[dict] = []

    # ---- Step 1: detect the personality aspect ----
    detected_aspect = "personality"  # default fallback
    best_score = 0

    for aspect, config in PERSONALITY_ASPECTS.items():
        score = sum(1 for kw in config["keywords"] if kw in lower_query)
        if score > best_score:
            best_score = score
            detected_aspect = aspect

    # ---- Step 2: extract entity name from query ----
    all_senders = conn.execute(
        "SELECT DISTINCT sender FROM messages WHERE sender != ''"
    ).fetchall()
    sender_names = [r[0].lower() for r in all_senders]

    target_entity = None
    for name in sender_names:
        if name in lower_query:
            target_entity = name
            break

    # ---- Step 3: gather candidate messages via FTS ----
    aspect_config = PERSONALITY_ASPECTS.get(detected_aspect, {})
    fts_terms = aspect_config.get("fts_terms", [])
    candidate_msgs: list[dict] = []

    if fts_terms:
        fts_query = _build_safe_fts_query(fts_terms)
        candidate_msgs = _fts_search(conn, fts_query, limit=limit * 3)

    # Fallback: direct query word search
    if not candidate_msgs:
        query_words = [w for w in lower_query.split()
                       if len(w) > 3 and w not in {"what", "does", "like",
                                                    "kind", "person", "with",
                                                    "how", "about", "their",
                                                    "they", "have", "been"}]
        if query_words:
            fts_query = _build_safe_fts_query(query_words)
            candidate_msgs = _fts_search(conn, fts_query, limit=limit * 3)

    # If we have a target entity and no FTS results, get their messages directly
    if not candidate_msgs and target_entity:
        candidate_msgs = _get_messages_by_sender(conn, target_entity)[:limit * 3]

    # ---- Step 4: score candidates using style vectors ----
    _use_style_vec = False
    try:
        from truememory.personality_style_vec import (
            compute_style_vector,
            cosine_similarity,
            get_entity_style_vector,
        )
        _use_style_vec = True
    except ImportError:
        pass

    if _use_style_vec and candidate_msgs:
        profile_vec = None
        if target_entity:
            profile_vec = get_entity_style_vector(conn, target_entity)
            if profile_vec is None:
                for r in all_senders:
                    if r[0].lower() == target_entity:
                        profile_vec = get_entity_style_vector(conn, r[0])
                        if profile_vec:
                            break

        q_vec = compute_style_vector(query)

        scored: list[dict] = []
        for msg in candidate_msgs:
            sender = msg.get("sender", "")
            same_entity = (sender.lower() == target_entity) if target_entity else False
            base = 5.0 if same_entity else 0.0

            cv = compute_style_vector(msg.get("content", ""))
            sim_query = cosine_similarity(q_vec, cv)
            sim_profile = cosine_similarity(profile_vec, cv) if profile_vec else 0.0

            vec_score = base + 0.5 * sim_query + 0.5 * sim_profile

            scored.append({
                "id": msg.get("id"),
                "content": msg["content"],
                "sender": sender,
                "recipient": msg.get("recipient", ""),
                "timestamp": msg.get("timestamp", ""),
                "source": "style_vec",
                "aspect": detected_aspect,
                "score": vec_score,
            })

        scored.sort(key=lambda r: r["score"], reverse=True)
        results = scored[:limit]

    else:
        # Fallback: use FTS results directly (legacy behavior)
        if target_entity:
            candidate_msgs = [
                r for r in candidate_msgs
                if r["sender"].lower() == target_entity
            ]
        for r in candidate_msgs[:limit]:
            results.append({
                "content": r["content"],
                "sender": r["sender"],
                "recipient": r.get("recipient", ""),
                "timestamp": r.get("timestamp", ""),
                "source": "fts",
                "aspect": detected_aspect,
                "score": r.get("score", 0.0),
            })

    # ---- Step 5: add profile data ----
    if target_entity:
        profile = get_entity_profile(conn, target_entity)
        if profile:
            summary_parts = []

            if detected_aspect in ("personality", "communication"):
                style = profile.get("communication_style", {})
                if isinstance(style, str):
                    style = json.loads(style) if style else {}
                traits = profile.get("traits", [])
                if isinstance(traits, str):
                    traits = json.loads(traits) if traits else []
                summary_parts.append(
                    f"Communication style: {style.get('formality', 'unknown')}, "
                    f"avg message length: {style.get('avg_length', 0):.0f} chars, "
                    f"emoji: {'yes' if style.get('uses_emoji') else 'no'}, "
                    f"typical greeting: '{style.get('typical_greeting', 'N/A')}'"
                )
                if traits:
                    summary_parts.append(f"Traits: {', '.join(traits)}")

            if detected_aspect in ("personality", "relationships"):
                rels = profile.get("relationships", {})
                if isinstance(rels, str):
                    rels = json.loads(rels) if rels else {}
                if rels:
                    top_contacts = sorted(
                        rels.items(),
                        key=lambda x: x[1].get("message_count", 0)
                        if isinstance(x[1], dict) else 0,
                        reverse=True,
                    )[:5]
                    contact_strs = []
                    for name, info in top_contacts:
                        if isinstance(info, dict):
                            contact_strs.append(
                                f"{name} ({info.get('message_count', 0)} msgs, "
                                f"{info.get('primary_topic', 'general')})"
                            )
                    if contact_strs:
                        summary_parts.append(
                            f"Top contacts: {', '.join(contact_strs)}"
                        )

            topics = profile.get("topics", [])
            if isinstance(topics, str):
                topics = json.loads(topics) if topics else []
            if topics and detected_aspect == "personality":
                summary_parts.append(f"Key topics: {', '.join(topics[:5])}")

            for part in summary_parts:
                results.insert(0, {
                    "content": part,
                    "sender": target_entity,
                    "recipient": "",
                    "timestamp": "",
                    "source": "profile",
                    "aspect": detected_aspect,
                    "score": 1.0,
                })

    return results[:limit]


def update_entity_profile_incremental(
    conn: sqlite3.Connection, sender: str, message: str,
    recipient: str = "",
) -> None:
    """
    Incrementally update an entity profile from a single new message.

    Called from :meth:`TrueMemoryEngine.add` so that profiles build up as
    memories are added one at a time (MCP / production workflow).  The bulk
    :func:`build_entity_profiles` used by ``ingest()`` is unchanged.

    Args:
        conn:      Open database connection.
        sender:    Who said the message (maps to the entity profile).
        message:   The memory text.
        recipient: Who it was said to (optional).
    """
    if not sender or not message:
        return

    # ── Read existing profile (if any) ────────────────────────────────
    existing = get_entity_profile(conn, sender)

    if existing:
        msg_count = existing["message_count"] + 1
        old_topics = existing.get("topics") or []
        old_traits = existing.get("traits") or []
        old_style = existing.get("communication_style") or {}
        old_rels = existing.get("relationships") or {}
    else:
        msg_count = 1
        old_topics = []
        old_traits = []
        old_style = {}
        old_rels = {}

    # ── Communication style (rolling average) ─────────────────────────
    old_avg = old_style.get("avg_length", 0.0)
    old_count = msg_count - 1
    new_avg = ((old_avg * old_count) + len(message)) / msg_count

    uses_emoji = old_style.get("uses_emoji", False) or _detect_emoji(message)

    msg_dict = {"content": message}
    if msg_count == 1:
        formality = _assess_formality([msg_dict])
    else:
        formality = old_style.get("formality", "mixed")

    comm_style = {
        "avg_length": round(new_avg, 1),
        "uses_emoji": uses_emoji,
        "formality": formality,
        "typical_greeting": old_style.get("typical_greeting", ""),
    }

    # ── Topics (single-message extraction, threshold=1) ─────────────
    # _extract_topics requires >= 2 hits per cluster which is too strict
    # for a single message.  Inline a threshold-1 scan instead.
    lower = message.lower()
    words = set(lower.split())
    _topic_clusters = {
        "startup/business": {"startup", "company", "founded", "revenue",
                             "funding", "investor", "pitch", "valuation",
                             "mrr", "arr", "series", "seed", "raise"},
        "technology": {"code", "deploy", "database", "api", "backend",
                       "frontend", "server", "aws", "cloud", "repo",
                       "kubernetes", "docker", "github", "python", "go",
                       "rewrite", "migrate", "typescript", "javascript"},
        "health/fitness": {"gym", "run", "running", "workout", "health",
                           "marathon", "sleep", "meditation", "therapy",
                           "doctor", "weight"},
        "relationships": {"dating", "anniversary", "valentine", "proposal",
                          "ring", "wedding", "girlfriend", "boyfriend",
                          "partner"},
        "food/dining": {"restaurant", "coffee", "tacos", "sushi", "dinner",
                        "lunch", "breakfast", "eat", "cook", "bar"},
        "finance": {"salary", "savings", "money", "bank", "mortgage",
                    "house", "rent", "401k", "investment", "budget"},
        "career": {"job", "quit", "hired", "hiring", "employee",
                   "cofounder", "cto", "ceo", "role", "interview"},
        "pets": {"dog", "puppy", "vet", "pet", "shelter"},
        "entertainment": {"show", "movie", "netflix", "watch", "concert",
                          "festival", "book", "game"},
        "family": {"mom", "dad", "sister", "brother", "parents", "family",
                   "birthday"},
    }
    new_topics = [t for t, kws in _topic_clusters.items() if words & kws]
    topics = list(dict.fromkeys(old_topics + new_topics))

    # ── Traits (single-message extraction, threshold=1) ───────────────
    _trait_indicators = {
        "ambitious": {"goal", "growth", "scale", "expand", "raise",
                      "million", "revenue", "build", "launch", "grind"},
        "anxious": {"worried", "anxious", "stress", "panic", "scared",
                    "nervous", "overwhelmed"},
        "caring": {"love", "miss", "proud", "care", "support"},
        "analytical": {"data", "metrics", "numbers", "analysis",
                       "measure", "accuracy", "benchmark"},
        "health-conscious": {"gym", "workout", "run", "meditation",
                             "sleep", "diet", "healthy", "fitness"},
        "technical": {"code", "deploy", "api", "database", "server",
                      "algorithm", "architecture", "framework", "debug"},
        "entrepreneurial": {"startup", "founder", "company", "pitch",
                            "investor", "equity", "valuation"},
    }
    new_traits = [
        tr for tr, inds in _trait_indicators.items()
        if any(ind in lower for ind in inds)
    ]
    traits = list(dict.fromkeys(old_traits + new_traits))

    # ── Relationships ─────────────────────────────────────────────────
    if recipient:
        rel = old_rels.get(recipient, {"message_count": 0, "primary_topic": "general"})
        rel["message_count"] = rel.get("message_count", 0) + 1
        lower = message.lower()
        if any(w in lower for w in ("code", "deploy", "api", "database", "bug", "server")):
            rel["primary_topic"] = "technical"
        elif any(w in lower for w in ("worried", "anxious", "scared", "stress")):
            rel["primary_topic"] = "emotional"
        elif any(w in lower for w in ("dinner", "drinks", "hang out", "watch")):
            rel["primary_topic"] = "social"
        elif any(w in lower for w in ("revenue", "investor", "pricing", "customer")):
            rel["primary_topic"] = "business"
        old_rels[recipient] = rel

    # ── Store ─────────────────────────────────────────────────────────
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO entity_profiles
           (entity, message_count, traits, communication_style,
            topics, relationships, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            sender, msg_count,
            json.dumps(traits), json.dumps(comm_style),
            json.dumps(topics), json.dumps(old_rels), now,
        ),
    )
    conn.commit()


def get_entity_profile(conn: sqlite3.Connection,
                       entity: str) -> dict | None:
    """
    Get a pre-built entity profile from the database.

    Args:
        conn:   Open database connection.
        entity: Entity name to look up (case-insensitive).

    Returns:
        Profile dict with keys ``entity``, ``message_count``, ``traits``,
        ``communication_style``, ``topics``, ``relationships``,
        ``updated_at``.  Returns ``None`` if no profile exists for the
        entity.
    """
    row = conn.execute(
        "SELECT entity, message_count, traits, communication_style, "
        "       topics, relationships, updated_at "
        "FROM entity_profiles WHERE LOWER(entity) = LOWER(?)",
        (entity,),
    ).fetchone()

    if not row:
        return None

    return {
        "entity": row[0],
        "message_count": row[1],
        "traits": json.loads(row[2]) if row[2] else [],
        "communication_style": json.loads(row[3]) if row[3] else {},
        "topics": json.loads(row[4]) if row[4] else [],
        "relationships": json.loads(row[5]) if row[5] else {},
        "updated_at": row[6],
    }


def get_communication_pattern(conn: sqlite3.Connection,
                              entity1: str,
                              entity2: str) -> dict:
    """
    Analyze how *entity1* communicates specifically with *entity2*.

    Examines all messages from entity1 to entity2 and extracts:

    - **message_count**: total messages in this direction.
    - **avg_length**: average character length of messages.
    - **common_topics**: topic clusters detected in these messages.
    - **tone_indicators**: formality, emoji usage, greeting style.
    - **sample_messages**: a few representative messages (earliest, latest,
      longest).

    Args:
        conn:    Open database connection.
        entity1: The sender to analyze.
        entity2: The recipient to analyze communication toward.

    Returns:
        Dict with the analysis fields above.  Returns a dict with
        ``message_count: 0`` if no messages exist between the pair.
    """
    rows = conn.execute(
        "SELECT id, content, sender, recipient, timestamp, category, modality "
        "FROM messages "
        "WHERE LOWER(sender) = LOWER(?) AND LOWER(recipient) = LOWER(?) "
        "ORDER BY timestamp",
        (entity1, entity2),
    ).fetchall()

    messages = [
        {
            "id": r[0], "content": r[1], "sender": r[2],
            "recipient": r[3], "timestamp": r[4],
            "category": r[5], "modality": r[6],
        }
        for r in rows
    ]

    if not messages:
        return {
            "entity1": entity1,
            "entity2": entity2,
            "message_count": 0,
            "avg_length": 0.0,
            "common_topics": [],
            "tone_indicators": {},
            "sample_messages": [],
        }

    lengths = [len(m["content"]) for m in messages]
    avg_length = sum(lengths) / len(lengths)

    topics = _extract_topics(messages)
    formality = _assess_formality(messages)
    uses_emoji = any(_detect_emoji(m["content"]) for m in messages)
    greeting = _find_typical_greeting(messages)

    # Pick representative samples
    samples = []
    if messages:
        samples.append(messages[0])   # first message
        samples.append(messages[-1])  # most recent
        # longest message (most substantive)
        longest = max(messages, key=lambda m: len(m["content"]))
        if longest not in samples:
            samples.append(longest)

    return {
        "entity1": entity1,
        "entity2": entity2,
        "message_count": len(messages),
        "avg_length": round(avg_length, 1),
        "common_topics": topics,
        "tone_indicators": {
            "formality": formality,
            "uses_emoji": uses_emoji,
            "typical_greeting": greeting,
        },
        "sample_messages": [
            {"content": s["content"][:200], "timestamp": s["timestamp"]}
            for s in samples
        ],
    }


def resolve_entity(conn, name_query, context=""):
    """
    Multi-signal entity resolution: resolve ambiguous entity names.
    3-stage: FTS5 name match -> embedding similarity -> context disambiguation.

    Returns the best matching entity name from the database.
    """
    name_lower = name_query.lower().strip()

    # Stage 1: Direct name match
    row = conn.execute(
        "SELECT DISTINCT sender FROM messages WHERE LOWER(sender) = ?",
        (name_lower,)
    ).fetchone()
    if row:
        return row[0]

    # Stage 2: Partial/fuzzy match
    rows = conn.execute(
        "SELECT DISTINCT sender FROM messages WHERE sender != '' ORDER BY sender"
    ).fetchall()
    all_senders = [r[0] for r in rows]

    # Try substring matching
    partial_matches = [s for s in all_senders if name_lower in s.lower() or s.lower() in name_lower]
    if len(partial_matches) == 1:
        return partial_matches[0]

    # Stage 3: Context disambiguation
    if context and len(partial_matches) > 1:
        # Score each candidate by co-occurrence with context terms
        context_words = set(context.lower().split())
        best_match = None
        best_score = -1

        for candidate in partial_matches:
            # Get messages from this candidate
            msgs = conn.execute(
                "SELECT content FROM messages WHERE LOWER(sender) = LOWER(?) LIMIT 50",
                (candidate,)
            ).fetchall()

            # Count context word overlap
            score = 0
            for (content,) in msgs:
                content_lower = content.lower()
                score += sum(1 for w in context_words if w in content_lower and len(w) > 3)

            if score > best_score:
                best_score = score
                best_match = candidate

        if best_match:
            return best_match

    # If multiple partial matches but no context, return first
    if partial_matches:
        return partial_matches[0]

    return name_query  # Return original if no match found


def build_dunbar_hierarchy(conn, primary_entity=None):
    """
    Classify contacts by interaction frequency/recency into Dunbar layers:
    - intimate (5): closest, most frequent
    - close (15): regular interaction
    - friend (50): moderate interaction
    - acquaintance (150+): occasional

    Stores in entity_relationships table.
    """
    if primary_entity is None:
        return {}

    # Count messages per contact
    rows = conn.execute(
        """
        SELECT name, COUNT(*) as cnt, MAX(ts) as last_ts FROM (
            SELECT recipient as name, timestamp as ts FROM messages WHERE LOWER(sender) = LOWER(?) AND recipient != ''
            UNION ALL
            SELECT sender as name, timestamp as ts FROM messages WHERE LOWER(recipient) = LOWER(?) AND sender != ''
        ) GROUP BY name ORDER BY cnt DESC
        """,
        (primary_entity, primary_entity)
    ).fetchall()

    if not rows:
        return {}

    # Clear existing relationships for this entity
    conn.execute(
        "DELETE FROM entity_relationships WHERE LOWER(entity_a) = LOWER(?)",
        (primary_entity,)
    )

    max_count = rows[0][1] if rows else 1
    hierarchy = {}

    for name, count, last_ts in rows:
        # Normalize count to 0-1 range
        freq_score = count / max_count

        # Determine Dunbar layer
        if freq_score > 0.6:
            layer = "intimate"
        elif freq_score > 0.3:
            layer = "close"
        elif freq_score > 0.1:
            layer = "friend"
        else:
            layer = "acquaintance"

        hierarchy[name] = {
            "message_count": count,
            "last_interaction": last_ts or "",
            "dunbar_layer": layer,
            "strength": round(freq_score, 3),
        }

        conn.execute(
            "INSERT INTO entity_relationships "
            "(entity_a, entity_b, relationship_type, strength, dunbar_layer, last_interaction) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (primary_entity, name, "contact", round(freq_score, 3), layer, last_ts or "")
        )

    conn.commit()
    return hierarchy
