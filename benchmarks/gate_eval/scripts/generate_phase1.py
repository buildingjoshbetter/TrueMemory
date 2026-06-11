#!/usr/bin/env python3
"""Generate Phase 1 of GateLoCoMo benchmark: Conv 1, Sessions 1-6 (200 messages)."""

import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

NOISE_TYPES = {
    "N1": "filler", "N2": "greeting_farewell", "N3": "reaction",
    "N4": "meta_conversation", "N5": "echo",
}


def make_timestamps(start: str, gaps: list[int]) -> list[str]:
    t = datetime.fromisoformat(start)
    result = [t.isoformat()]
    for g in gaps:
        t += timedelta(seconds=g)
        result.append(t.isoformat())
    return result


def build(sess_num: int, date: str, start: str, gaps: list[int], msgs: list[tuple]) -> list[dict]:
    ts = make_timestamps(start, gaps)
    out = []
    for i, (spk, txt, cat, note) in enumerate(msgs):
        out.append({
            "id": f"conv1_s{sess_num:02d}_{i+1:03d}",
            "conversation_id": "conv1",
            "session": f"session_{sess_num}",
            "session_date": date,
            "speaker": spk,
            "recipient": "Jordan" if spk == "Alex" else "Alex",
            "content": txt,
            "timestamp": ts[i],
            "category": cat,
            "is_signal": not cat.startswith("N"),
            "noise_type": NOISE_TYPES.get(cat),
            "notes": note,
        })
    return out


# ═══════════════════════════════════════════════════════════════════════════
# SESSION 1: Weekend catch-up (30 msgs, 2025-11-03, Monday morning)
# Noise-heavy — mostly "how was your weekend" chat
# ═══════════════════════════════════════════════════════════════════════════
s1 = [
    ("Jordan", "yo", "N2", "Opening greeting"),
    ("Alex", "heyyy", "N2", "Greeting response"),
    ("Alex", "how was the hike", "N1", "Conversational question"),
    ("Jordan", "dude so good", "N1", "Vague positive"),
    ("Jordan", "point reyes was gorgeous", "B2", "Location fact in casual framing"),
    ("Alex", "nice", "N1", "Minimal acknowledgment"),
    ("Jordan", "did the tomales point trail. like 12 miles round trip", "S1", "Specific trail name and distance"),
    ("Alex", "damn", "N3", "Reaction to distance"),
    ("Alex", "that's a lot lol", "N1", "Filler response"),
    ("Jordan", "yeah I can barely walk today 😂", "N1", "Physical complaint, no retrievable info"),
    ("Alex", "haha", "N1", "Laughter filler"),
    ("Jordan", "worth it tho. the views from the bluff were insane", "N1", "Vague positive, no specific info"),
    ("Alex", "jealous. I literally did nothing all weekend", "N1", "Self-deprecating filler"),
    ("Jordan", "nothing?? lol", "N1", "Echo question"),
    ("Alex", "ok fine I tried that new ramen spot on Valencia", "B2", "Restaurant street mentioned casually"),
    ("Jordan", "oh which one", "N1", "Question filler"),
    ("Alex", "Mensho", "B1", "Restaurant name enables future reference"),
    ("Alex", "honestly might be the best ramen in the city", "S2", "Strong food preference"),
    ("Jordan", "oh damn high praise", "N3", "Reaction to opinion"),
    ("Alex", "the chashu was insane. and they do this truffle thing on the tonkotsu that's wild", "S2", "Specific food preference and dish detail"),
    ("Jordan", "ok I need to try it", "N5", "Echoing intent"),
    ("Alex", "yeah def go. go for lunch tho the line is shorter", "B1", "Practical advice enables future reference"),
    ("Jordan", "noted", "N1", "Acknowledgment filler"),
    ("Alex", "what else is new", "N4", "Conversation transition"),
    ("Jordan", "not much honestly", "N1", "Filler"),
    ("Jordan", "oh I started watching severance finally", "B2", "Casual fact about TV watching"),
    ("Alex", "omg WAIT it's so good right", "N3", "Excited reaction"),
    ("Jordan", "I'm 4 eps in lol", "N1", "Trivial progress detail"),
    ("Jordan", "anyway gotta run. meeting in 5", "N2", "Farewell"),
    ("Alex", "later ✌️", "N2", "Farewell"),
]
g1 = [30, 45, 20, 30, 60, 15, 45, 20, 30, 60, 120, 45, 30, 90, 30, 60, 20, 45, 30, 180, 60, 30, 45, 20, 120, 30, 60, 45, 30]


# ═══════════════════════════════════════════════════════════════════════════
# SESSION 2: Stripe frustration + apartment hunting (35 msgs, 2025-11-07, Friday evening)
# Mix of signal and noise — Alex venting about work, Jordan apartment hunting
# ═══════════════════════════════════════════════════════════════════════════
s2 = [
    ("Alex", "ok I need to vent", "N4", "Meta-conversation opener"),
    ("Jordan", "oh no what happened", "N3", "Concerned reaction"),
    ("Alex", "so remember that reorg from like three weeks ago", "B3", "Temporal reference to past event"),
    ("Jordan", "yeah", "N1", "Acknowledgment"),
    ("Alex", "my team got completely split up. they moved me to payments core", "S1", "Team change and new team name"),
    ("Alex", "but the real issue is my new manager. this PM named Derek", "S1", "Manager name and role type"),
    ("Jordan", "uh oh", "N3", "Reaction"),
    ("Alex", "zero engineering background. came from McKinsey or something", "S1", "Manager background detail"),
    ("Jordan", "yikes", "N3", "Reaction"),
    ("Alex", "and today he literally told the team we need to 'be more agile' and 'move with urgency'", "S1", "Workplace event detail with quote"),
    ("Jordan", "lmao the consulting brain", "N1", "Commentary filler"),
    ("Alex", "honestly the culture at Stripe has just gotten so corporate. it's not the same company I joined", "S2", "Opinion about employer culture, names Stripe"),
    ("Alex", "I joined in 2022 right after their layoffs and even then it was different", "S1", "Employment start year and context"),
    ("Jordan", "yeah I remember you saying that", "N5", "Echo acknowledgment"),
    ("Alex", "I'm seriously thinking about leaving", "S3", "Decision statement about leaving job"),
    ("Jordan", "like actively looking?", "N1", "Follow-up question"),
    ("Alex", "yeah. updated LinkedIn last week, already got a few recruiter messages", "S1", "Job search activity status"),
    ("Jordan", "oh nice", "N1", "Acknowledgment"),
    ("Alex", "looking at ML infrastructure roles mostly. maybe research engineering", "S2", "Job role preference"),
    ("Alex", "honestly anything in AI. I'm so done with fintech", "S2", "Field preference and anti-preference"),
    ("Jordan", "I can see that", "N5", "Echo agreement"),
    ("Alex", "there's a couple interesting ones. Anthropic has a role that looks perfect actually", "S1", "Specific target company mentioned"),
    ("Jordan", "oh Anthropic is sick. my friend Lisa works there actually", "B2", "Casual fact: friend name and employer"),
    ("Alex", "wait really?? what does she do", "N3", "Interested reaction"),
    ("Jordan", "she's on the research team I think. loves it there", "B1", "Friend role, enables future intro reference"),
    ("Alex", "hmm interesting. I might ask you for an intro", "B1", "Potential future action, enables callback"),
    ("Jordan", "yeah for sure just lmk", "N1", "Casual agreement"),
    ("Alex", "anyway enough about my work drama. how's the apartment hunt", "N4", "Meta-conversation transition"),
    ("Jordan", "ugh brutal", "N1", "Vague negative"),
    ("Jordan", "I've been looking mostly in Hayes Valley and the Mission", "S1", "Target neighborhoods for apartment"),
    ("Jordan", "but like a decent 1br in Hayes Valley starts at 3k minimum", "S1", "Market price data point"),
    ("Alex", "god that's insane", "N3", "Reaction to price"),
    ("Jordan", "yeah. I'm trying to stay under 2800 if possible", "S1", "Personal budget constraint"),
    ("Jordan", "might have to expand out to the Richmond or Sunset", "B1", "Potential expansion neighborhoods"),
    ("Alex", "honestly the Richmond is underrated", "S2", "Opinion about neighborhood"),
]
g2 = [45, 60, 20, 120, 30, 60, 45, 20, 90, 30, 45, 30, 20, 60, 30, 120, 45, 30, 20, 60, 45, 30, 180, 30, 60, 30, 45, 60, 30, 20, 120, 30, 45, 30]


# ═══════════════════════════════════════════════════════════════════════════
# SESSION 3: Bad day, quick check-in (25 msgs, 2025-11-12, Wednesday night)
# HIGH NOISE — Alex venting, short messages
# ═══════════════════════════════════════════════════════════════════════════
s3 = [
    ("Alex", "ughhhh", "N1", "Frustration filler"),
    ("Jordan", "uh oh", "N3", "Concerned reaction"),
    ("Alex", "worst. day.", "N1", "Complaint filler"),
    ("Jordan", "what happened", "N1", "Question filler"),
    ("Alex", "Derek called an all-hands to basically tell the eng team we're all underperforming", "S1", "Work event with manager name callback"),
    ("Jordan", "oh god", "N3", "Reaction"),
    ("Alex", "in front of like 40 people", "N1", "Elaboration, no new retrievable facts"),
    ("Jordan", "that's so unprofessional", "N3", "Reaction"),
    ("Alex", "RIGHT", "N5", "Agreement echo"),
    ("Alex", "I'm so done with this place honestly", "S5", "Emotional disclosure about job"),
    ("Jordan", "I'm sorry dude", "N1", "Sympathy filler"),
    ("Alex", "I literally had Blind open during the all-hands lol", "B2", "Casual fact: browsing job site at work"),
    ("Jordan", "haha ok that's kinda iconic", "N1", "Filler response"),
    ("Alex", "half my old team has left already. like 4 people since the reorg", "S1", "Team attrition data point"),
    ("Jordan", "damn", "N3", "Reaction"),
    ("Alex", "it's been like this since the reorg in October honestly", "B3", "Temporal reference to reorg month"),
    ("Jordan", "do you think it'll get better", "N1", "Question filler"),
    ("Alex", "honestly no", "N1", "Short answer filler"),
    ("Jordan", "yeah", "N1", "Agreement filler"),
    ("Alex", "whatever. how was your day", "N4", "Conversation transition"),
    ("Jordan", "fine. nothing crazy. hey you wanna grab a drink this week? you seem like you need one", "B1", "Suggestion with emotional context"),
    ("Alex", "god yes please", "N1", "Agreement"),
    ("Jordan", "Thursday? that bar on 16th", "B3", "Temporal plan + location reference"),
    ("Alex", "works for me 👍", "N1", "Acceptance filler"),
    ("Jordan", "cool. hang in there ❤️", "N2", "Farewell with encouragement"),
]
g3 = [30, 20, 45, 120, 30, 20, 45, 20, 30, 60, 45, 30, 60, 30, 45, 30, 20, 30, 120, 60, 45, 30, 45, 30]


# ═══════════════════════════════════════════════════════════════════════════
# SESSION 4: First date recap + apartment found (40 msgs, 2025-11-17, Monday lunch)
# MORE SIGNAL — details about Priya, Jordan's new apartment
# ═══════════════════════════════════════════════════════════════════════════
s4 = [
    ("Jordan", "ok spill", "N1", "Conversational prompt"),
    ("Alex", "about what 👀", "N1", "Playful dodge"),
    ("Jordan", "the DATE. you went out Saturday right??", "B3", "Temporal reference to date night"),
    ("Alex", "haha ok ok so", "N1", "Transitional filler"),
    ("Alex", "her name is Priya. she's a product designer at Figma", "S1", "Name, job title, and company"),
    ("Jordan", "oh nice", "N1", "Acknowledgment"),
    ("Alex", "we matched on Hinge like two weeks ago and been texting nonstop", "B2", "How they met + casual temporal fact"),
    ("Alex", "went to this wine bar in Noe Valley called Wildhawk", "S1", "Date venue name and neighborhood"),
    ("Jordan", "oh I've been there actually. they have that little garden patio right?", "B1", "Venue detail enables future reference"),
    ("Alex", "yes! that's where we sat actually. it was perfect", "N1", "Elaboration filler"),
    ("Alex", "honestly we ended up talking for like three hours", "B3", "Duration marker"),
    ("Jordan", "oh damn", "N3", "Reaction"),
    ("Alex", "I haven't clicked with someone like this in so long honestly", "S5", "Emotional disclosure about romantic connection"),
    ("Alex", "she's really into rock climbing", "S1", "Date's primary hobby"),
    ("Alex", "and she lived in Tokyo for two years after college which is so cool", "B2", "Casual biographical fact about date"),
    ("Jordan", "oh wow that's awesome", "N3", "Reaction"),
    ("Alex", "yeah I've been wanting to try climbing honestly. she's gonna take me to her gym, Dogpatch Boulders", "S2", "Personal interest + specific gym name"),
    ("Jordan", "haha watch you become obsessed", "N1", "Commentary filler"),
    ("Alex", "probably lol", "N1", "Agreement filler"),
    ("Alex", "oh and she's really into indie music. Japanese Breakfast, Mitski, that kind of stuff", "S1", "Music preferences with specific artist names"),
    ("Jordan", "solid taste", "N5", "Approval echo"),
    ("Alex", "right?? we're going out again this Thursday", "S3", "Second date commitment with timing"),
    ("Alex", "she picked this sushi spot in Japantown. Sushi Sato", "S1", "Second date restaurant and neighborhood"),
    ("Jordan", "oh nice I've heard good things", "N1", "Filler"),
    ("Alex", "I'm excited honestly", "S5", "Emotional disclosure about anticipation"),
    ("Alex", "ok but enough about me. APARTMENT. what happened with the viewings", "N4", "Conversation transition"),
    ("Jordan", "OK SO", "N1", "Dramatic opener filler"),
    ("Jordan", "I think I found one", "S4", "Life event: found apartment"),
    ("Alex", "wait WHAT", "N3", "Excited reaction"),
    ("Jordan", "yeah! a 1br on Balboa St, between 6th and 7th Ave. inner Richmond", "S1", "Full address details of new apartment"),
    ("Alex", "omg", "N3", "Reaction"),
    ("Jordan", "$2,400 a month", "S1", "Monthly rent amount"),
    ("Alex", "dude that's honestly so good for a 1br", "N5", "Validating echo"),
    ("Jordan", "I know right? and it's a corner unit so it gets a ton of natural light", "S1", "Apartment feature: corner unit, natural light"),
    ("Jordan", "hardwood floors, laundry in the building, and there's a little balcony off the bedroom", "S1", "Multiple apartment features: floors, laundry, balcony"),
    ("Jordan", "I can move in December 1st", "S1", "Move-in date"),
    ("Alex", "that's amazing congrats!! 🎉", "N3", "Celebration reaction"),
    ("Jordan", "thanks I'm so relieved. been looking since September honestly", "S5", "Emotional relief + apartment search start month"),
    ("Alex", "was that like two and a half months of searching?", "B3", "Temporal duration question"),
    ("Jordan", "yeah basically. saw probably 15 places before this one lol", "B2", "Casual fact about total places viewed"),
]
g4 = [30, 20, 60, 30, 45, 30, 20, 90, 30, 45, 30, 20, 30, 20, 60, 45, 30, 20, 120, 30, 20, 30, 60, 30, 45, 180, 30, 60, 30, 20, 45, 30, 20, 30, 30, 60, 45, 30, 60]


# ═══════════════════════════════════════════════════════════════════════════
# SESSION 5: Anthropic interview + landlord issue (35 msgs, 2025-11-24, Monday evening)
# Mix — interview excitement, landlord frustration
# ═══════════════════════════════════════════════════════════════════════════
s5 = [
    ("Alex", "ok so guess what", "N4", "Meta opener"),
    ("Jordan", "what", "N1", "Response"),
    ("Alex", "I have an interview at Anthropic next Tuesday", "S1", "Interview fact: company and day"),
    ("Jordan", "WAIT", "N3", "Reaction"),
    ("Jordan", "are you serious", "N3", "Reaction"),
    ("Alex", "dead serious. recruiter reached out last week and it moved super fast", "B3", "Temporal context about recruitment speed"),
    ("Jordan", "dude that's huge", "N3", "Reaction"),
    ("Alex", "I know I'm kind of freaking out honestly", "S5", "Emotional disclosure: interview anxiety"),
    ("Alex", "it's for a senior ML infrastructure role on the core platform team", "S1", "Role title and team name"),
    ("Jordan", "wait you should talk to Lisa. my friend who works there remember?", "B1", "Callback to session 2 friend connection"),
    ("Alex", "oh right!! yeah can you intro me?", "S3", "Request for professional introduction"),
    ("Jordan", "already texted her. she said she'd be happy to chat", "S3", "Action taken: arranged intro with Lisa"),
    ("Alex", "you're the best", "N1", "Gratitude filler"),
    ("Alex", "it's a full day thing. five interview rounds", "S1", "Interview format: full day, 5 rounds"),
    ("Jordan", "oh damn what are the rounds", "N1", "Question"),
    ("Alex", "system design, two coding rounds, behavioral, and then a team lunch with the hiring manager", "S1", "Specific interview round breakdown"),
    ("Jordan", "that's intense", "N3", "Reaction"),
    ("Alex", "yeah I've been doing leetcode every night this week 😭", "B2", "Casual fact: nightly leetcode prep"),
    ("Jordan", "oh god leetcode", "N3", "Commiserating reaction"),
    ("Alex", "it's miserable but I'm doing like 4 hours a day at this point lol", "B2", "Casual fact: daily prep intensity"),
    ("Alex", "Lisa actually sent me some tips which was really helpful", "S1", "Prep fact: Lisa provided interview tips"),
    ("Jordan", "oh nice what'd she say", "N1", "Question"),
    ("Alex", "she said the system design round is the hardest and they really care about scale and reliability thinking", "S1", "Insider interview insight"),
    ("Jordan", "makes sense for an AI company", "N5", "Echo agreement"),
    ("Alex", "yeah. anyway how's the new place??", "N4", "Topic transition"),
    ("Jordan", "it's great honestly. I love the neighborhood. so many good restaurants on Clement St", "S2", "Neighborhood preference + specific street"),
    ("Jordan", "only issue is the heater's broken", "S1", "Apartment problem: broken heater"),
    ("Alex", "wait what? you just moved in", "N3", "Reaction"),
    ("Jordan", "yeah it's been out since day one. I've emailed the landlord three times now", "S1", "Problem duration and contact attempts"),
    ("Alex", "it's getting cold too. that's not ok", "N1", "Commentary"),
    ("Jordan", "the building is from like the 1920s so everything is ancient", "S1", "Building age fact"),
    ("Jordan", "honestly I might have to file a complaint with the city if he doesn't fix it this week", "S3", "Potential escalation decision"),
    ("Alex", "you should honestly. you're paying 2400 a month for that place", "B1", "References prior rent amount as support"),
    ("Jordan", "yeah you're right", "N5", "Agreement echo"),
    ("Alex", "ok gotta go study. wish me luck Tuesday 🙏", "B3", "Temporal reference to interview + farewell"),
]
g5 = [30, 120, 30, 20, 90, 45, 30, 20, 60, 30, 45, 30, 20, 60, 120, 30, 45, 30, 20, 30, 60, 30, 60, 45, 90, 30, 60, 45, 30, 60, 45, 30, 20, 60]


# ═══════════════════════════════════════════════════════════════════════════
# SESSION 6: Got the Anthropic offer! (35 msgs, 2025-12-02, Tuesday afternoon)
# HIGH SIGNAL in second half — salary, start date, decisions
# ═══════════════════════════════════════════════════════════════════════════
s6 = [
    ("Alex", "JORDAN", "N1", "Excitement opener"),
    ("Alex", "I GOT IT", "S4", "Life event: received job offer"),
    ("Jordan", "wait WHAT", "N3", "Reaction"),
    ("Jordan", "THE ANTHROPIC JOB??", "B1", "Clarifying context, links to prior sessions"),
    ("Alex", "YES", "N1", "Confirmation filler"),
    ("Jordan", "DUDE", "N3", "Reaction"),
    ("Alex", "I know I'm literally shaking rn", "S5", "Emotional state disclosure"),
    ("Jordan", "congrats oh my god 🎉🎉", "N3", "Celebration reaction"),
    ("Alex", "the recruiter called like an hour ago with the formal offer", "B3", "Temporal detail about when offer arrived"),
    ("Alex", "ok so details. title is Senior ML Infrastructure Engineer", "S1", "Official job title"),
    ("Alex", "base is $245k", "S1", "Base salary amount"),
    ("Jordan", "holy shit", "N3", "Reaction to salary"),
    ("Alex", "plus RSUs over 4 years. total comp is like $380k all in", "S1", "Full compensation: RSUs + total comp"),
    ("Alex", "that's almost a 40% raise from what I'm making at Stripe", "S1", "Comparative compensation fact"),
    ("Jordan", "you deserve it honestly. especially after the Derek saga", "B1", "Callback to manager issues"),
    ("Alex", "thanks man. seriously", "N1", "Gratitude filler"),
    ("Alex", "start date is January 6th", "S1", "Start date at new company"),
    ("Alex", "so I gotta give notice at Stripe basically next week", "S3", "Decision: giving notice timeline"),
    ("Jordan", "wow that's coming up fast", "N5", "Echo reaction"),
    ("Alex", "yeah I'm telling Derek on Monday lol", "S3", "Specific resignation plan"),
    ("Jordan", "lol bet that'll feel good", "N1", "Commentary"),
    ("Alex", "I'm not gonna lie... yeah kinda", "B2", "Casual emotional admission"),
    ("Alex", "their office is on Mission St near Embarcadero. like 20 min bike from my place", "S1", "Office location and commute time"),
    ("Jordan", "oh nice. isn't that right by the Ferry Building?", "B1", "Location context question"),
    ("Alex", "yeah pretty much! they have this rooftop deck apparently which is cool", "S1", "Office feature detail"),
    ("Alex", "honestly I haven't been this excited about anything in a long time", "S5", "Emotional reflection on career change"),
    ("Jordan", "I can tell. this is so your vibe", "N5", "Affirming echo"),
    ("Alex", "haha Priya was actually the first person I called. before my parents even", "B2", "Casual relationship depth indicator"),
    ("Jordan", "oh wow things are really serious with her huh", "N1", "Commentary question"),
    ("Alex", "yeah we've been seeing each other every weekend. it's been like a month and a half now", "S1", "Relationship status and duration"),
    ("Jordan", "I love that for you. new job, new girl, whole new chapter", "N1", "Supportive filler"),
    ("Alex", "ok that's cheesy but yeah honestly this fall has been wild", "B3", "Temporal seasonal reflection"),
    ("Jordan", "we need to celebrate. Friday? I'm buying", "B3", "Temporal celebration plan"),
    ("Alex", "no way I'm buying. consider it a preview of my first paycheck lol", "S3", "Commitment + playful salary reference"),
    ("Jordan", "haha deal. proud of you dude ❤️", "N2", "Closing farewell"),
]
g6 = [10, 30, 20, 20, 30, 30, 30, 60, 20, 30, 45, 30, 20, 90, 30, 20, 30, 30, 45, 30, 60, 30, 120, 30, 45, 30, 90, 30, 60, 30, 120, 45, 30, 60]


# ═══════════════════════════════════════════════════════════════════════════
# BUILD AND WRITE
# ═══════════════════════════════════════════════════════════════════════════
sessions = [
    (1, "2025-11-03", "2025-11-03T10:15:00", g1, s1),
    (2, "2025-11-07", "2025-11-07T18:30:00", g2, s2),
    (3, "2025-11-12", "2025-11-12T21:45:00", g3, s3),
    (4, "2025-11-17", "2025-11-17T12:30:00", g4, s4),
    (5, "2025-11-24", "2025-11-24T19:00:00", g5, s5),
    (6, "2025-12-02", "2025-12-02T16:15:00", g6, s6),
]

all_messages = []
for num, date, start, gaps, msgs in sessions:
    assert len(gaps) == len(msgs) - 1, \
        f"Session {num}: {len(gaps)} gaps but {len(msgs)} msgs (need {len(msgs)-1} gaps)"
    all_messages.extend(build(num, date, start, gaps, msgs))

# Compute statistics
cat_counts = Counter(m["category"] for m in all_messages)
noise_total = sum(v for k, v in cat_counts.items() if k.startswith("N"))
signal_total = sum(v for k, v in cat_counts.items() if k.startswith("S"))
border_total = sum(v for k, v in cat_counts.items() if k.startswith("B"))
speaker_counts = Counter(m["speaker"] for m in all_messages)

output = {
    "conversation_id": "conv1",
    "part": 1,
    "sessions_covered": [1, 2, 3, 4, 5, 6],
    "speakers": {
        "Alex": "Software engineer at Stripe, exploring ML/AI roles",
        "Jordan": "Close friend, apartment hunting in San Francisco",
    },
    "date_range": {"start": "2025-11-03", "end": "2025-12-02"},
    "message_count": len(all_messages),
    "category_distribution": {
        "noise": {k: v for k, v in sorted(cat_counts.items()) if k.startswith("N")},
        "signal": {k: v for k, v in sorted(cat_counts.items()) if k.startswith("S")},
        "borderline": {k: v for k, v in sorted(cat_counts.items()) if k.startswith("B")},
        "totals": {
            "noise": noise_total,
            "signal": signal_total,
            "borderline": border_total,
            "noise_pct": round(noise_total / len(all_messages) * 100, 1),
            "signal_pct": round(signal_total / len(all_messages) * 100, 1),
            "borderline_pct": round(border_total / len(all_messages) * 100, 1),
        },
    },
    "speaker_balance": {
        k: {"count": v, "pct": f"{v / len(all_messages) * 100:.1f}%"}
        for k, v in speaker_counts.items()
    },
    "messages": all_messages,
}

out_path = Path("benchmarks/gate_eval/datasets/gate_benchmark_conv1_part1.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"✓ Wrote {len(all_messages)} messages to {out_path}")
print(f"  Noise:      {noise_total} ({noise_total/len(all_messages)*100:.1f}%)")
print(f"  Signal:     {signal_total} ({signal_total/len(all_messages)*100:.1f}%)")
print(f"  Borderline: {border_total} ({border_total/len(all_messages)*100:.1f}%)")
print(f"  Speakers:   {dict(speaker_counts)}")
print(f"  Categories: {dict(sorted(cat_counts.items()))}")
print()

# Per-session breakdown
for num, _, _, _, msgs in sessions:
    sess_msgs = [m for m in all_messages if m["session"] == f"session_{num}"]
    sc = Counter(m["category"] for m in sess_msgs)
    n = sum(v for k, v in sc.items() if k.startswith("N"))
    s = sum(v for k, v in sc.items() if k.startswith("S"))
    b = sum(v for k, v in sc.items() if k.startswith("B"))
    sp = Counter(m["speaker"] for m in sess_msgs)
    print(f"  Session {num}: {len(sess_msgs)} msgs | N={n} S={s} B={b} | {dict(sp)}")
