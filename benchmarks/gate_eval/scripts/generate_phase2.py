#!/usr/bin/env python3
"""Generate Phase 2 of GateLoCoMo benchmark: Conv 1, Sessions 7-12 (200 messages)."""

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
# SESSION 7: Alex's first week at Anthropic (30 msgs, 2026-01-08)
# Excited but overwhelmed. Jordan offers advice.
# ═══════════════════════════════════════════════════════════════════════════
s7 = [
    ("Alex", "update from inside the building", "N4", "Meta-conversation opener"),
    ("Jordan", "yesss how's the first week??", "N3", "Excited reaction"),
    ("Alex", "dude it's so different from Stripe", "S2", "Culture comparison opinion"),
    ("Alex", "like everyone is genuinely nice?? is that allowed at a tech company", "S2", "Culture assessment"),
    ("Jordan", "lol the bar was underground at Stripe huh", "N1", "Commentary filler"),
    ("Alex", "haha ok fair", "N1", "Agreement filler"),
    ("Alex", "my team is 8 people. really tight-knit", "S1", "Team size fact"),
    ("Alex", "manager is this woman named Ava. she was at DeepMind before this", "S1", "Manager name and background"),
    ("Jordan", "oh sick", "N1", "Minimal reaction"),
    ("Alex", "she's great. already asked me what I actually want to work on in my first 1:1", "S2", "Management style assessment"),
    ("Jordan", "that's how it should be honestly", "N5", "Agreement echo"),
    ("Alex", "right?? Derek would literally never lol", "B1", "Callback to Stripe manager from Part 1"),
    ("Alex", "first project is building their model evaluation infrastructure. benchmarks and monitoring", "S1", "First project description"),
    ("Jordan", "oh that's perfect for you", "N1", "Supportive filler"),
    ("Alex", "yeah it's exactly what I wanted. I'm just drinking from a firehose rn", "S5", "Emotional: aligned but overwhelmed"),
    ("Jordan", "that's normal tho. remember my first month at Dropbox? I was completely useless", "B2", "Casual fact: Jordan works at Dropbox"),
    ("Alex", "haha true true", "N1", "Agreement filler"),
    ("Alex", "the office is really nice btw. remember I said it was on Mission near Embarcadero?", "B1", "Callback to office location from Part 1"),
    ("Jordan", "yeah", "N1", "Acknowledgment"),
    ("Alex", "there's this rooftop deck and they do all-team lunches on Wednesdays", "S1", "Office perks: rooftop deck, Wednesday lunches"),
    ("Jordan", "free food every Wednesday? I'm jealous", "N5", "Echo with envy"),
    ("Alex", "lol yep. also got a MacBook Pro and a standing desk on day one. Stripe made me wait two weeks for my laptop", "B2", "Casual comparison fact"),
    ("Jordan", "haha already comparing everything to Stripe", "N1", "Commentary filler"),
    ("Alex", "obviously. oh btw how's the apartment? heater ever get fixed?", "N4", "Topic transition with callback"),
    ("Jordan", "YES finally. landlord sent someone last week", "S1", "Heater fixed + temporal context"),
    ("Alex", "oh thank god. only took what like a month?", "B3", "Temporal duration reference"),
    ("Jordan", "basically lol. but yeah the place is great now honestly. I'm really settling in", "S5", "Emotional: content with apartment"),
    ("Alex", "good you deserve it after that whole search", "N1", "Supportive filler"),
    ("Alex", "ok gotta run, team standup", "N2", "Farewell"),
    ("Jordan", "go crush it 💪", "N2", "Farewell"),
]
g7 = [45, 30, 20, 60, 30, 45, 30, 20, 90, 30, 60, 30, 20, 45, 120, 30, 60, 30, 20, 45, 30, 60, 120, 30, 45, 20, 30, 45, 30]


# ═══════════════════════════════════════════════════════════════════════════
# SESSION 8: Jordan's housewarming party planning (35 msgs, 2026-01-12)
# Guest list, food, logistics
# ═══════════════════════════════════════════════════════════════════════════
s8 = [
    ("Jordan", "ok I need your help with something", "N4", "Meta opener"),
    ("Alex", "what's up", "N1", "Response filler"),
    ("Jordan", "housewarming party. I'm finally doing it", "S4", "Life event: hosting housewarming"),
    ("Alex", "YESSS finally. when", "N3", "Excited reaction"),
    ("Jordan", "thinking Saturday the 25th. two weeks from now", "S1", "Party date"),
    ("Alex", "I'm there", "S3", "Commitment to attend"),
    ("Jordan", "ok so I need to figure out food and drinks", "N4", "Planning meta"),
    ("Jordan", "I was thinking like a low-key thing. maybe 15-20 people?", "S1", "Expected guest count"),
    ("Alex", "that sounds perfect for your place", "N1", "Supportive filler"),
    ("Jordan", "yeah exactly. the living room is actually pretty spacious for a 1br", "B2", "Casual fact about apartment layout"),
    ("Jordan", "for food I was thinking like a taco bar situation", "S1", "Food plan: taco bar"),
    ("Alex", "ohhh yes", "N3", "Enthusiastic reaction"),
    ("Jordan", "get the tortillas from that place on Clement. Viva something?", "B2", "Casual restaurant name reference"),
    ("Alex", "YES do that. their tortillas are so good", "N5", "Echo enthusiasm"),
    ("Jordan", "ok and then I'll do carnitas, chicken tinga, guac, pico, the works", "S1", "Specific menu items"),
    ("Alex", "ok I'm getting hungry just thinking about this", "N1", "Commentary filler"),
    ("Jordan", "haha. for drinks I was thinking BYOB plus I'll grab a keg of something local", "S1", "Drinks plan: BYOB + keg"),
    ("Alex", "a keg?? who are you lol", "N3", "Amused reaction"),
    ("Jordan", "I know I know but it's actually cheaper than buying cases", "N1", "Justification filler"),
    ("Jordan", "oh and I want one of those portable fire pit things for the balcony", "S1", "Party item: fire pit for balcony"),
    ("Alex", "oh that's a great call. it's been freezing", "N1", "Supportive filler"),
    ("Jordan", "right? ok who should I invite besides obviously you", "N4", "Guest list meta"),
    ("Alex", "can I bring Priya?", "S3", "Decision to bring partner to social event"),
    ("Jordan", "YES please I've been dying to meet her", "N3", "Excited reaction"),
    ("Alex", "haha perfect. she's so excited too", "S5", "Emotional: Priya's excitement about friend milestone"),
    ("Jordan", "what about the college crew? Mike, Sarah, Tina?", "S1", "Friend names from college group"),
    ("Alex", "oh for sure. Mike is in town I think", "B2", "Casual fact: Mike's location"),
    ("Jordan", "nice. and I'll invite some work people. my coworker Anil is really fun", "S1", "Coworker name: Anil"),
    ("Alex", "more the merrier", "N1", "Filler"),
    ("Jordan", "I also want to get a decent speaker. my bluetooth one is tiny", "S1", "Need for speaker"),
    ("Alex", "oh I have a JBL Charge you can borrow. it's loud as hell", "B1", "Context: Alex owns JBL speaker"),
    ("Alex", "I'll bring it", "S3", "Commitment to lend speaker"),
    ("Jordan", "oh perfect that saves me like $80", "N1", "Filler"),
    ("Jordan", "ok this is happening. I'm actually hyped", "N1", "Excitement filler"),
    ("Alex", "same 🙌", "N1", "Agreement filler"),
]
g8 = [30, 60, 30, 45, 20, 30, 60, 30, 120, 30, 45, 20, 30, 60, 30, 45, 30, 20, 120, 30, 60, 45, 30, 20, 60, 30, 45, 30, 20, 120, 30, 60, 30, 45]


# ═══════════════════════════════════════════════════════════════════════════
# SESSION 9: Quick catch-up, Priya relationship serious (25 msgs, 2026-01-16)
# Alex's relationship milestone. Short session.
# ═══════════════════════════════════════════════════════════════════════════
s9 = [
    ("Alex", "hey you free to chat for a sec", "N2", "Opening greeting"),
    ("Jordan", "yeah what's up", "N1", "Response filler"),
    ("Alex", "so Priya met my parents this weekend", "S4", "Life event: partner met parents"),
    ("Jordan", "oh wow that's a big step", "N3", "Reaction"),
    ("Alex", "yeah it kind of just happened. they were visiting from Sacramento and we all got dinner", "B3", "Temporal context + parents' location"),
    ("Jordan", "how'd it go", "N1", "Question filler"),
    ("Alex", "really well actually. my mom loved her. they bonded over cooking stuff", "S1", "Family interaction detail: mom + Priya bonded"),
    ("Jordan", "aww that's sweet. my mom would never lol she still asks if I'm eating enough every single call", "B2", "Casual fact: Jordan's mom's behavior"),
    ("Alex", "haha classic mom stuff. but yeah my dad kept asking Priya about Figma. classic dad trying to be cool about tech", "B2", "Casual fact: Alex's dad's tech interest"),
    ("Jordan", "lol that's cute", "N1", "Filler"),
    ("Alex", "yeah. I think they could tell I'm genuinely happy", "S5", "Emotional disclosure about relationship"),
    ("Jordan", "are you?", "N1", "Follow-up question"),
    ("Alex", "honestly yeah. like really. it's been what, almost 3 months now?", "S5", "Emotional + temporal: relationship duration"),
    ("Jordan", "that's awesome dude", "N1", "Supportive filler"),
    ("Alex", "thanks. I don't wanna jinx it but it's really good", "N1", "Filler"),
    ("Jordan", "you won't jinx it. you guys are great together", "N1", "Reassurance filler"),
    ("Alex", "❤️", "N1", "Emoji filler"),
    ("Alex", "how's work btw. anything new at Dropbox", "N4", "Topic transition, confirms employer"),
    ("Jordan", "meh same old. we're doing this big migration thing that's kind of a nightmare", "S1", "Work project fact"),
    ("Alex", "oof", "N3", "Reaction"),
    ("Jordan", "yeah it's fine. at least it's not Derek-level bad lol", "B1", "Callback to Derek as reference point"),
    ("Alex", "haha literally nothing is Derek-level bad", "N1", "Commentary filler"),
    ("Jordan", "lol exactly. ok I gotta hop to a meeting", "N2", "Farewell"),
    ("Alex", "same actually. standup in 2", "N2", "Farewell"),
    ("Jordan", "later ✌️", "N2", "Farewell"),
]
g9 = [30, 60, 30, 20, 90, 30, 45, 30, 20, 60, 30, 20, 120, 30, 60, 30, 45, 30, 20, 60, 30, 45, 30, 30]


# ═══════════════════════════════════════════════════════════════════════════
# SESSION 10: Deep career/life direction conversation (40 msgs, 2026-01-22)
# HIGH SIGNAL — career goals, PM switch, tech lead path
# ═══════════════════════════════════════════════════════════════════════════
s10 = [
    ("Jordan", "hey can I ask you something kind of serious", "N4", "Meta opener"),
    ("Alex", "yeah of course what's up", "N1", "Response"),
    ("Jordan", "do you ever think about where you want to be in like 5 years", "S2", "Career/life reflection topic"),
    ("Alex", "lol what prompted this", "N1", "Question filler"),
    ("Jordan", "idk. I've been at Dropbox for 3 years now and I'm not sure I want to do this forever", "S1", "Employment duration + career doubt"),
    ("Alex", "oh wow ok", "N3", "Reaction"),
    ("Jordan", "like the work is fine but it doesn't excite me the way your Anthropic stuff seems to excite you", "S5", "Emotional: career dissatisfaction expressed through comparison"),
    ("Alex", "I mean that's a real feeling tho. you should take it seriously", "N1", "Advice filler"),
    ("Jordan", "yeah", "N1", "Agreement"),
    ("Alex", "honestly that's exactly how I felt before I left Stripe. started creeping in around September", "B3", "Temporal reference to when career doubts started"),
    ("Alex", "what would you want to do if you could do anything", "N1", "Open question"),
    ("Jordan", "honestly? I've been thinking about product management", "S2", "Career aspiration: PM"),
    ("Alex", "oh interesting. why PM specifically", "N1", "Follow-up question"),
    ("Jordan", "I feel like I'm always the person on the team thinking about the user. why we're building things not just how", "S2", "Self-assessment of PM fit"),
    ("Alex", "that's actually really true. you've always been like that", "N5", "Affirming echo"),
    ("Jordan", "I actually took this online PM course from Reforge a few weeks ago. it was really eye-opening", "B2", "Casual fact: Reforge course, temporal"),
    ("Alex", "oh nice. are you thinking about actually making the switch?", "N1", "Question"),
    ("Jordan", "maybe. I've been looking at some associate PM roles. Google has one on the Cloud team that looks really interesting", "S3", "Career switch consideration + specific company/team"),
    ("Alex", "dude you should totally go for it", "N1", "Encouragement"),
    ("Jordan", "you think? I feel like I'd basically be starting over from scratch", "S5", "Emotional: career change anxiety"),
    ("Alex", "I mean yeah kind of but that's literally what I just did and it was the best decision I've made in years", "B1", "Self-reference as reassuring context"),
    ("Jordan", "true. you do seem way happier", "N5", "Echo observation"),
    ("Alex", "I really am. like night and day from Stripe honestly", "S5", "Emotional: happiness at new job"),
    ("Alex", "I think you should go for it. life's too short to feel meh about your work", "S2", "Career advice/opinion"),
    ("Jordan", "yeah you're right", "N5", "Agreement echo"),
    ("Alex", "oh and Lisa from Anthropic actually told me something similar when I was debating my move. she said it's never too late to pivot", "B1", "Callback to Lisa from Sessions 2 and 5"),
    ("Jordan", "ok enough about my existential crisis lol. what about you. where do you see yourself in 5 years", "N4", "Topic return"),
    ("Alex", "honestly I want to be leading a team eventually. not just individual contributor work", "S2", "Career aspiration: leadership"),
    ("Jordan", "oh like eng management?", "N1", "Clarifying question"),
    ("Alex", "yeah or tech lead. something where I get to shape the direction of what we're building", "S2", "Career aspiration detail"),
    ("Alex", "Ava actually mentioned there might be a tech lead opportunity on my team in about 6 months", "S1", "Promotion timeline fact"),
    ("Jordan", "oh wow already?", "N3", "Reaction"),
    ("Alex", "yeah the team is growing fast and she said if I prove myself on the eval project I'd be first in line", "S1", "Promotion path detail"),
    ("Jordan", "that's incredible", "N1", "Supportive filler"),
    ("Alex", "yeah we'll see. no pressure at all lol", "N1", "Self-deprecating filler"),
    ("Alex", "and long term I'm really drawn to AI safety. I want to work on stuff that actually matters", "S2", "Long-term career values: AI safety"),
    ("Jordan", "I love that", "N1", "Supportive filler"),
    ("Alex", "yeah. it feels good to be at a place where the mission actually resonates with what I care about", "S5", "Emotional: values alignment"),
    ("Jordan", "ok well this conversation was extremely adult of us", "N1", "Meta commentary filler"),
    ("Alex", "haha remember when our biggest problem was which bar to hit on a Friday night", "B1", "Callback to shared college history"),
]
g10 = [30, 120, 30, 180, 60, 30, 45, 20, 60, 30, 120, 30, 90, 45, 30, 20, 60, 30, 90, 45, 30, 60, 30, 20, 90, 30, 120, 30, 60, 20, 45, 30, 60, 30, 120, 30, 60, 45, 30]


# ═══════════════════════════════════════════════════════════════════════════
# SESSION 11: Travel plans + updates (35 msgs, 2026-01-28)
# Mix of noise and temporal signal — Tahoe trip, Google app update
# ═══════════════════════════════════════════════════════════════════════════
s11 = [
    ("Alex", "hey so the housewarming was incredible btw. did I tell you that", "N4", "Meta opener with callback"),
    ("Jordan", "haha you told me like 3 times already but I appreciate it every time", "N1", "Filler"),
    ("Alex", "ok well it was!! Priya loved it too", "S5", "Emotional: Priya enjoyed the party"),
    ("Jordan", "aw good. I really liked her btw. you guys are adorable together", "S2", "Jordan's opinion of Priya"),
    ("Alex", "thanks!! means a lot. she said the same thing about you actually", "N1", "Social filler"),
    ("Jordan", "I'm very likeable 💅", "N1", "Self-deprecating humor filler"),
    ("Alex", "lol humble too", "N1", "Banter filler"),
    ("Alex", "hey so random question. are you doing anything Presidents Day weekend? Feb 14-17", "B3", "Temporal reference + date range for trip"),
    ("Jordan", "oh good question. I don't think so why", "N1", "Question filler"),
    ("Alex", "Priya and I were talking about doing a road trip up to Tahoe. you should totally come", "S3", "Travel plan + invitation"),
    ("Jordan", "oh that sounds amazing actually", "N3", "Positive reaction"),
    ("Alex", "yeah she's never been to Tahoe and I haven't gone since college honestly", "B2", "Casual temporal fact about last visit"),
    ("Jordan", "wait it'll be Valentine's Day tho. you don't want like couple time?", "B3", "Temporal awareness: Valentine's overlap"),
    ("Alex", "haha we talked about it. she actually suggested inviting people. she wants to do like a group cabin thing", "S1", "Priya's suggestion + group plan detail"),
    ("Jordan", "oh that's even better", "N3", "Reaction"),
    ("Alex", "yeah there's this cabin on Airbnb in Truckee that sleeps 8. like $400 a night split between everyone", "S1", "Specific location + price breakdown"),
    ("Jordan", "oh that's super reasonable", "N1", "Commentary filler"),
    ("Alex", "right? I was thinking you, me, Priya, Mike, Sarah, maybe your coworker Anil?", "S1", "Proposed guest list"),
    ("Jordan", "oh Anil would be so down. that dude lives for snowboarding", "B2", "Casual fact about Anil's hobby"),
    ("Alex", "perfect. I'll send out a group text this week", "S3", "Action plan commitment"),
    ("Jordan", "sick. I'm excited", "N1", "Excitement filler"),
    ("Alex", "same. ok other topic. did you end up applying to that Google PM thing?", "N4", "Topic transition with callback"),
    ("Jordan", "I did!! submitted last Thursday actually", "S1", "Application submitted + temporal"),
    ("Alex", "oh nice!! how do you feel about it", "N1", "Question filler"),
    ("Jordan", "nervous honestly. the JD was pretty intense but I think I put together a solid application", "S5", "Emotional: nervous but confident about PM app"),
    ("Alex", "I'm sure you did. when do you think you'll hear back?", "N1", "Question filler"),
    ("Jordan", "no idea. Google is notoriously slow lol. could be weeks", "B3", "Temporal expectation"),
    ("Alex", "haha yeah that tracks", "N1", "Commentary filler"),
    ("Jordan", "but yeah I'm cautiously optimistic", "S5", "Emotional state about career move"),
    ("Alex", "as you should be. you'd genuinely be a great PM", "N1", "Supportive filler"),
    ("Jordan", "aw ok that was like 5 people now so it's still a good group for a cabin", "S1", "Confirmed trip group size"),
    ("Alex", "oh btw I keep meaning to ask. have you finished Severance yet?", "B1", "Callback to Session 1 TV show"),
    ("Jordan", "YES oh my god. the finale was unreal", "B2", "Casual fact: finished the show"),
    ("Alex", "no spoilers!! I'm still only on like episode 6 somehow", "B2", "Casual fact: Alex's progress in show"),
    ("Jordan", "haha ok ok lips sealed 🤐", "N1", "Filler"),
]
g11 = [60, 30, 45, 30, 20, 60, 90, 30, 45, 30, 20, 120, 30, 60, 30, 45, 30, 20, 60, 30, 120, 30, 45, 30, 60, 30, 20, 90, 45, 30, 120, 30, 60, 30]


# ═══════════════════════════════════════════════════════════════════════════
# SESSION 12: Reflection + Jordan's big news (35 msgs, 2026-02-02)
# META-SIGNAL — references earlier sessions, summarizes major events
# ═══════════════════════════════════════════════════════════════════════════
s12 = [
    ("Jordan", "hey so how's month one at Anthropic", "N4", "Meta-reflection opener"),
    ("Alex", "honestly kind of surreal that it's already been a month", "B3", "Temporal reflection"),
    ("Jordan", "right? time flies", "N1", "Agreement filler"),
    ("Alex", "like can you believe everything that happened since November", "N4", "Reflection setup"),
    ("Alex", "lol the Alex redemption arc", "B1", "Meta-reference to narrative of prior sessions"),
    ("Jordan", "haha forreal tho", "N1", "Agreement filler"),
    ("Alex", "three months ago I was miserable at Stripe under Derek", "B1", "Callback to Stripe/Derek"),
    ("Jordan", "you really were. those vent sessions were rough lol", "N5", "Echo with reference"),
    ("Alex", "and now I'm at Anthropic making way more money doing work I genuinely care about", "S1", "Current state summary"),
    ("Jordan", "plus you have Priya", "B1", "Callback to relationship"),
    ("Alex", "plus I have Priya 🥺", "S5", "Emotional acknowledgment"),
    ("Alex", "honestly she's been the best part. we're coming up on 3 months and it still feels brand new", "S1", "Relationship duration + current state"),
    ("Jordan", "that's the sweet spot honestly", "N1", "Commentary filler"),
    ("Alex", "yeah. oh and she actually got promoted at Figma last week. Senior Product Designer now", "S1", "Priya's promotion + new title"),
    ("Jordan", "oh nice!! tell her congrats from me", "N1", "Polite filler"),
    ("Alex", "will do. ok but what about you tho. any updates on the Google thing?", "N4", "Topic transition"),
    ("Jordan", "ok SO. I got an email yesterday", "S4", "Life event: interview progression"),
    ("Alex", "wait", "N3", "Reaction"),
    ("Alex", "WAIT", "N3", "Reaction"),
    ("Jordan", "they want to do a phone screen next Wednesday 😭", "S1", "Interview detail: phone screen + day"),
    ("Alex", "DUDE", "N3", "Reaction"),
    ("Alex", "oh man this is really happening for you", "S5", "Emotional investment in friend's success"),
    ("Jordan", "maybe!! I'm trying not to get too ahead of myself", "S5", "Cautious emotional state"),
    ("Alex", "no get excited. you took the Reforge course, you updated your whole portfolio, you've been doing the work", "B1", "Callback to PM course + preparation arc"),
    ("Jordan", "haha true. ok yeah I'm excited", "N1", "Agreement filler"),
    ("Alex", "I'm gonna send you the same energy you gave me before my Anthropic loop", "B1", "Callback to pre-interview support"),
    ("Jordan", "I appreciate that 🙏", "N1", "Gratitude filler"),
    ("Alex", "oh btw are we still good for Tahoe Presidents Day weekend?", "B3", "Callback to trip plans + temporal"),
    ("Jordan", "yes! I talked to Anil and he's in. so that's you me Priya Anil and Mike?", "S1", "Confirmed attendee list"),
    ("Alex", "yeah Mike confirmed yesterday. Sarah can't make it, she has a wedding that weekend", "S1", "Trip logistics: confirmed + conflict"),
    ("Jordan", "aw bummer. 5 people is still great for a cabin tho", "N1", "Commentary filler"),
    ("Alex", "totally. the Truckee cabin sleeps 8 so plenty of room", "B1", "Callback to cabin details from Session 11"),
    ("Jordan", "perfect. I'm pumped", "N1", "Excitement filler"),
    ("Alex", "same. ok gotta go but seriously good luck Wednesday. you're gonna crush it", "B3", "Temporal reference to phone screen"),
    ("Jordan", "thanks dude. talk soon ❤️", "N2", "Farewell"),
]
g12 = [45, 30, 60, 30, 120, 45, 30, 60, 30, 20, 90, 30, 60, 120, 30, 20, 60, 30, 20, 90, 30, 45, 60, 30, 20, 120, 30, 60, 30, 45, 30, 120, 45, 30]


# ═══════════════════════════════════════════════════════════════════════════
# BUILD AND WRITE
# ═══════════════════════════════════════════════════════════════════════════
sessions = [
    (7,  "2026-01-08", "2026-01-08T12:30:00", g7,  s7),
    (8,  "2026-01-12", "2026-01-12T11:00:00", g8,  s8),
    (9,  "2026-01-16", "2026-01-16T20:15:00", g9,  s9),
    (10, "2026-01-22", "2026-01-22T21:00:00", g10, s10),
    (11, "2026-01-28", "2026-01-28T14:00:00", g11, s11),
    (12, "2026-02-02", "2026-02-02T19:30:00", g12, s12),
]

all_messages = []
for num, date, start, gaps, msgs in sessions:
    assert len(gaps) == len(msgs) - 1, \
        f"Session {num}: {len(gaps)} gaps but {len(msgs)} msgs (need {len(msgs)-1} gaps)"
    all_messages.extend(build(num, date, start, gaps, msgs))

cat_counts = Counter(m["category"] for m in all_messages)
noise_total = sum(v for k, v in cat_counts.items() if k.startswith("N"))
signal_total = sum(v for k, v in cat_counts.items() if k.startswith("S"))
border_total = sum(v for k, v in cat_counts.items() if k.startswith("B"))
speaker_counts = Counter(m["speaker"] for m in all_messages)

output = {
    "conversation_id": "conv1",
    "part": 2,
    "sessions_covered": [7, 8, 9, 10, 11, 12],
    "speakers": {
        "Alex": "Now at Anthropic as Senior ML Infra Engineer, dating Priya",
        "Jordan": "At Dropbox, new apartment in Richmond, exploring PM career switch",
    },
    "date_range": {"start": "2026-01-08", "end": "2026-02-02"},
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

out_path = Path("benchmarks/gate_eval/datasets/gate_benchmark_conv1_part2.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"✓ Wrote {len(all_messages)} messages to {out_path}")
print(f"  Noise:      {noise_total} ({noise_total/len(all_messages)*100:.1f}%)")
print(f"  Signal:     {signal_total} ({signal_total/len(all_messages)*100:.1f}%)")
print(f"  Borderline: {border_total} ({border_total/len(all_messages)*100:.1f}%)")
print(f"  Speakers:   {dict(speaker_counts)}")
print(f"  Categories: {dict(sorted(cat_counts.items()))}")
for num, _, _, _, _ in sessions:
    sm = [m for m in all_messages if m["session"] == f"session_{num}"]
    sc = Counter(m["category"] for m in sm)
    n = sum(v for k, v in sc.items() if k.startswith("N"))
    s = sum(v for k, v in sc.items() if k.startswith("S"))
    b = sum(v for k, v in sc.items() if k.startswith("B"))
    sp = Counter(m["speaker"] for m in sm)
    print(f"  Session {num}: {len(sm)} msgs | N={n} S={s} B={b} | {dict(sp)}")
