#!/usr/bin/env python3
"""Phase 4: Conv 3 — Dev & Casey (romantic partners), 400 messages, 14 sessions over 8 weeks.

Dev = detailed planner, sends links/prices, makes lists. Uses "babe", "Cas".
Casey = emotional, exclamation marks, pet names ("babe", "babyyyy", "D").
       Career change arc from law to UX design.

Category targets: 55% noise, 30% signal, 15% borderline.
"""

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


def build(sess_num: int, date: str, start: str, gaps: list[int],
          msgs: list[tuple], conv="conv3") -> list[dict]:
    ts = make_timestamps(start, gaps)
    speakers = set(m[0] for m in msgs)
    sp_list = sorted(speakers)
    out = []
    for i, (spk, txt, cat, note) in enumerate(msgs):
        recip = [s for s in sp_list if s != spk][0]
        out.append({
            "id": f"{conv}_s{sess_num:02d}_{i+1:03d}",
            "conversation_id": conv,
            "session": f"session_{sess_num}",
            "session_date": date,
            "speaker": spk,
            "recipient": recip,
            "content": txt,
            "timestamp": ts[i],
            "category": cat,
            "is_signal": not cat.startswith("N"),
            "noise_type": NOISE_TYPES.get(cat),
            "notes": note,
        })
    return out


def gaps_for(n, pattern=None):
    """Generate n-1 gaps (seconds) from a cycling pattern."""
    if pattern is None:
        pattern = [45, 30, 60, 20, 90, 30, 45, 120, 30, 60,
                   20, 45, 30, 90, 60, 30, 45, 20, 120, 30]
    return [pattern[i % len(pattern)] for i in range(n - 1)]


# ═══════════════════════════════════════════════════════════════════════════
# CONVERSATION 3: Dev & Casey  (romantic partners)
# Dev  = planner, prices/links/lists, warm but measured. "babe", "Cas"
# Casey = emotional, exclamation marks, pet names. "babe", "babyyyy", "D"
# 14 sessions over 8 weeks, 400 messages, 55% noise target
# ═══════════════════════════════════════════════════════════════════════════

# ── Session 1: Good morning texts, Dev cooking dinner (30 msgs, 2025-11-02 Sun)
# HIGH NOISE — affectionate fluff
s1 = [
    ("Casey", "good morningggg babe ☀️", "N2", "Morning greeting"),
    ("Dev", "morning Cas 🥰", "N2", "Morning greeting"),
    ("Casey", "I slept SO good last night omg", "N1", "Sleep commentary filler"),
    ("Dev", "same honestly. that new pillow is life-changing", "B2", "Casual fact: new pillow"),
    ("Casey", "right?! best $60 ever spent", "N5", "Affirming echo"),
    ("Dev", "haha truth", "N1", "Filler"),
    ("Casey", "whatcha doing today", "N1", "Question filler"),
    ("Dev", "I wanna try making that pasta from the NYT Cooking video. the one with the pistachio pesto?", "S1", "Cooking plan: pistachio pesto pasta, NYT source"),
    ("Casey", "omg YES. that looked amazing", "N3", "Excited reaction"),
    ("Dev", "I need to grab basil, pistachios, and pecorino. probably like $15 worth of stuff", "S1", "Ingredient list + cost estimate"),
    ("Casey", "ooh get some of that sourdough from the bakery too", "S2", "Preference: bakery sourdough"),
    ("Dev", "the one on 5th? Flour & Co?", "B1", "Context: bakery name + location"),
    ("Casey", "yes!! their rosemary loaf is insane", "S2", "Preference: rosemary loaf"),
    ("Dev", "done. I'll swing by around 4", "B3", "Temporal: shopping at 4pm"),
    ("Casey", "you're the best 🥺", "N1", "Affection filler"),
    ("Dev", "anything for you babe", "N1", "Affection filler"),
    ("Casey", "ugh I love you", "N1", "Love declaration filler"),
    ("Dev", "love you more", "N1", "Love declaration filler"),
    ("Casey", "not possible!!", "N1", "Protest filler"),
    ("Dev", "lol ok ok we could do this all day", "N4", "Meta-conversation"),
    ("Casey", "haha fine. I'm gonna go shower", "N1", "Filler"),
    ("Dev", "k. oh hey what time do you want to eat? like 7?", "B3", "Temporal: dinner at 7"),
    ("Casey", "7 is perfect! I'll open that bottle of wine from Sarah's party", "B2", "Casual fact: wine from Sarah's party"),
    ("Dev", "ooh the Malbec? yes please", "S2", "Preference: Malbec wine"),
    ("Casey", "you know it babe 🍷", "N1", "Confirmation filler"),
    ("Dev", "this is gonna be a good Sunday", "N1", "Sentiment filler"),
    ("Casey", "the BEST Sunday", "N5", "Echo"),
    ("Dev", "ok heading to the store. text you later", "N2", "Farewell"),
    ("Casey", "have fun!! love youuu", "N2", "Farewell"),
    ("Dev", "love you ❤️", "N2", "Farewell"),
]
g1 = gaps_for(30, [60, 30, 45, 20, 30, 60, 90, 30, 20, 45,
                   30, 20, 30, 60, 20, 15, 15, 20, 30, 120])

# ── Session 2: Quick check-in, Casey burnout (25 msgs, 2025-11-04 Tue)
s2 = [
    ("Dev", "hey how was your day", "N2", "Evening check-in"),
    ("Casey", "ugh", "N1", "Exhaustion filler"),
    ("Casey", "honestly terrible", "S5", "Emotional: bad day"),
    ("Dev", "oh no what happened", "N3", "Concerned reaction"),
    ("Casey", "just like... everything? we had back to back depositions and then two partners dumped briefs on me at 4pm", "S1", "Work details: depositions + brief dump"),
    ("Dev", "that's brutal. did you eat?", "N1", "Caring question filler"),
    ("Casey", "no 😭 I had like a granola bar at 2", "B2", "Casual fact: skipped real meals"),
    ("Dev", "Cas. ok I'm ordering thai. green curry?", "S2", "Food preference check: green curry"),
    ("Casey", "you angel. yes please extra spicy", "S2", "Preference: extra spicy"),
    ("Dev", "on it", "N1", "Confirmation filler"),
    ("Casey", "I'm just so burned out D", "S5", "Emotional: burnout admission"),
    ("Casey", "like I used to love this job and now I literally dread every morning", "S5", "Emotional: dread going to work"),
    ("Dev", "I know babe. it's been like this for a while now huh", "N5", "Empathetic echo"),
    ("Casey", "yeah. months honestly", "B3", "Temporal: months of burnout"),
    ("Dev", "have you thought about what you want to do about it?", "N1", "Open question filler"),
    ("Casey", "idk. like I can't just quit obviously. student loans and everything", "S1", "Constraint: student loans"),
    ("Dev", "no I know. just saying we can figure something out if you need a change", "N1", "Supportive filler"),
    ("Casey", "yeah. maybe. idk", "N1", "Deflection filler"),
    ("Casey", "can we just watch something dumb on Netflix tonight", "B2", "Casual fact: Netflix as comfort routine"),
    ("Dev", "absolutely. that new comedy special?", "N1", "Suggestion filler"),
    ("Casey", "perfect", "N1", "Agreement filler"),
    ("Dev", "Thai should be there in 30. come home whenever", "B3", "Temporal: food ETA"),
    ("Casey", "leaving now. 15 min", "B3", "Temporal: 15 min commute"),
    ("Dev", "❤️", "N1", "Heart filler"),
    ("Casey", "love you. seriously. thank you", "N2", "Grateful farewell"),
]
g2 = gaps_for(25, [120, 15, 30, 90, 30, 45, 20, 30, 60, 120,
                   20, 30, 45, 30, 60, 30, 20, 90, 30, 45])

# ── Session 3: Dev finds cheap flights to Japan! (30 msgs, 2025-11-09 Sun)
s3 = [
    ("Dev", "CAS", "N1", "Excitement opener"),
    ("Dev", "BABE", "N1", "Excitement opener"),
    ("Dev", "LOOK AT THIS", "N1", "Excitement opener"),
    ("Casey", "what what what!!", "N3", "Urgent reaction"),
    ("Dev", "I found flights to Tokyo for $890 round trip on ANA. direct from LAX!!", "S1", "Fact: flight price, airline, route"),
    ("Casey", "SHUT UP", "N3", "Shocked reaction"),
    ("Dev", "I'm dead serious. March dates too which is cherry blossom season 🌸", "S1", "Fact: March timing = cherry blossoms"),
    ("Casey", "oh my god oh my god oh my god", "N3", "Overwhelmed reaction"),
    ("Casey", "we've been talking about Japan for YEARS", "N4", "Meta: referencing past conversations"),
    ("Dev", "I know!! the prices are usually like $1400+ so this is insane", "S1", "Fact: usual price comparison"),
    ("Casey", "ok wait can we actually afford it tho", "N1", "Practical concern filler"),
    ("Dev", "I ran the numbers. flights would be $1780 for both of us. I think we could do 2 weeks for about $5k total including everything", "S1", "Fact: total budget estimate $5k for 2 weeks"),
    ("Casey", "hmmm that's a lot but also like... when else are we gonna find $890 flights", "N1", "Deliberation filler"),
    ("Dev", "exactly. and we've got like $3200 in the travel fund already", "S1", "Fact: travel fund balance $3200"),
    ("Casey", "wait we do?? I thought it was like $2k", "N3", "Surprised reaction"),
    ("Dev", "nah I've been adding $200/month since August", "S1", "Fact: savings rate $200/month since August"),
    ("Casey", "babyyyy you're incredible", "N1", "Affection filler"),
    ("Dev", "I try 😌 so is that a yes?", "N1", "Prompt filler"),
    ("Casey", "YES. a million times YES!!", "S3", "Decision: agreeing to Japan trip"),
    ("Dev", "let's gooooo 🇯🇵", "N3", "Celebration reaction"),
    ("Casey", "ok but I need to request the time off. when exactly in March?", "B3", "Temporal: time-off request needed"),
    ("Dev", "the cheapest dates are March 15 to 29. two full weeks", "S1", "Fact: proposed dates March 15-29"),
    ("Casey", "I'll email my manager tomorrow. omg I'm literally shaking", "S5", "Emotional: physical excitement"),
    ("Dev", "haha same. ok I'm gonna put together a rough itinerary tonight", "N1", "Planning filler"),
    ("Casey", "of course you are lol. classic Dev", "N1", "Teasing filler"),
    ("Dev", "you love my spreadsheets", "N1", "Teasing filler"),
    ("Casey", "I really do 😂", "N5", "Affirming echo"),
    ("Dev", "alright let me start researching neighborhoods. I'll send you options", "N1", "Planning filler"),
    ("Casey", "this is the best day ever!!!", "N1", "Excitement filler"),
    ("Dev", "❤️🇯🇵❤️", "N1", "Emoji filler"),
]
g3 = gaps_for(30, [10, 5, 30, 60, 20, 45, 15, 30, 60, 20,
                   120, 30, 45, 30, 20, 30, 20, 30, 45, 20])

# ── Session 4: Casey terrible day, emotional support (28 msgs, 2025-11-12 Wed)
s4 = [
    ("Casey", "I am going to scream", "S5", "Emotional: fury at work"),
    ("Dev", "uh oh. what now", "N3", "Concerned reaction"),
    ("Casey", "Gerald is the WORST human being alive", "S1", "Fact: senior partner named Gerald"),
    ("Casey", "he literally yelled at me in front of the whole team because I missed a comma in a filing", "S1", "Fact: public humiliation over typo"),
    ("Dev", "a comma?? are you serious", "N3", "Outraged reaction"),
    ("Casey", "A COMMA. like it wasn't even a substantive error!!", "N5", "Echo with emphasis"),
    ("Dev", "that's completely unacceptable. has he done this before?", "B1", "Context: establishing pattern of behavior"),
    ("Casey", "yeah he's been getting worse honestly. last week he made Rachel cry in a meeting", "S1", "Fact: Gerald made Rachel cry too"),
    ("Dev", "is there an HR process for this?", "N1", "Practical question"),
    ("Casey", "lol D this is big law. Gerald IS the process. he's a name partner", "S1", "Fact: Gerald is a name partner"),
    ("Dev", "that's messed up. I'm sorry babe", "N1", "Sympathy filler"),
    ("Casey", "like I went to law school for 3 years and $180k in debt to get screamed at about commas", "S1", "Fact: $180k law school debt"),
    ("Dev", "I know. you deserve better than this", "N1", "Supportive filler"),
    ("Casey", "honestly today made me realize I don't think I even want to be a lawyer anymore", "S5", "Emotional: questioning career path"),
    ("Dev", "wow. ok. like... at all?", "N3", "Surprised reaction"),
    ("Casey", "idk. maybe? I've been thinking about it for a while but today just", "B3", "Temporal: 'a while' indicates ongoing dissatisfaction"),
    ("Casey", "like broke something in me? does that make sense", "S5", "Emotional: feeling broken"),
    ("Dev", "it makes total sense. we can talk about it more when you get home", "N1", "Supportive deferral"),
    ("Casey", "yeah. sorry for the vent", "N4", "Meta: acknowledging vent"),
    ("Dev", "never apologize for that. that's literally what I'm here for", "N1", "Reassurance filler"),
    ("Casey", "🥺 what did I do to deserve you", "N1", "Affection filler"),
    ("Dev", "you existed. that's enough", "N1", "Affection filler"),
    ("Casey", "ok I'm gonna cry again but in a good way this time lol", "N1", "Humor through tears"),
    ("Dev", "I'll have dinner ready when you get home. the leftover ramen", "B2", "Casual fact: leftover ramen at home"),
    ("Casey", "perfect. leaving in 20", "B3", "Temporal: 20 min ETA"),
    ("Dev", "take your time. I love you", "N2", "Farewell"),
    ("Casey", "love you so much D", "N2", "Farewell"),
    ("Dev", "❤️", "N1", "Heart filler"),
]
g4 = gaps_for(28, [30, 15, 90, 20, 30, 30, 60, 30, 45, 60,
                   120, 20, 30, 20, 45, 30, 30, 20, 30, 60])

# ── Session 5: Japan planning — hotels, neighborhoods, budget (30 msgs, 2025-11-16 Sun)
s5 = [
    ("Dev", "ok babe Japan planning time. I made a spreadsheet 📊", "N4", "Meta: planning session opener"),
    ("Casey", "omg here we go hahaha I love it", "N1", "Teasing filler"),
    ("Dev", "so I've been researching neighborhoods. for Tokyo I think we should split between Shibuya and Shinjuku", "S1", "Fact: target neighborhoods Shibuya + Shinjuku"),
    ("Casey", "what's the difference?", "N1", "Question filler"),
    ("Dev", "Shibuya is more trendy, shopping, nightlife. Shinjuku has the Golden Gai bar district and better transit connections", "S1", "Fact: neighborhood characteristics"),
    ("Casey", "oh I definitely wanna see Golden Gai!! I saw it on TikTok", "S2", "Preference: wants Golden Gai"),
    ("Dev", "it's on the list. ok so for Kyoto I found this incredible traditional ryokan", "S1", "Fact: ryokan in Kyoto"),
    ("Casey", "a what?", "N1", "Question filler"),
    ("Dev", "it's a traditional Japanese inn. tatami floors, futon beds, communal baths. this one is $120/night", "S1", "Fact: ryokan description + price $120/night"),
    ("Casey", "omg that sounds amazing!! how many nights in Kyoto?", "N3", "Excited reaction"),
    ("Dev", "I was thinking 4 nights Kyoto, 8 nights Tokyo, 2 nights Osaka", "S1", "Fact: proposed night split by city"),
    ("Casey", "wait why Osaka?", "N1", "Question filler"),
    ("Dev", "street food capital of Japan. they call it the 'kitchen of Japan'", "S1", "Fact: Osaka known for street food"),
    ("Casey", "ok you had me at street food", "N1", "Agreement filler"),
    ("Dev", "lol figured. ok budget breakdown", "N4", "Meta: transition to budget"),
    ("Dev", "flights: $1780. accommodation: roughly $1500 for 14 nights. food/transport/activities: $1500. total: $4780", "S1", "Fact: full budget breakdown $4780"),
    ("Casey", "so under $5k!!", "N5", "Echo confirmation"),
    ("Dev", "yep. and we already have $3200 saved. need $1580 more by March", "S1", "Fact: savings gap $1580"),
    ("Casey", "that's totally doable. like $400/month for 4 months", "B3", "Temporal: savings timeline 4 months"),
    ("Dev", "exactly. I can cover most of it. you focus on the loan payments", "S3", "Decision: Dev covers extra savings"),
    ("Casey", "D... you don't have to do that", "N1", "Protest filler"),
    ("Dev", "I want to. this trip is gonna be incredible", "N1", "Reassurance filler"),
    ("Casey", "it really is. I cannot WAIT to see the cherry blossoms 🌸", "S2", "Preference: excited for cherry blossoms"),
    ("Dev", "oh also I found a cooking class in Kyoto. traditional kaiseki cuisine. $80 per person", "S1", "Fact: Kyoto cooking class $80pp"),
    ("Casey", "YES. book it immediately", "S3", "Decision: book cooking class"),
    ("Dev", "done. March 18th afternoon slot", "S1", "Fact: cooking class date"),
    ("Casey", "this is going to be the best trip of our lives", "N1", "Excitement filler"),
    ("Dev", "agreed. ok let me book the flights before the price goes up", "S3", "Decision: booking flights now"),
    ("Casey", "DO IT. omg this is real!!", "N3", "Excited reaction"),
    ("Dev", "it's real babe 🇯🇵✨", "N1", "Confirmation filler"),
]
g5 = gaps_for(30, [30, 60, 30, 90, 20, 60, 30, 120, 30, 45,
                   30, 60, 20, 30, 45, 90, 30, 20, 60, 30])

# ── Session 6: Casey looking at UX design courses (25 msgs, 2025-11-19 Wed)
s6 = [
    ("Casey", "ok so don't judge me", "N4", "Meta: preface"),
    ("Dev", "never. what's up", "N1", "Response filler"),
    ("Casey", "I've been looking at UX design bootcamps", "S4", "Life event: exploring career pivot to UX design"),
    ("Dev", "wait really?? that's awesome!", "N3", "Excited reaction"),
    ("Casey", "yeah I know it sounds random but hear me out", "N4", "Meta: building argument"),
    ("Casey", "so remember how I redesigned all the filing templates at work and everyone loved them?", "B1", "Context: redesigned filing templates"),
    ("Dev", "yeah you spent like an entire weekend on that", "N5", "Confirming echo"),
    ("Casey", "right and it was the most fun I've had at work in MONTHS. like I was in a flow state the whole time", "S5", "Emotional: flow state experience"),
    ("Dev", "I mean that tracks. you've always had a great eye for design", "B2", "Casual fact: Casey has design aptitude"),
    ("Casey", "so I found this bootcamp called Designlab. it's 6 months, fully online, and they have a UX career track", "S1", "Fact: Designlab, 6 months, online, UX career track"),
    ("Dev", "how much does it cost?", "N1", "Question"),
    ("Casey", "ok so this is the scary part. it's $6,500", "S1", "Fact: bootcamp cost $6,500"),
    ("Dev", "hmm. that's not nothing but it's not crazy for a career change program either", "N1", "Assessment filler"),
    ("Casey", "yeah and they have payment plans. like $1100/month for 6 months", "S1", "Fact: payment plan option $1100/month"),
    ("Dev", "have you looked at reviews? placement rates?", "N1", "Due diligence question"),
    ("Casey", "YES ok so their placement rate is 89% within 6 months of graduating. and the average starting salary for UX designers in LA is like $85k", "S1", "Fact: 89% placement rate, $85k avg salary"),
    ("Dev", "that's actually really solid. and you'd be way happier", "N1", "Assessment filler"),
    ("Casey", "right?! like I wouldn't be getting screamed at by Gerald lol", "B1", "Context: callback to Gerald"),
    ("Dev", "ok so what's holding you back?", "N1", "Question"),
    ("Casey", "I guess just... the fear? like what if I'm not good enough. what if I'm making a huge mistake leaving law", "S5", "Emotional: fear and self-doubt"),
    ("Dev", "Cas, you literally taught yourself Figma in a weekend. you redesigned an entire template system for fun. you're gonna crush this", "B1", "Context: Casey taught herself Figma"),
    ("Casey", "ugh ok when you put it like that", "N1", "Concession filler"),
    ("Dev", "just think about it. I'll support whatever you decide. zero pressure", "N1", "Supportive filler"),
    ("Casey", "I love you so much", "N2", "Grateful farewell"),
    ("Dev", "love you too. now go do some secret Figma stuff 😂", "N2", "Teasing farewell"),
]
g6 = gaps_for(25, [45, 30, 60, 20, 30, 90, 30, 120, 30, 45,
                   30, 60, 20, 30, 90, 30, 20, 45, 60, 30])

# ── Session 7: Apartment renovation — kitchen remodel (32 msgs, 2025-11-23 Sun)
s7 = [
    ("Dev", "ok so I spent all morning looking at kitchen renovation stuff", "N4", "Meta: topic opener"),
    ("Casey", "oh god here we go 😂", "N1", "Teasing filler"),
    ("Dev", "hear me out! the cabinets are literally falling apart", "B1", "Context: cabinets in bad shape"),
    ("Casey", "no you're right they ARE bad. the one above the stove doesn't even close anymore", "B1", "Context: specific cabinet issue"),
    ("Dev", "exactly. so I got some quotes online and I think we could redo the whole kitchen for $15k-$20k", "S1", "Fact: renovation budget $15k-$20k"),
    ("Casey", "that's... a lot of money babe", "N1", "Concern filler"),
    ("Dev", "it is but listen. Shaker-style cabinets in white, quartz countertops, new backsplash. total transformation", "S1", "Fact: proposed materials — Shaker cabinets, quartz, backsplash"),
    ("Casey", "ok quartz is gorgeous. what color were you thinking?", "N1", "Question"),
    ("Dev", "I like the Calacatta-look quartz. white with grey veining. it's about $55-75 per square foot installed", "S1", "Fact: quartz style + price $55-75/sqft"),
    ("Casey", "oooh that's the one that looks like marble right?", "B1", "Context: marble-look clarification"),
    ("Dev", "exactly but way more durable. no sealing, no staining", "S1", "Fact: quartz advantages over marble"),
    ("Casey", "ok I'm into it", "S2", "Preference: likes the quartz choice"),
    ("Dev", "for the backsplash I was thinking subway tile. classic, cheap, easy to clean", "S1", "Fact: subway tile backsplash proposal"),
    ("Casey", "ooh what about that herringbone pattern? my friend Lina just did hers and it's stunning", "S2", "Preference: herringbone pattern, friend Lina reference"),
    ("Dev", "oh that would look amazing actually. herringbone subway tile. great call", "N5", "Affirming echo"),
    ("Casey", "see I have good ideas too 😤", "N1", "Playful protest filler"),
    ("Dev", "never doubted it babe. ok so biggest question: can we do this ourselves or hire someone?", "N1", "Question"),
    ("Casey", "D you literally flooded the bathroom trying to fix the faucet last year", "B2", "Casual fact: Dev flooded bathroom previously"),
    ("Dev", "... that was ONE time", "B2", "Casual fact: flooding was one incident"),
    ("Casey", "hire someone lol", "S3", "Decision: hire a contractor"),
    ("Dev", "yeah you're probably right. I'll start getting quotes this week", "N1", "Agreement filler"),
    ("Casey", "also can we do one of those farmhouse sinks? the big deep ones", "S2", "Preference: farmhouse sink"),
    ("Dev", "a Belfast sink? yeah those run about $400-600 for a good one", "S1", "Fact: sink type + price range"),
    ("Casey", "yes!! ok I'm actually getting excited about this", "N3", "Excited reaction"),
    ("Dev", "same. our kitchen is gonna be incredible", "N1", "Excitement filler"),
    ("Casey", "we're really adulting huh", "N1", "Commentary filler"),
    ("Dev", "full blown adulting. Japan trip AND a kitchen reno", "N5", "Echo callback"),
    ("Casey", "lol we're either really smart or really dumb", "N1", "Humor filler"),
    ("Dev", "why not both 😂", "N1", "Humor filler"),
    ("Dev", "ok I'll put together a full cost breakdown this week", "N1", "Planning filler"),
    ("Casey", "my little spreadsheet king 👑", "N1", "Teasing filler"),
    ("Dev", "you know it 📊", "N1", "Self-aware filler"),
]
g7 = gaps_for(32, [45, 30, 60, 30, 90, 20, 120, 30, 30, 45,
                   30, 60, 20, 30, 30, 90, 30, 45, 30, 20])

# ── Session 8: Dev making ramen from scratch (25 msgs, 2025-11-26 Wed)
s8 = [
    ("Dev", "babe. I'm doing it. I'm making ramen from scratch", "S1", "Fact: attempting homemade ramen"),
    ("Casey", "wait ACTUAL from scratch?? like the broth and everything??", "N3", "Surprised reaction"),
    ("Dev", "everything. tonkotsu broth. pork bones, the whole deal", "S1", "Fact: tonkotsu style, pork bones"),
    ("Casey", "D that takes like a million hours doesn't it", "B1", "Context: awareness of long broth time"),
    ("Dev", "12 hours for the broth 😅 I started at 6am", "S1", "Fact: broth time 12 hours, started 6am"),
    ("Casey", "you absolute madman lol", "N3", "Amused reaction"),
    ("Dev", "the apartment smells INSANE already. like a ramen shop", "N1", "Descriptive filler"),
    ("Casey", "omg I can't wait to get home", "N1", "Excitement filler"),
    ("Dev", "ok so here's my ingredient list: 4 lbs pork neck bones, dried shiitake, kombu, niboshi, ginger, garlic, whole black peppercorns", "S1", "Fact: detailed ingredient list"),
    ("Casey", "I have no idea what half of those are but I trust you", "N1", "Trust filler"),
    ("Dev", "lol niboshi are dried sardines. they add umami depth", "S1", "Fact: niboshi = dried sardines for umami"),
    ("Casey", "you're literally a food scientist now", "N1", "Teasing filler"),
    ("Dev", "I also made the tare — that's the seasoning base. soy sauce, mirin, sake, and a little brown sugar", "S1", "Fact: tare ingredients"),
    ("Casey", "babe you should start a food blog honestly", "B2", "Casual: idea for Dev to food blog"),
    ("Dev", "haha nah this is just for us. oh I'm also doing chashu pork belly", "S1", "Fact: making chashu pork belly"),
    ("Casey", "the rolled up braised one?? 🤤", "B1", "Context: Casey recognizes chashu preparation"),
    ("Dev", "yep. 2 lbs of pork belly rolled and tied, braising for 3 hours in soy-mirin", "S1", "Fact: chashu preparation method"),
    ("Casey", "I am literally drooling at my desk rn", "N1", "Reaction filler"),
    ("Dev", "haha sorry. or you're welcome? eggs are marinating too. ajitsuke tamago", "S1", "Fact: marinated eggs (ajitsuke tamago)"),
    ("Casey", "ok I need to stop reading these I'm starving 😩", "N1", "Hunger filler"),
    ("Dev", "come home at 7. it'll be ready", "B3", "Temporal: ready at 7"),
    ("Casey", "I will RUN home", "N1", "Enthusiasm filler"),
    ("Dev", "I got noodles from the Japanese market on Sawtelle too. Sun Noodle brand", "S1", "Fact: noodle brand + store location"),
    ("Casey", "you thought of everything. I love you so much", "N2", "Grateful farewell"),
    ("Dev", "love you. prepare your taste buds 🍜", "N2", "Farewell"),
]
g8 = gaps_for(25, [60, 30, 45, 30, 120, 20, 90, 30, 60, 30,
                   45, 30, 120, 30, 60, 20, 30, 45, 60, 30])

# ── Session 9: Casey considering quitting law (30 msgs, 2025-12-01 Mon)
s9 = [
    ("Casey", "D can we talk about something serious", "N4", "Meta: serious topic opener"),
    ("Dev", "of course. everything ok?", "N3", "Concerned reaction"),
    ("Casey", "yeah. well. kind of", "N1", "Hesitation filler"),
    ("Casey", "I've been thinking about this nonstop and I think I want to quit the firm", "S3", "Decision: wants to quit law firm"),
    ("Dev", "ok. tell me more", "N1", "Listening prompt filler"),
    ("Casey", "like actually quit. not 'think about it someday' quit. like give my notice quit", "S3", "Decision: serious about quitting"),
    ("Dev", "I hear you. what's the timeline you're thinking?", "N1", "Practical question"),
    ("Casey", "I want to start Designlab in January. so I'd give two weeks notice in mid-December", "S1", "Fact: Designlab January start, notice mid-December"),
    ("Dev", "ok and financially — walk me through it", "N1", "Practical question"),
    ("Casey", "so my salary right now is $95k. student loan payments are $1400/month. rent is split with you obviously", "S1", "Fact: salary $95k, loan payment $1400/month"),
    ("Dev", "right. and the bootcamp is $6500", "B1", "Context: callback to bootcamp cost"),
    ("Casey", "yeah. so I've got about $12k in savings. that would cover the bootcamp plus like 4 months of loan payments", "S1", "Fact: $12k savings, 4-month runway"),
    ("Dev", "what about picking up some freelance legal work? contract stuff?", "B1", "Context: freelance legal work idea"),
    ("Casey", "I was thinking that actually! there's a legal research platform called LawClerk where you can do freelance work", "S1", "Fact: LawClerk freelance platform"),
    ("Dev", "that's smart. so you'd have some income while doing the bootcamp", "N5", "Affirming echo"),
    ("Casey", "yeah hopefully enough to cover the loans at least", "B1", "Context: freelance covering loan payments"),
    ("Dev", "and I can cover a bigger share of rent for 6 months. that's fine", "S3", "Decision: Dev covers more rent"),
    ("Casey", "babe no that's not fair to you", "N1", "Protest filler"),
    ("Dev", "Cas. we're a team. this is what teams do", "N1", "Reassurance filler"),
    ("Casey", "I'm literally tearing up at work rn lol", "S5", "Emotional: moved to tears"),
    ("Dev", "well don't let Gerald see 😂", "N1", "Humor filler"),
    ("Casey", "hahaha omg", "N1", "Laugh filler"),
    ("Casey", "so you really think I should do this?", "B1", "Context: seeking partner validation before career pivot"),
    ("Dev", "100%. you're miserable at the firm. you light up when you talk about design. the math works out. what's the downside?", "N1", "Encouragement filler"),
    ("Casey", "the downside is I'm 28 starting over from scratch", "S5", "Emotional: fear of starting over"),
    ("Dev", "you're 28 with a law degree and the courage to chase something that makes you happy. that's not starting over. that's leveling up", "N1", "Reframe filler"),
    ("Casey", "ok I'm full on crying now thanks a lot 😭❤️", "S5", "Emotional: overwhelmed with gratitude"),
    ("Dev", "good tears?", "N1", "Check-in filler"),
    ("Casey", "the BEST tears. ok. I'm doing it. I'm actually doing this", "S3", "Decision: final commitment to career change"),
    ("Dev", "let's go baby 🚀", "N3", "Celebration reaction"),
]
g9 = gaps_for(30, [60, 30, 20, 90, 30, 60, 30, 120, 30, 45,
                   30, 60, 20, 30, 45, 90, 30, 20, 60, 30])

# ── Session 10: Quick lovey-dovey check-in, HIGH NOISE (20 msgs, 2025-12-04 Thu)
s10 = [
    ("Casey", "hiiiii babe", "N2", "Greeting"),
    ("Dev", "hey beautiful ❤️", "N2", "Greeting"),
    ("Casey", "I miss youuu. this work trip is killing me", "B2", "Casual fact: Casey on a work trip"),
    ("Dev", "I miss you too. house is too quiet without you", "B2", "Casual fact: Dev home alone, they live together"),
    ("Casey", "awww 🥺 how many more days", "N1", "Question filler"),
    ("Dev", "you're back Saturday right?", "B3", "Temporal: return day"),
    ("Casey", "yes!! Saturday morning. my flight lands at 10:15am", "S1", "Fact: return flight time"),
    ("Dev", "I'll pick you up. want me to make brunch?", "B2", "Casual fact: Dev will pick Casey up from airport"),
    ("Casey", "omg yes please. your eggs benedict?", "S2", "Preference: Dev's eggs benedict"),
    ("Dev", "you got it", "N1", "Confirmation filler"),
    ("Casey", "you're the best human on this planet", "N1", "Affection filler"),
    ("Dev", "debatable but I'll take it", "N1", "Humor filler"),
    ("Casey", "not debatable 😤", "N1", "Protest filler"),
    ("Dev", "lol ok ok. hey I love you", "N1", "Love declaration filler"),
    ("Casey", "I love you more", "N1", "Love declaration filler"),
    ("Dev", "not possible", "N5", "Echo of earlier pattern"),
    ("Casey", "hahaha we are disgusting", "N1", "Self-aware filler"),
    ("Dev", "proudly 😂", "N1", "Agreement filler"),
    ("Casey", "ok gotta go to this dinner thing. love you goodnight! 💕", "N2", "Farewell"),
    ("Dev", "night babe. sleep well ❤️", "N2", "Farewell"),
]
g10 = gaps_for(20, [30, 45, 20, 30, 60, 30, 20, 45, 30, 20,
                    30, 15, 20, 15, 20, 30, 30, 60, 30, 45])

# ── Session 11: Japan trip booked! Flight + Airbnb details (28 msgs, 2025-12-07 Sun)
s11 = [
    ("Dev", "Cas.", "N1", "Attention getter"),
    ("Dev", "it's booked.", "S4", "Life event: Japan trip fully booked"),
    ("Casey", "WHAT. EVERYTHING??", "N3", "Shocked reaction"),
    ("Dev", "everything. flights, Airbnb, ryokan. all of it", "S1", "Fact: all accommodations confirmed"),
    ("Casey", "AHHHHH!!!! 😭😭😭", "N3", "Overwhelmed reaction"),
    ("Dev", "ok here are the details", "N4", "Meta: transition to details"),
    ("Dev", "flights: ANA, LAX → Narita, departing March 15 at 11:30am, arriving March 16 at 3:45pm local time", "S1", "Fact: outbound flight details"),
    ("Casey", "omg it's so real", "N1", "Awe filler"),
    ("Dev", "return: March 29, Narita → LAX, departing 5:10pm arriving same day 11:20am (time zone magic)", "S1", "Fact: return flight details"),
    ("Casey", "I love time zone magic lol", "N1", "Comment filler"),
    ("Dev", "Airbnb in Shibuya confirmed. 1-bedroom, walking distance to Shibuya Crossing. $95/night for 8 nights = $760", "S1", "Fact: Airbnb details — Shibuya, $95/night, $760 total"),
    ("Casey", "walking distance to Shibuya Crossing?! babe!!", "N3", "Excited reaction"),
    ("Dev", "yep. the reviews are insane. 4.97 stars, 200+ reviews. the host speaks English too", "S1", "Fact: Airbnb rating 4.97, 200+ reviews"),
    ("Casey", "you're unreal. what about Kyoto?", "N1", "Question filler"),
    ("Dev", "ryokan Kumo no Ue — 'above the clouds.' 4 nights, $120/night = $480. includes breakfast!", "S1", "Fact: ryokan name, price, breakfast included"),
    ("Casey", "ryokan Kumo no Ue 🥺 that's so beautiful", "N1", "Sentiment filler"),
    ("Dev", "and Osaka: a hostel-hotel hybrid near Dotonbori for 2 nights. $65/night", "S1", "Fact: Osaka accommodation $65/night near Dotonbori"),
    ("Casey", "ok so total damage?", "N1", "Budget question filler"),
    ("Dev", "flights: $1780. accommodation: $1370. we're at $3150 so far with $1850 left for food/transport/activities", "S1", "Fact: running total + remaining budget"),
    ("Casey", "babe this is so well planned I could cry", "S5", "Emotional: moved by planning"),
    ("Dev", "I already bought a Japan Rail Pass too. 14-day pass, $420 for both of us. unlimited bullet trains", "S1", "Fact: JR Pass $420 for both, 14 days, unlimited shinkansen"),
    ("Casey", "BULLET TRAINS", "N3", "Excited reaction"),
    ("Dev", "told you I'd handle the spreadsheet 😌", "N1", "Self-satisfied filler"),
    ("Casey", "best spreadsheet you've ever made and you've made A LOT", "N1", "Teasing filler"),
    ("Dev", "98 days and counting 🇯🇵", "B3", "Temporal: countdown"),
    ("Casey", "I literally cannot wait. this is going to change our lives", "N1", "Excitement filler"),
    ("Dev", "agreed. now help me pick restaurants. I have a list of 47", "B2", "Casual fact: Dev curated 47 restaurant options"),
    ("Casey", "FORTY SEVEN?! 😂 ok send it", "N3", "Amused reaction"),
]
g11 = gaps_for(28, [10, 45, 20, 30, 15, 60, 30, 120, 20, 90,
                    30, 60, 30, 45, 30, 60, 30, 90, 20, 30])

# ── Session 12: Renovation contractor quotes (30 msgs, 2025-12-14 Sun)
s12 = [
    ("Dev", "ok renovation update. I got three contractor quotes back", "S1", "Fact: received three contractor quotes"),
    ("Casey", "ooh ok lay it on me", "N1", "Prompt filler"),
    ("Dev", "contractor 1: Mike's Remodeling. $22k. 8 weeks. decent reviews", "S1", "Fact: quote 1 — Mike's, $22k, 8 weeks"),
    ("Casey", "oof $22k is over budget", "B1", "Context: $22k exceeds their $15-20k range"),
    ("Dev", "yeah. contractor 2: some guy named Steve off Thumbtack. $14k but his reviews are sketchy", "S1", "Fact: quote 2 — Steve, $14k, bad reviews"),
    ("Casey", "hard no on sketchy Steve lol", "N3", "Amused rejection reaction"),
    ("Dev", "lol agreed. contractor 3 though... Marco from Roma Construction", "S1", "Fact: quote 3 — Marco, Roma Construction"),
    ("Casey", "ok?", "N1", "Prompt filler"),
    ("Dev", "$18k, 6 week timeline, 4.9 stars on Yelp with 85 reviews. he specializes in kitchen and bath", "S1", "Fact: Marco's quote $18k, 6 weeks, 4.9 stars, 85 reviews"),
    ("Casey", "oh I like that. $18k is right in our budget", "B1", "Context: $18k within budget"),
    ("Dev", "AND he came over yesterday and was super professional. brought a portfolio, references, the whole thing", "S1", "Fact: Marco visited, brought portfolio"),
    ("Casey", "wait he came to the apartment already?", "N3", "Surprised reaction"),
    ("Dev", "yeah I wanted to see how he carried himself. he measured everything, asked about our style preferences", "S1", "Fact: Marco measured + discussed style"),
    ("Casey", "aw D you did all that while I was out?", "N1", "Touched filler"),
    ("Dev", "of course. he also said the Calacatta quartz we want is a great choice. he works with a supplier that does $62/sqft installed", "S1", "Fact: quartz price through Marco $62/sqft"),
    ("Casey", "omg that's on the lower end of what you quoted before!", "N5", "Echo referencing earlier prices"),
    ("Dev", "exactly. I think Marco is our guy", "S3", "Decision: choosing Marco"),
    ("Casey", "I agree. when would he start?", "B3", "Temporal: asking about start date"),
    ("Dev", "he said he could start mid-January. so we'd be done by end of February", "S1", "Fact: start mid-January, done end of February"),
    ("Casey", "before the Japan trip!! perfect timing", "B3", "Temporal: renovation done before Japan"),
    ("Dev", "that's what I was thinking. come home to a brand new kitchen AND jet lag lol", "N1", "Humor filler"),
    ("Casey", "hahaha living our best life", "N1", "Filler"),
    ("Dev", "ok one more thing. we decided on quartz. do we want the Shaker cabinets in white or navy?", "B1", "Context: setting up cabinet color decision"),
    ("Casey", "oh that's tough. can we do white uppers and navy lowers?? I saw that on Pinterest", "S2", "Preference: two-tone cabinets white/navy"),
    ("Dev", "ooh two-tone. that's actually trending right now. I love it", "S2", "Preference: agrees with two-tone"),
    ("Casey", "yesss ok I'm so excited!! tell Marco we're in", "S3", "Decision: confirming Marco"),
    ("Dev", "sending him an email right now. we're locked in babe", "N1", "Action filler"),
    ("Casey", "our kitchen is gonna be SO cute omg", "N1", "Excitement filler"),
    ("Dev", "cute AND functional. that's the Dev special 😌", "N1", "Self-satisfied filler"),
    ("Casey", "😂 love you nerd", "N2", "Farewell"),
]
g12 = gaps_for(30, [30, 60, 30, 90, 20, 30, 45, 120, 30, 60,
                    30, 45, 30, 20, 90, 30, 60, 30, 45, 20])

# ── Session 13: Casey enrolled in Designlab, gave notice (27 msgs, 2025-12-17 Wed)
s13 = [
    ("Casey", "D", "N1", "Attention getter"),
    ("Casey", "I did it", "N1", "Lead-in filler"),
    ("Casey", "I enrolled in Designlab AND I gave my two weeks notice today", "S4", "Life event: enrolled in Designlab + gave notice"),
    ("Dev", "CASEY", "N3", "Excited reaction"),
    ("Dev", "OH MY GOD", "N3", "Excited reaction"),
    ("Dev", "I am SO proud of you babe", "N1", "Pride filler"),
    ("Casey", "I'm shaking lol", "S5", "Emotional: physically shaking"),
    ("Dev", "tell me everything. how did it go?", "N1", "Prompt filler"),
    ("Casey", "ok so I walked into the managing partner's office this morning. Janet, not Gerald obviously lol", "S1", "Fact: told Janet (managing partner), not Gerald"),
    ("Casey", "and I just said it. I said I'm pursuing a career change and my last day will be December 31st", "S4", "Life event: gave notice, last day Dec 31st"),
    ("Dev", "how did she react?", "B1", "Context: prompting for Janet's reaction"),
    ("Casey", "she was actually really nice about it? she said she could tell I'd been unhappy and she wished me well", "S1", "Fact: Janet's supportive reaction"),
    ("Dev", "aw that's classy", "N1", "Assessment filler"),
    ("Casey", "yeah. she even said if I ever wanted to come back the door was open", "S1", "Fact: open door to return"),
    ("Dev", "that's great to have as a safety net. and Gerald?", "B1", "Context: asking about Gerald's reaction"),
    ("Casey", "lol Gerald just said 'noted' and went back to his laptop", "S1", "Fact: Gerald's dismissive response"),
    ("Dev", "wow. shocking. what a gem", "N1", "Sarcasm filler"),
    ("Casey", "hahaha right?! ANYWAY. Designlab starts January 6th. 6-month program, I'll be done by July", "S1", "Fact: Designlab dates — Jan 6 start, done by July"),
    ("Dev", "July!! that's so exciting. what's the curriculum like?", "N1", "Question filler"),
    ("Casey", "so first 2 months are foundations — design thinking, user research, wireframing. then 2 months of UI/visual design. then 2 months of prototyping and portfolio building", "S1", "Fact: curriculum breakdown by phase"),
    ("Dev", "and you get a mentor right?", "B1", "Context: setting up mentor reveal"),
    ("Casey", "yes!! a 1-on-1 industry mentor for the whole program. mine is a senior UX designer at Spotify named Priya", "S1", "Fact: mentor Priya, senior UX at Spotify"),
    ("Dev", "a Spotify designer?? that's incredible", "N3", "Impressed reaction"),
    ("Casey", "I KNOW. I literally stalked her LinkedIn and her portfolio is insane", "N1", "Excitement filler"),
    ("Dev", "Cas this is going to be so good. 2026 is YOUR year", "N1", "Encouragement filler"),
    ("Casey", "our year babe. Japan, new kitchen, new career. we're doing it all!! 🚀", "B3", "Temporal: 2026 plans summary"),
    ("Dev", "we really are. I love you so much. I'm so proud of you", "N2", "Farewell"),
]
g13 = gaps_for(27, [10, 15, 30, 10, 30, 45, 60, 20, 90, 30,
                    60, 20, 30, 30, 120, 30, 45, 20, 60, 30])

# ── Session 14: Dev's dinner party, Japan countdown, year reflection (40 msgs, 2025-12-27 Sat)
s14 = [
    ("Casey", "babyyyy last night was AMAZING", "N2", "Morning-after greeting"),
    ("Dev", "you think?? I was so nervous", "B2", "Casual fact: Dev nervous about hosting"),
    ("Casey", "D. you cooked a full 5-course dinner for 8 people and everyone was raving about it", "S1", "Fact: 5-course dinner for 8 guests"),
    ("Dev", "ok the ramen course was definitely the highlight", "S2", "Preference: ramen was best course"),
    ("Casey", "are you kidding?? the ramen was incredible but that miso-glazed salmon was otherworldly", "S2", "Preference: miso-glazed salmon was best"),
    ("Dev", "haha I was worried about that one actually. first time doing the 48-hour miso marinade", "S1", "Fact: 48-hour miso marinade technique"),
    ("Casey", "well it was perfect. Sarah literally asked for the recipe like 3 times", "B2", "Casual fact: Sarah loved it, asked for recipe"),
    ("Dev", "I know 😂 I'll text it to her later", "N1", "Filler"),
    ("Casey", "and Ryan said it was better than most restaurants he's been to and he's like a total food snob", "B2", "Casual fact: Ryan's high praise, Ryan is food snob"),
    ("Dev", "ok that one actually means a lot coming from him", "N1", "Gratitude filler"),
    ("Casey", "you should be proud of yourself babe. seriously. a year ago you could barely boil pasta", "B3", "Temporal: Dev's cooking growth over a year"),
    ("Dev", "hey that's not entirely... ok yeah that's fair lol", "B2", "Casual fact: Dev admits prior cooking inability"),
    ("Casey", "the fact that you went from that to hosting an 8-person dinner party with homemade ramen broth is actually insane", "S1", "Fact: Dev's cooking milestone trajectory"),
    ("Dev", "I guess I found my thing. like how you found design", "B1", "Context: parallel to Casey's career pivot"),
    ("Casey", "exactly!! we're both becoming the best versions of ourselves 🥺", "S5", "Emotional: growth reflection"),
    ("Dev", "ok don't make me cry at 10am", "N1", "Deflection filler"),
    ("Casey", "haha too late! ok let's talk Japan. 77 DAYS", "B3", "Temporal: 77-day countdown"),
    ("Dev", "I literally have a countdown widget on my phone", "B2", "Casual fact: Dev has countdown widget"),
    ("Casey", "of course you do 😂 ok so what do we still need to do", "N1", "Question filler"),
    ("Dev", "ok checklist: 1) get yen exchanged, 2) buy a pocket wifi, 3) figure out the cooking class outfits, 4) Casey needs a new suitcase", "S1", "Fact: Japan prep checklist"),
    ("Casey", "wait why do I need a new suitcase", "B1", "Context: setting up broken suitcase reveal"),
    ("Dev", "babe your suitcase wheel literally fell off on the Portland trip", "B2", "Casual fact: broken suitcase from Portland"),
    ("Casey", "oh yeah lol ok fair", "N1", "Concession filler"),
    ("Dev", "I found a good one on Amazon. Away carry-on, $275 but there's a 20% off code right now", "S1", "Fact: suitcase recommendation + price + discount"),
    ("Casey", "ooh Away luggage is so cute. get the green one!", "S2", "Preference: green Away suitcase"),
    ("Dev", "the sage green? done", "B2", "Casual fact: ordered sage green suitcase"),
    ("Casey", "yesss. ok also I wanted to talk about something", "N4", "Meta: topic transition"),
    ("Dev", "what's up?", "N1", "Prompt filler"),
    ("Casey", "I just want to say... this year has been the most transformative year of my life", "S5", "Emotional: year reflection"),
    ("Casey", "like I went from crying in the office bathroom to enrolling in a design bootcamp. I booked a trip to JAPAN. we're renovating our kitchen", "S5", "Emotional: summarizing transformation"),
    ("Dev", "Cas...", "N1", "Moved filler"),
    ("Casey", "and none of it would have happened without you believing in me when I didn't believe in myself", "S5", "Emotional: gratitude for Dev's support"),
    ("Dev", "ok NOW I'm crying at 10am", "N1", "Emotional filler"),
    ("Casey", "GOOD haha", "N1", "Humor filler"),
    ("Dev", "you did this though. all I did was hold the flashlight. you walked through the door", "N1", "Reframe filler"),
    ("Casey", "ugh stop being poetic it's making it worse 😭", "N1", "Protest filler"),
    ("Dev", "haha I love you. 2026 is going to be our best year yet", "S5", "Emotional: forward-looking optimism"),
    ("Casey", "new kitchen in February, Japan in March, I finish Designlab in July... it's all happening", "S1", "Fact: 2026 timeline summary"),
    ("Casey", "I love you so much D. you're my favorite person in the whole entire world", "N2", "Farewell"),
    ("Dev", "you're mine too Cas. always. ❤️", "N2", "Farewell"),
]
g14 = gaps_for(40, [30, 60, 30, 45, 90, 20, 30, 120, 30, 60,
                    45, 30, 60, 20, 30, 90, 30, 20, 45, 30])


# ═══════════════════════════════════════════════════════════════════════════
# BUILD AND WRITE
# ═══════════════════════════════════════════════════════════════════════════
sessions = [
    (1,  "2025-11-02", "2025-11-02T09:15:00", g1,  s1),
    (2,  "2025-11-04", "2025-11-04T18:30:00", g2,  s2),
    (3,  "2025-11-09", "2025-11-09T10:00:00", g3,  s3),
    (4,  "2025-11-12", "2025-11-12T17:45:00", g4,  s4),
    (5,  "2025-11-16", "2025-11-16T11:00:00", g5,  s5),
    (6,  "2025-11-19", "2025-11-19T20:15:00", g6,  s6),
    (7,  "2025-11-23", "2025-11-23T10:30:00", g7,  s7),
    (8,  "2025-11-26", "2025-11-26T12:00:00", g8,  s8),
    (9,  "2025-12-01", "2025-12-01T19:00:00", g9,  s9),
    (10, "2025-12-04", "2025-12-04T21:30:00", g10, s10),
    (11, "2025-12-07", "2025-12-07T14:00:00", g11, s11),
    (12, "2025-12-14", "2025-12-14T11:00:00", g12, s12),
    (13, "2025-12-17", "2025-12-17T16:30:00", g13, s13),
    (14, "2025-12-27", "2025-12-27T10:00:00", g14, s14),
]

all_messages = []
for num, date, start, gaps, msgs in sessions:
    assert len(gaps) == len(msgs) - 1, \
        f"Session {num}: {len(gaps)} gaps for {len(msgs)} msgs (need {len(msgs)-1})"
    all_messages.extend(build(num, date, start, gaps, msgs))

# ── Category + speaker stats ──
cat_counts = Counter(m["category"] for m in all_messages)
noise_total = sum(v for k, v in cat_counts.items() if k.startswith("N"))
signal_total = sum(v for k, v in cat_counts.items() if k.startswith("S"))
border_total = sum(v for k, v in cat_counts.items() if k.startswith("B"))
speaker_counts = Counter(m["speaker"] for m in all_messages)

output = {
    "conversation_id": "conv3",
    "speakers": {
        "Dev": "Detailed planner, sends links/prices/lists, warm but measured. Uses 'babe', 'Cas' for Casey.",
        "Casey": "Emotional, exclamation marks, pet names ('babe', 'babyyyy', 'D'). Career change from law to UX design.",
    },
    "date_range": {"start": "2025-11-02", "end": "2025-12-27"},
    "message_count": len(all_messages),
    "category_distribution": {
        "noise": {k: v for k, v in sorted(cat_counts.items()) if k.startswith("N")},
        "signal": {k: v for k, v in sorted(cat_counts.items()) if k.startswith("S")},
        "borderline": {k: v for k, v in sorted(cat_counts.items()) if k.startswith("B")},
        "totals": {
            "noise": noise_total, "signal": signal_total, "borderline": border_total,
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

out_dir = Path(__file__).resolve().parent.parent / "datasets"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "gate_benchmark_conv3.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"✓ Wrote {len(all_messages)} messages to {out_path}")
print(f"  Noise:      {noise_total} ({noise_total/len(all_messages)*100:.1f}%)")
print(f"  Signal:     {signal_total} ({signal_total/len(all_messages)*100:.1f}%)")
print(f"  Borderline: {border_total} ({border_total/len(all_messages)*100:.1f}%)")
print(f"  Speakers:   {dict(speaker_counts)}")
for num, _, _, _, _ in sessions:
    sm = [m for m in all_messages if m["session"] == f"session_{num}"]
    sc = Counter(m["category"] for m in sm)
    n = sum(v for k, v in sc.items() if k.startswith("N"))
    s = sum(v for k, v in sc.items() if k.startswith("S"))
    b = sum(v for k, v in sc.items() if k.startswith("B"))
    print(f"  Session {num:2d}: {len(sm):3d} msgs | N={n:2d} S={s:2d} B={b:2d}")
