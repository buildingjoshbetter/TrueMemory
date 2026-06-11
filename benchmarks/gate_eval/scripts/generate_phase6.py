#!/usr/bin/env python3
"""Generate GateLoCoMo Conv 5: Taylor & Morgan (college friends reconnecting).

400 messages across 6 sessions over 2 weeks (2025-11-08 to 2025-11-22).
Taylor lives in Berlin (product manager at N26), Morgan in Portland (new parent, baby Olive).
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


def build(sess_num: int, date: str, start: str, gaps: list[int], msgs: list[tuple], conv: str = "conv5") -> list[dict]:
    ts = make_timestamps(start, gaps)
    out = []
    for i, (spk, txt, cat, note) in enumerate(msgs):
        out.append({
            "id": f"{conv}_s{sess_num:02d}_{i+1:03d}",
            "conversation_id": conv,
            "session": f"session_{sess_num}",
            "session_date": date,
            "speaker": spk,
            "recipient": "Morgan" if spk == "Taylor" else "Taylor",
            "content": txt,
            "timestamp": ts[i],
            "category": cat,
            "is_signal": not cat.startswith("N"),
            "noise_type": NOISE_TYPES.get(cat),
            "notes": note,
        })
    return out


# ═══════════════════════════════════════════════════════════════════════════
# SESSION 1: The reconnection (70 msgs, 2025-11-08 Saturday)
# Taylor reaches out after 5 years. Massive catching-up energy.
# ═══════════════════════════════════════════════════════════════════════════
s1 = [
    ("Taylor", "omg is this still your number?!", "N2", "Opening greeting after 5 years"),
    ("Taylor", "it's Taylor!! from UO!!", "S1", "Self-identification + college reference"),
    ("Morgan", "WAIT", "N3", "Shock reaction"),
    ("Morgan", "TAYLOR???", "N3", "Continued shock"),
    ("Morgan", "oh my god hi!!! yes it's still me!!", "N2", "Excited greeting"),
    ("Taylor", "dude I can't believe this still works haha", "N1", "Relief filler"),
    ("Taylor", "I was literally cleaning out my contacts and saw your name and was like I NEED to text this person", "B1", "Context-enabling: reason for reconnecting"),
    ("Morgan", "omg I'm so happy you did", "N1", "Warm filler"),
    ("Morgan", "it's been what like 5 years??", "B3", "Temporal marker: 5 years since contact"),
    ("Taylor", "at least!! I think the last time we actually talked was like right after graduation?", "B3", "Temporal marker: last contact was post-graduation"),
    ("Morgan", "yeah that sounds right", "N5", "Agreement echo"),
    ("Morgan", "ok SO. catch me up. what are you doing where are you TELL ME EVERYTHING", "N4", "Meta-conversation: requesting update"),
    ("Taylor", "ok ok ok so", "N4", "Stalling meta"),
    ("Taylor", "I live in Berlin", "S1", "Key fact: Taylor lives in Berlin"),
    ("Morgan", "WHAT", "N3", "Shock reaction"),
    ("Morgan", "like Berlin GERMANY??", "N3", "Clarification shock"),
    ("Taylor", "haha yes Berlin Germany!!", "B1", "Context-enabling: confirming location"),
    ("Taylor", "I moved here about 3 years ago", "S1", "Fact: moved to Berlin ~3 years ago"),
    ("Taylor", "I work at this fintech company called N26, I'm a product manager there", "S1", "Fact: works at N26 as product manager"),
    ("Morgan", "shut UP", "N3", "Disbelief reaction"),
    ("Morgan", "that is insane", "N3", "Amazement reaction"),
    ("Morgan", "how did that even happen??", "N4", "Requesting backstory"),
    ("Taylor", "honestly it was kind of random? I was at this startup in San Francisco for like 2 years after school and then N26 was hiring and they were like do you want to move to Berlin", "S1", "Background: SF startup before Berlin move"),
    ("Taylor", "and I was like... yes?? obviously??", "B1", "Context-enabling: accepted Berlin offer"),
    ("Morgan", "dude that's incredible", "N1", "Supportive filler"),
    ("Taylor", "ok but WAIT. what about you!! where are you what's going on", "N4", "Meta: redirecting to Morgan"),
    ("Morgan", "ok so", "N4", "Stalling meta"),
    ("Morgan", "I'm in Portland", "S1", "Fact: Morgan lives in Portland"),
    ("Taylor", "oh nice!!", "N1", "Positive filler"),
    ("Morgan", "yeah we moved here like 4 years ago", "B3", "Temporal marker: moved to Portland ~4 years ago"),
    ("Morgan", "and um", "N4", "Hesitation meta"),
    ("Morgan", "I HAD A BABY", "S4", "Major life event: had a baby"),
    ("Taylor", "WAIT WHAT", "N3", "Shock reaction"),
    ("Taylor", "oh my GOD", "N3", "Overwhelmed reaction"),
    ("Taylor", "MORGAN", "N3", "Emphasis reaction"),
    ("Taylor", "a BABY???", "N3", "Repeated disbelief"),
    ("Morgan", "haha yes!! a little girl, her name is Olive", "S1", "Fact: baby girl named Olive"),
    ("Taylor", "Olive omg that's the cutest name ever", "N3", "Gushing reaction"),
    ("Morgan", "she's 3 months old", "S1", "Fact: Olive is 3 months old"),
    ("Taylor", "I literally cannot handle this", "N1", "Overwhelmed filler"),
    ("Taylor", "wait so who's the... partner? are you married? I need the full story", "N4", "Meta: requesting details"),
    ("Morgan", "haha ok so I'm with Chris, we've been together like 5 years now", "S1", "Fact: partner named Chris, together 5 years"),
    ("Morgan", "not married yet but like basically married you know", "S1", "Fact: not married but long-term"),
    ("Taylor", "totally", "N5", "Agreement echo"),
    ("Morgan", "Chris is amazing honestly. works in software, we met at this thing in Portland right after I moved here", "B2", "Casual embedded facts: Chris's job, how they met"),
    ("Taylor", "that is so cute", "B1", "Context-enabling: reacting to how Chris and Morgan met"),
    ("Morgan", "what about you?? seeing anyone in Berlin?", "N4", "Redirecting question"),
    ("Taylor", "haha not really, the dating scene here is... interesting", "B2", "Casual embedded opinion on Berlin dating"),
    ("Taylor", "but we can get into that later lol", "N4", "Deferring topic"),
    ("Morgan", "haha fair", "N1", "Acceptance filler"),
    ("Morgan", "ok but wait so you're a product manager?? at a fintech??", "N4", "Circling back"),
    ("Taylor", "yeah! N26 is like a mobile banking app, it's pretty big in Europe", "S1", "Fact: N26 is mobile banking, big in Europe"),
    ("Taylor", "honestly it's a great gig. the work-life balance in Germany is unreal compared to the US", "S2", "Opinion: German work-life balance superior"),
    ("Morgan", "omg I bet", "N5", "Agreement echo"),
    ("Taylor", "like I get 30 days of vacation a year", "S1", "Fact: 30 vacation days"),
    ("Morgan", "thirty DAYS??", "N3", "Shock at vacation days"),
    ("Morgan", "I used to get like 10 at my old job lol", "B2", "Casual embedded fact about old job"),
    ("Taylor", "right?? it's insane. Europe just does it different", "B1", "Context-enabling: cultural comparison"),
    ("Morgan", "so jealous", "B1", "Context-enabling: US vs Europe work culture contrast"),
    ("Taylor", "ok I have so much more to tell you but I gotta run to dinner with friends", "N2", "Farewell setup"),
    ("Taylor", "can we keep catching up tomorrow or whenever??", "N2", "Continuation request"),
    ("Morgan", "YES absolutely", "N5", "Emphatic agreement"),
    ("Morgan", "I'm literally just sitting here feeding Olive at 2am so I have all the time in the world haha", "B2", "Casual fact: up at 2am feeding baby"),
    ("Taylor", "omg you're up at 2am??", "N3", "Reaction to late hour"),
    ("Morgan", "welcome to my life lol", "N1", "Self-deprecating filler"),
    ("Taylor", "haha ok sending you so much love, talk soon!!", "N2", "Farewell"),
    ("Morgan", "so glad you texted!! this made my night honestly", "S5", "Emotional: reconnection joy"),
    ("Taylor", "same!! night! or morning or whatever time it is for you haha", "N2", "Farewell"),
    ("Morgan", "haha night!", "N2", "Farewell"),
    ("Morgan", "wait what time is it in Berlin right now", "B3", "Temporal curiosity"),
]
g1 = [15, 25, 8, 20, 30, 12, 35, 20, 45, 25, 60, 20, 10, 30, 8, 15, 25, 30, 45, 15, 10, 40, 60,
      20, 35, 25, 15, 30, 20, 15, 45, 25, 10, 12, 8, 30, 15, 25, 45, 60, 30, 20, 15, 30, 25, 40,
      30, 15, 60, 20, 35, 30, 45, 15, 20, 40, 25, 15, 90, 30, 20, 15, 25, 30, 20, 45, 25, 30, 40]


# ═══════════════════════════════════════════════════════════════════════════
# SESSION 2: College memories + more catching up (65 msgs, 2025-11-10 Mon)
# Professor Huang, the house on 18th Street, Tailored coffee shop.
# Morgan shares baby milestones. Taylor talks Berlin nightlife + German.
# ═══════════════════════════════════════════════════════════════════════════
s2 = [
    ("Taylor", "ok I'm back!! sorry I disappeared for a day", "N2", "Greeting after gap"),
    ("Morgan", "no worries!! I've been in a baby coma anyway", "N1", "Self-deprecating filler"),
    ("Taylor", "haha how's little Olive doing??", "N4", "Topic opener"),
    ("Morgan", "she is AMAZING actually", "B1", "Context-enabling: positive framing of Olive"),
    ("Morgan", "she did her first real smile last week and I literally cried", "S4", "Life event: Olive's first smile"),
    ("Taylor", "omg stop that's so precious", "N3", "Gushing reaction"),
    ("Morgan", "like not a gas smile, an actual looking-at-me smile", "B1", "Context-enabling detail"),
    ("Taylor", "I can't", "N1", "Overwhelmed filler"),
    ("Morgan", "honestly it makes the sleep deprivation almost worth it", "S5", "Emotional: sleep deprivation struggle"),
    ("Morgan", "I said almost lol", "N1", "Self-correcting humor filler"),
    ("Taylor", "haha how bad is the sleep situation", "N4", "Follow-up question"),
    ("Morgan", "dude. she wakes up every 2-3 hours still", "S1", "Fact: Olive wakes every 2-3 hours"),
    ("Morgan", "Chris and I do shifts but it's still brutal", "S1", "Fact: doing shifts with Chris"),
    ("Taylor", "that sounds honestly terrible", "B1", "Context-enabling: empathizing with sleep struggle"),
    ("Morgan", "it IS terrible haha but also she's the most beautiful thing I've ever seen so", "S5", "Emotional: love despite exhaustion"),
    ("Taylor", "aww", "N1", "Warm filler"),
    ("Taylor", "ok this is so random but I was thinking about this last night", "N4", "Topic transition"),
    ("Taylor", "do you remember Professor Huang's lectures??", "B1", "Context-enabling: college callback"),
    ("Morgan", "oh my GOD", "N3", "Reaction to memory"),
    ("Morgan", "the worst lectures of all time", "B1", "Context-enabling: characterizing Professor Huang"),
    ("Taylor", "the WORST. he would just read the slides word for word", "B1", "Context-enabling: lecture detail"),
    ("Morgan", "and they were like 200 slides per class", "B1", "Context-enabling: slide count"),
    ("Taylor", "haha yes!! and remember he would get mad if you didn't take notes", "B1", "Context-enabling: Huang detail"),
    ("Morgan", "even though the slides were POSTED ONLINE", "N5", "Echo agreement with emphasis"),
    ("Taylor", "exactly!! what was that class even, econ something?", "B1", "Context-enabling: trying to remember class"),
    ("Morgan", "econ 301. intermediate micro. the bane of my existence junior year", "S1", "Fact: it was Econ 301, junior year"),
    ("Taylor", "right!! junior year. god that was the year we all lived in that house on 18th", "S1", "Fact: lived in house on 18th Street junior year"),
    ("Morgan", "the 18th Street house!! that place was a disaster", "N3", "Nostalgic reaction"),
    ("Taylor", "honestly the best year though", "S5", "Emotional: nostalgia for junior year"),
    ("Morgan", "it really was", "N5", "Agreement echo"),
    ("Morgan", "remember going to Tailored every morning before class?", "S1", "Fact: frequented coffee shop called Tailored"),
    ("Taylor", "omg Tailored!! I literally dream about their lattes", "S2", "Preference: loved Tailored's lattes"),
    ("Morgan", "their oat milk latte was life-changing", "S2", "Preference: Tailored's oat milk latte"),
    ("Taylor", "does it still exist?? please tell me it still exists", "N4", "Hopeful question"),
    ("Morgan", "honestly I have no idea, I haven't been back to Eugene in forever", "B3", "Temporal: hasn't been to Eugene in a long time"),
    ("Taylor", "same. well obviously lol I'm in another country", "B1", "Context-enabling: distance from Eugene"),
    ("Morgan", "haha right", "N1", "Agreement filler"),
    ("Taylor", "speaking of which do you want to hear about Berlin nightlife because it is UNHINGED", "N4", "Topic transition"),
    ("Morgan", "YES please I live vicariously through people who leave the house after 8pm now", "N3", "Enthusiastic reaction"),
    ("Taylor", "ok so Berlin clubs literally don't open until midnight and they go until like Monday morning", "S1", "Fact: Berlin clubs midnight to Monday"),
    ("Taylor", "there's this one called Berghain that's like the most famous club in the world", "S1", "Fact: Berghain club"),
    ("Morgan", "wait I've heard of that", "B1", "Context-enabling: recognizes Berghain"),
    ("Taylor", "yeah it's in this old power plant in Friedrichshain, my neighborhood", "S1", "Fact: Berghain in Friedrichshain"),
    ("Morgan", "your NEIGHBORHOOD has the most famous club in the world??", "N3", "Amazement"),
    ("Taylor", "haha well I actually live in Kreuzberg but it's right next door basically", "S1", "Fact: Taylor lives in Kreuzberg"),
    ("Morgan", "that's wild. I don't even know what those words mean", "N1", "Humorous filler"),
    ("Taylor", "they're neighborhoods! like districts. Berlin has a bunch of them with really distinct vibes", "B1", "Context-enabling: Berlin geography"),
    ("Taylor", "Kreuzberg is like the artsy multicultural one. tons of Turkish food, street art everywhere", "S1", "Fact: Kreuzberg is artsy, multicultural, Turkish food"),
    ("Morgan", "that sounds amazing", "N1", "Supportive filler"),
    ("Taylor", "it is. honestly I love it here even though I miss the US sometimes", "S5", "Emotional: loves Berlin, misses US"),
    ("Morgan", "I bet. do you speak German?", "N4", "Follow-up question"),
    ("Taylor", "haha I'm trying!! I take classes twice a week", "S1", "Fact: German classes twice a week"),
    ("Taylor", "I can order food and ask for directions but like don't ask me to have a deep conversation", "B2", "Casual embedded self-assessment: basic German level"),
    ("Morgan", "that's still impressive!!", "N1", "Supportive filler"),
    ("Taylor", "the thing is everyone in Berlin speaks English so it's actually hard to practice", "S1", "Fact: everyone in Berlin speaks English"),
    ("Morgan", "oh that's kind of funny actually", "N1", "Amused filler"),
    ("Taylor", "right?? I'll try to order in German and they just switch to English immediately", "B2", "Casual embedded fact about daily experience"),
    ("Morgan", "haha that would be so frustrating", "N1", "Sympathetic filler"),
    ("Taylor", "it is!! but also convenient lol", "B2", "Casual embedded opinion: convenience of English in Berlin"),
    ("Morgan", "ok I need to go, Olive's doing her scream thing", "N2", "Farewell"),
    ("Taylor", "haha go go go! mom duty calls", "N2", "Farewell"),
    ("Morgan", "talk later this week??", "N2", "Continuation request"),
    ("Taylor", "absolutely!!", "N5", "Emphatic agreement"),
    ("Morgan", "ok bye!! this was so fun", "N2", "Farewell"),
    ("Taylor", "so fun!! bye!!", "N2", "Farewell"),
]
g2 = [45, 30, 25, 15, 35, 20, 25, 30, 20, 15, 40, 25, 20, 35, 30, 45, 20, 30, 15, 25, 20, 15,
      30, 20, 40, 35, 25, 20, 35, 30, 25, 20, 45, 30, 25, 15, 60, 20, 30, 25, 20, 30, 15, 25,
      30, 20, 35, 25, 40, 15, 25, 20, 30, 25, 20, 30, 25, 20, 45, 30, 25, 20, 15, 20]


# ═══════════════════════════════════════════════════════════════════════════
# SESSION 3: Trip planning seeds + deeper sharing (70 msgs, 2025-11-13 Thu)
# Taylor mentions US trip. Berlin apartment cost shocks Morgan.
# Morgan opens up about postpartum struggles.
# ═══════════════════════════════════════════════════════════════════════════
s3 = [
    ("Taylor", "hey!! how's the little one today", "N2", "Greeting"),
    ("Morgan", "she's actually napping right now which feels like a miracle", "B2", "Casual embedded fact: Olive napping"),
    ("Taylor", "ooh rare free time!!", "N1", "Excited filler"),
    ("Morgan", "I know right?? I have like maybe 45 minutes before she wakes up screaming", "B2", "Casual embedded fact: Olive nap duration"),
    ("Taylor", "haha use them wisely", "N1", "Humorous filler"),
    ("Morgan", "I'm using them to talk to you so clearly I have my priorities straight", "B1", "Context-enabling: choosing to reconnect over rest"),
    ("Taylor", "aww haha ok so I have news", "N4", "Topic setup"),
    ("Taylor", "I think I'm gonna come back to the US for a visit in February", "S3", "Decision: planning US trip in February"),
    ("Morgan", "WAIT REALLY??", "N3", "Excited reaction"),
    ("Taylor", "yeah!! I haven't been back in like a year and a half and I'm getting homesick", "S5", "Emotional: homesick, hasn't visited in 1.5 years"),
    ("Morgan", "omg you HAVE to come to Portland", "N3", "Excited plea"),
    ("Taylor", "dude that's literally what I was thinking!!", "N5", "Echo agreement"),
    ("Taylor", "I want to meet Olive!!", "S5", "Emotional: wants to meet baby"),
    ("Morgan", "she would love you. well she loves everyone who holds her but still", "N1", "Humorous filler"),
    ("Taylor", "haha I'll take it", "N1", "Amused filler"),
    ("Taylor", "I'm looking at maybe mid-February? like the 14th through the 21st or something", "S1", "Fact: considering Feb 14-21 trip dates"),
    ("Morgan", "that would be perfect honestly. Chris has that week off too", "S1", "Fact: Chris off that week too"),
    ("Taylor", "oh amazing!! ok I'm gonna look at flights", "S3", "Decision: will look at flights"),
    ("Morgan", "yes do it do it do it", "N3", "Encouraging reaction"),
    ("Taylor", "haha ok ok", "N1", "Agreement filler"),
    ("Taylor", "oh btw I forgot to tell you about my apartment here", "N4", "Topic transition"),
    ("Taylor", "I have a 2-bedroom in Kreuzberg", "S1", "Fact: 2-bedroom in Kreuzberg"),
    ("Morgan", "ooh nice! what do you pay if you don't mind me asking", "N4", "Question about rent"),
    ("Taylor", "like 900 a month", "S1", "Fact: rent is $900/month for 2br"),
    ("Morgan", "NINE HUNDRED DOLLARS", "N3", "Shock at rent"),
    ("Morgan", "for a 2 bedroom??", "N3", "Continued shock"),
    ("Morgan", "in a major city??", "N3", "Escalating disbelief"),
    ("Taylor", "haha yep. euros technically but yeah basically 900", "B1", "Context-enabling: currency clarification"),
    ("Morgan", "dude our 2-bedroom in Portland is like $2,200", "S1", "Fact: Morgan's rent is $2,200"),
    ("Taylor", "WHAT", "N3", "Shock reaction"),
    ("Taylor", "that's insane", "N3", "Follow-up shock"),
    ("Morgan", "I know. and Portland isn't even that expensive compared to like SF or NYC", "B1", "Context-enabling: cost comparison"),
    ("Taylor", "honestly that's one of the best things about Berlin. rent is still kind of reasonable", "S2", "Opinion: Berlin rent is reasonable"),
    ("Taylor", "like compared to other European capitals it's cheap", "B1", "Context-enabling: European comparison"),
    ("Morgan", "I'm moving to Berlin confirmed", "B1", "Context-enabling: joking about Berlin move"),
    ("Taylor", "haha please do!! we'd be neighbors", "B1", "Context-enabling: proximity fantasy"),
    ("Morgan", "Chris would probably love it actually, tons of tech jobs there right?", "B2", "Casual embedded fact: Chris is in tech"),
    ("Taylor", "yeah there's a huge startup scene. lots of international companies too so English is fine", "S1", "Fact: Berlin has big startup scene"),
    ("Morgan", "hmm don't tempt me", "N1", "Playful filler"),
    ("Taylor", "haha", "N1", "Filler"),
    ("Taylor", "ok but real talk how are YOU doing", "N4", "Transition to deeper topic"),
    ("Taylor", "like not just baby stuff but like... you", "N4", "Clarifying the question"),
    ("Morgan", "honestly?", "N4", "Hesitation meta"),
    ("Morgan", "it's been really hard", "S5", "Emotional: struggling"),
    ("Taylor", "yeah?", "N1", "Gentle prompt"),
    ("Morgan", "like I love Olive more than anything obviously", "B1", "Context-enabling qualifier"),
    ("Morgan", "but postpartum has been rough. like really rough", "S5", "Emotional: postpartum struggle"),
    ("Morgan", "I feel so isolated sometimes. like I used to go out and see people every day at work and now it's just me and a baby in the house", "S5", "Emotional: isolation, loss of social contact"),
    ("Taylor", "oh Morgan", "N1", "Sympathetic filler"),
    ("Taylor", "I'm really sorry you're going through that", "N1", "Supportive filler"),
    ("Morgan", "and Chris is amazing but he went back to work after 6 weeks so it's mostly just me during the day", "S1", "Fact: Chris back at work after 6 weeks"),
    ("Morgan", "I was a graphic designer before this, did I tell you that?", "S1", "Fact: Morgan was a graphic designer"),
    ("Taylor", "no! that's awesome", "N1", "Supportive filler"),
    ("Morgan", "yeah I worked at this agency downtown. I loved it honestly", "S1", "Fact: worked at agency downtown"),
    ("Morgan", "I'm on parental leave right now but idk when or if I'll go back", "S1", "Fact: on parental leave, unsure about return"),
    ("Taylor", "that's a big decision", "N1", "Reflective filler"),
    ("Morgan", "yeah. like financially we kind of need me to go back eventually but also daycare is SO expensive", "S5", "Emotional: financial stress around childcare"),
    ("Morgan", "sorry I'm dumping all this on you", "N4", "Meta: self-conscious"),
    ("Taylor", "no no no don't be sorry!! this is exactly what friends are for", "N1", "Supportive filler"),
    ("Taylor", "honestly I feel bad that we lost touch for so long", "S5", "Emotional: regret about lost contact"),
    ("Morgan", "me too. life just gets so crazy", "S5", "Emotional: mutual regret"),
    ("Taylor", "well I'm here now. and I'm literally coming to visit you in February so", "B3", "Temporal marker: February visit callback"),
    ("Morgan", "haha that actually means so much. like genuinely", "S5", "Emotional: gratitude"),
    ("Taylor", "of course dude", "N1", "Reassuring filler"),
    ("Taylor", "ok lighter topic: have you discovered any good Portland restaurants I need to hit when I visit??", "N4", "Topic transition"),
    ("Morgan", "oh my god YES. ok there's this place called Canard that's incredible", "S1", "Fact: restaurant rec Canard"),
    ("Morgan", "and obviously you have to do Voodoo Doughnuts it's like a requirement", "S1", "Fact: Voodoo Doughnuts mention"),
    ("Taylor", "ooh I've heard of Voodoo!! yes absolutely on the list", "S2", "Preference: wants to visit Voodoo"),
    ("Morgan", "ok Olive's up, I hear the pterodactyl screech", "N2", "Farewell"),
    ("Taylor", "haha go get her!! talk soon", "N2", "Farewell"),
]
g3 = [30, 20, 25, 15, 20, 35, 25, 30, 15, 35, 15, 20, 25, 30, 45, 25, 35, 20, 15, 60, 20, 30,
      25, 20, 10, 8, 30, 25, 15, 20, 25, 30, 25, 40, 20, 30, 25, 20, 30, 90, 20, 30, 25, 20, 15,
      30, 40, 20, 25, 30, 25, 20, 30, 25, 30, 35, 20, 30, 25, 35, 25, 20, 30, 20, 25, 35, 25,
      30, 20]


# ═══════════════════════════════════════════════════════════════════════════
# SESSION 4: Baby photos + college friend group (65 msgs, 2025-11-16 Sun)
# Morgan sends baby photos. Reminiscing about friends.
# Taylor shares dating update: seeing Lukas.
# ═══════════════════════════════════════════════════════════════════════════
s4 = [
    ("Morgan", "ok I'm sending you baby photos whether you asked for them or not", "N4", "Meta opener"),
    ("Morgan", "prepare yourself", "N1", "Setup filler"),
    ("Morgan", "[photo of Olive in a tiny pumpkin outfit]", "B1", "Context-enabling: photo reference"),
    ("Taylor", "I AM NOT OK", "N3", "Overwhelmed reaction"),
    ("Taylor", "SHE IS THE CUTEST BABY I'VE EVER SEEN", "N3", "Gushing reaction"),
    ("Morgan", "right?? that's her Halloween outfit", "N5", "Echo + context"),
    ("Morgan", "she was a pumpkin obviously", "B2", "Casual embedded fact: Halloween costume"),
    ("Taylor", "omg the little hat", "N3", "Detail reaction"),
    ("Morgan", "[photo of Olive sleeping on Chris's chest]", "B1", "Context-enabling: photo reference"),
    ("Taylor", "ok that one actually made me tear up", "N3", "Emotional reaction"),
    ("Taylor", "she looks so peaceful", "B1", "Context-enabling: baby photo observation"),
    ("Morgan", "that's like the only time she's peaceful lol", "B2", "Casual embedded fact: Olive is fussy"),
    ("Morgan", "when she's on Chris she just melts. it's her favorite spot", "B2", "Casual embedded fact: Olive loves being on Chris"),
    ("Taylor", "that is so sweet. you guys made a really cute baby", "N1", "Compliment filler"),
    ("Morgan", "thank you haha we did our best", "N1", "Humble filler"),
    ("Morgan", "[photo of Olive with wide eyes looking at camera]", "B1", "Context-enabling: photo"),
    ("Taylor", "THOSE EYES omg", "N3", "Reaction to photo"),
    ("Taylor", "she has your eyes by the way", "B2", "Casual observation"),
    ("Morgan", "everyone says that!! Chris says she has his nose though", "B2", "Casual embedded fact: Olive's features"),
    ("Taylor", "haha", "N1", "Filler"),
    ("Taylor", "ok this is making me SO broody", "N1", "Humorous filler"),
    ("Taylor", "ok speaking of people we know", "N4", "Topic transition"),
    ("Taylor", "have you kept up with anyone from college??", "N4", "Meta question"),
    ("Morgan", "honestly not really", "N1", "Admission filler"),
    ("Morgan", "I talk to Kelsey sometimes on instagram", "S1", "Fact: in touch with Kelsey via Instagram"),
    ("Taylor", "oh Kelsey!! how is she", "N4", "Follow-up question"),
    ("Morgan", "she's in New York now, works in publishing", "S1", "Fact: Kelsey in NYC, works in publishing"),
    ("Taylor", "oh that's so her", "B1", "Context-enabling: Kelsey characterization"),
    ("Morgan", "right?? she always wanted to do that", "N5", "Echo agreement"),
    ("Taylor", "what about Dave?", "N4", "Question about friend"),
    ("Morgan", "Dave is still in Eugene lol", "S1", "Fact: Dave still in Eugene"),
    ("Taylor", "no way", "N3", "Surprise reaction"),
    ("Morgan", "yeah he's brewing beer now apparently", "S1", "Fact: Dave brews beer"),
    ("Morgan", "like at an actual brewery, not just in his garage anymore", "B1", "Context-enabling: Dave upgraded from home brewing"),
    ("Taylor", "haha remember his terrible homebrew from senior year?? that IPA that tasted like gasoline??", "B1", "Context-enabling: college memory"),
    ("Morgan", "oh god YES that was so bad", "N5", "Agreement with emphasis"),
    ("Morgan", "apparently he's actually good at it now", "B1", "Context-enabling: Dave improved"),
    ("Taylor", "good for him honestly", "N1", "Supportive filler"),
    ("Taylor", "what about Priya?? I always wondered what happened to her", "N4", "Question about friend"),
    ("Morgan", "oh!! Priya is doing a PhD at MIT", "S1", "Fact: Priya doing PhD at MIT"),
    ("Taylor", "WHAT", "N3", "Shock reaction"),
    ("Taylor", "of course she is", "B1", "Context-enabling: Priya characterization"),
    ("Morgan", "haha right? she was always the smartest person in every room", "B1", "Context-enabling: Priya characterization"),
    ("Taylor", "what's her PhD in?", "N4", "Follow-up question"),
    ("Morgan", "something with neuroscience I think? computational neuroscience?", "S1", "Fact: Priya's PhD in computational neuroscience"),
    ("Taylor", "that's so badass", "B1", "Context-enabling: reaction to Priya's PhD"),
    ("Morgan", "totally", "N5", "Echo agreement"),
    ("Morgan", "ok but you said the dating thing was 'interesting' — spill", "N4", "Circling back to earlier topic"),
    ("Taylor", "haha ok fine", "N4", "Relenting meta"),
    ("Taylor", "so I'm actually kind of seeing someone", "S4", "Life event: seeing someone"),
    ("Morgan", "WAIT WHAT", "N3", "Shock reaction"),
    ("Morgan", "you buried the lede!!", "N3", "Reaction"),
    ("Taylor", "haha it's still pretty new!!", "N1", "Qualifier filler"),
    ("Taylor", "his name is Lukas. he's German", "S1", "Fact: seeing someone named Lukas, he's German"),
    ("Morgan", "omg a German boy", "N3", "Excited reaction"),
    ("Taylor", "haha yeah. we met at this bar in Friedrichshain a few weeks ago", "S1", "Fact: met Lukas at bar in Friedrichshain"),
    ("Taylor", "he's a photographer. really chill, super funny", "S1", "Fact: Lukas is a photographer"),
    ("Morgan", "ok I love this for you", "N1", "Supportive filler"),
    ("Taylor", "thanks haha it's really early but he's cool", "N1", "Qualifier filler"),
    ("Taylor", "he doesn't speak great English so it's helping my German actually lol", "B2", "Casual embedded fact: Lukas helps with German"),
    ("Morgan", "that's honestly the best way to learn", "N5", "Echo agreement"),
    ("Taylor", "right?? very immersive haha", "N5", "Echo with humor"),
    ("Morgan", "haha ok I gotta go, bath time for the pumpkin", "N2", "Farewell"),
    ("Taylor", "give her a squeeze for me!! bye!!", "N2", "Farewell"),
    ("Morgan", "will do! byeee", "N2", "Farewell"),
]
g4 = [20, 25, 30, 15, 25, 20, 30, 25, 20, 15, 25, 30, 20, 25, 20, 30, 15, 25, 20, 30, 45, 25,
      30, 25, 30, 20, 25, 30, 20, 25, 20, 25, 20, 35, 30, 25, 20, 40, 25, 20, 15, 30, 25, 20,
      30, 20, 25, 30, 20, 30, 15, 20, 25, 20, 30, 25, 20, 25, 30, 25, 20, 30, 25, 20]


# ═══════════════════════════════════════════════════════════════════════════
# SESSION 5: Trip planning in detail (65 msgs, 2025-11-19 Wed)
# Flights, spare room, Portland activities. N26 layoff drama.
# ═══════════════════════════════════════════════════════════════════════════
s5 = [
    ("Taylor", "ok so I've been looking at flights", "N4", "Meta opener"),
    ("Morgan", "ooh yay!!", "N3", "Excited reaction"),
    ("Taylor", "Portland flights from Berlin are actually not terrible if I connect through Amsterdam", "S1", "Fact: routing through Amsterdam"),
    ("Morgan", "oh nice how much are they?", "N4", "Follow-up question"),
    ("Taylor", "like 700 euros round trip which honestly isn't bad for transatlantic in February", "S1", "Fact: flight cost ~700 euros"),
    ("Morgan", "yeah that's really reasonable", "N5", "Agreement echo"),
    ("Taylor", "so I'm thinking February 14th to the 21st, that work?", "S1", "Fact: trip dates Feb 14-21"),
    ("Morgan", "Valentine's Day arrival lol", "N1", "Humorous observation filler"),
    ("Taylor", "haha oh god I didn't even think about that", "N1", "Amused filler"),
    ("Taylor", "sorry Lukas lol", "B2", "Casual embedded: Valentine's conflict with Lukas"),
    ("Morgan", "haha", "N1", "Filler"),
    ("Morgan", "but yes those dates work perfectly!!", "S3", "Decision: confirming dates work"),
    ("Morgan", "you should stay with us honestly", "S3", "Decision: offering to host"),
    ("Taylor", "wait really?? I was just gonna get an Airbnb", "N3", "Surprised reaction"),
    ("Morgan", "no way, we have a spare room", "S1", "Fact: Morgan has a spare room"),
    ("Morgan", "it's technically Olive's nursery but she sleeps in our room still anyway lol", "B2", "Casual fact: Olive sleeps in parents' room"),
    ("Taylor", "omg are you sure?? I don't want to be in the way with the baby and everything", "N4", "Clarification question"),
    ("Morgan", "dude you would literally be doing me a favor. I need adult interaction so badly", "S5", "Emotional: craving adult interaction"),
    ("Taylor", "haha ok done. I'm staying with you guys", "S3", "Decision: staying with Morgan"),
    ("Morgan", "yay!! Chris is gonna be pumped too, I've been telling him all about you", "B2", "Casual fact: Morgan talks about Taylor to Chris"),
    ("Taylor", "aww that's sweet", "N1", "Warm filler"),
    ("Taylor", "ok what should we do while I'm there?? I haven't been to Portland in forever", "N4", "Planning question"),
    ("Morgan", "have you ever actually been to Portland?", "N4", "Clarifying question"),
    ("Taylor", "hmm actually I don't think so?? just drove through once on the way to Seattle", "B2", "Casual fact: never properly visited Portland"),
    ("Morgan", "oh ok so you NEED to see everything then", "N4", "Planning meta"),
    ("Morgan", "Powell's Books is a must. it's like the biggest bookstore in the world or something", "S1", "Fact: Powell's Books recommendation"),
    ("Taylor", "omg I've heard of Powell's!! yes absolutely", "S2", "Preference: wants to see Powell's"),
    ("Morgan", "and Voodoo Doughnuts obviously, we talked about that", "B1", "Context-enabling: callback to earlier conversation"),
    ("Taylor", "yes!! the doughnut place", "N5", "Echo agreement"),
    ("Morgan", "there's also this amazing Japanese garden if it's not too rainy", "S1", "Fact: Portland Japanese Garden"),
    ("Taylor", "oh that sounds beautiful", "N1", "Positive filler"),
    ("Morgan", "speaking of which I should warn you", "N4", "Setup meta"),
    ("Morgan", "February in Portland is like... rain. just constant rain", "S1", "Fact: Portland February weather is rainy"),
    ("Taylor", "haha worse than Berlin? because Berlin in winter is pretty gray and miserable", "B1", "Context-enabling: Berlin winter comparison"),
    ("Morgan", "oh Berlin has nothing on Portland rain trust me", "S2", "Opinion: Portland rain worse"),
    ("Taylor", "great can't wait lol", "N1", "Sarcastic filler"),
    ("Morgan", "haha pack layers. and a raincoat. actually pack two raincoats", "N1", "Humorous filler"),
    ("Taylor", "noted haha", "N1", "Agreement filler"),
    ("Taylor", "ok totally different topic", "N4", "Topic transition"),
    ("Taylor", "things at work are kind of weird right now", "S5", "Emotional: work stress"),
    ("Morgan", "oh no what's going on?", "N4", "Follow-up question"),
    ("Taylor", "there are rumors that N26 might be doing layoffs", "S1", "Fact: N26 potential layoffs"),
    ("Morgan", "oh shit", "N3", "Concerned reaction"),
    ("Taylor", "yeah like nothing confirmed but there was this all-hands meeting last week that was very 'we need to be more efficient'", "B1", "Context-enabling: all-hands detail"),
    ("Taylor", "which is always code for layoffs lol", "B1", "Context-enabling: interpreting corporate speak"),
    ("Morgan", "ugh that's so stressful. do you think you're safe?", "N4", "Follow-up question"),
    ("Taylor", "honestly I think so? my team is pretty essential and I've only been here 3 years so I'm not expensive", "S2", "Self-assessment: probably safe from layoffs"),
    ("Taylor", "but like you never really know", "B1", "Context-enabling: layoff uncertainty"),
    ("Morgan", "yeah that's the worst part, the uncertainty", "N5", "Echo agreement"),
    ("Taylor", "exactly. and like what would I even do if I got laid off?? I'm in Germany on a work visa", "S1", "Fact: in Germany on work visa"),
    ("Morgan", "oh wow I didn't think about the visa thing", "N3", "Realization reaction"),
    ("Taylor", "yeah if I lose my job I'd have like 3 months to find a new one or leave the country basically", "S1", "Fact: 3 months to find new job or leave"),
    ("Morgan", "that's terrifying", "N3", "Emphatic reaction"),
    ("Taylor", "it is lol but I'm trying not to spiral about it", "S5", "Emotional: trying not to worry"),
    ("Taylor", "anyway it's probably fine. just tech industry being tech industry", "B1", "Context-enabling: tech layoff normalization"),
    ("Morgan", "well if worst comes to worst you can just move to Portland and be my live-in nanny", "B1", "Context-enabling: humorous backup plan"),
    ("Taylor", "haha honestly at this point? maybe", "N1", "Playing along filler"),
    ("Morgan", "I'm only half kidding", "N1", "Half-serious filler"),
    ("Taylor", "haha", "N1", "Filler"),
    ("Taylor", "ok I'm gonna keep looking at flights and figure out the dates for sure", "S3", "Decision: finalizing flight research"),
    ("Morgan", "yes please!! I'm so excited", "N3", "Enthusiastic reaction"),
    ("Taylor", "me too!! ok talk later this week, gotta jump on a call", "N2", "Farewell"),
    ("Morgan", "go go! talk soon!", "N2", "Farewell"),
    ("Taylor", "bye!!", "N2", "Farewell"),
    ("Morgan", "byeee", "N2", "Farewell"),
]
g5 = [30, 25, 35, 25, 30, 20, 25, 20, 30, 25, 20, 30, 20, 30, 20, 25, 35, 20, 30, 25, 40, 25,
      30, 45, 20, 25, 20, 25, 30, 25, 60, 20, 30, 25, 30, 25, 20, 90, 25, 30, 20, 15, 30, 25,
      30, 25, 20, 25, 30, 20, 25, 20, 25, 20, 25, 30, 25, 20, 30, 25, 25, 20, 25, 20]


# ═══════════════════════════════════════════════════════════════════════════
# SESSION 6: Trip confirmed + deep reflection (65 msgs, 2025-11-22 Sat)
# Flight booked, concrete plans. Olive milestone. Homesickness.
# How much has changed since college. Reunion dinner idea.
# ═══════════════════════════════════════════════════════════════════════════
s6 = [
    ("Taylor", "I BOOKED THE FLIGHT", "S3", "Decision: flight booked"),
    ("Morgan", "AHHHHH", "N3", "Excited scream"),
    ("Morgan", "YESSS", "N3", "Continued excitement"),
    ("Taylor", "February 14th arriving Portland at 4:15pm, leaving the 21st at 10am", "S1", "Fact: arrival/departure details"),
    ("Morgan", "this is happening this is actually happening", "N3", "Excitement reaction"),
    ("Taylor", "I connect through Amsterdam like I said, it's like a 14 hour journey total", "S1", "Fact: 14 hour journey, Amsterdam connection"),
    ("Morgan", "oof that's long but worth it", "N1", "Sympathetic filler"),
    ("Taylor", "ok so we need to actually plan this out", "N4", "Planning meta"),
    ("Morgan", "ok let me think", "N4", "Thinking meta"),
    ("Morgan", "first of all you're staying with us, that's settled", "S3", "Decision: confirmed hosting"),
    ("Taylor", "yes ma'am", "N5", "Agreement echo"),
    ("Morgan", "Powell's Books we said, that's like a half day thing honestly because it's MASSIVE", "S1", "Fact: Powell's is a half-day visit"),
    ("Taylor", "perfect. I could spend a whole day in a bookstore", "S2", "Preference: loves bookstores"),
    ("Morgan", "Voodoo Doughnuts, we'll do that one morning", "S3", "Decision: Voodoo morning trip"),
    ("Taylor", "yes!!", "N5", "Agreement"),
    ("Morgan", "there's also this neighborhood called Alberta Street that has amazing food and shops", "S1", "Fact: Alberta Street rec"),
    ("Morgan", "and if the weather cooperates we should do a hike in Forest Park", "S1", "Fact: Forest Park hike suggestion"),
    ("Taylor", "oh yes I've been dying for good hiking. Berlin is flat as a pancake", "S2", "Preference: wants hiking, Berlin is flat"),
    ("Morgan", "well Portland will fix that. we've got hills AND mountains", "B1", "Context-enabling: Portland geography"),
    ("Taylor", "I want to see Mount Hood!!", "S2", "Preference: wants to see Mount Hood"),
    ("Morgan", "oh we can definitely drive out there. it's like an hour and a half", "S1", "Fact: Mt Hood is 1.5 hrs from Portland"),
    ("Taylor", "oh also!! can we do that restaurant you mentioned? Canard?", "B1", "Context-enabling: callback to earlier rec"),
    ("Morgan", "YES absolutely. I'll make a reservation", "S3", "Decision: Canard reservation"),
    ("Morgan", "oh oh oh also I have an idea", "N4", "Idea intro"),
    ("Morgan", "what if we try to get some of the old crew together for dinner while you're here", "S3", "Decision: proposing reunion dinner"),
    ("Taylor", "omg YES. like a mini reunion??", "N3", "Enthusiastic reaction"),
    ("Morgan", "yeah!! like whoever can make it", "N4", "Elaborating"),
    ("Morgan", "Kelsey could probably fly down from New York for a weekend", "B1", "Context-enabling: Kelsey logistics"),
    ("Taylor", "and Dave is still in Eugene so that's only like a 2 hour drive", "B1", "Context-enabling: Dave logistics"),
    ("Morgan", "exactly!! Priya might be harder since she's at MIT but worth asking", "B1", "Context-enabling: Priya logistics"),
    ("Taylor", "dude this would be incredible. when's the last time we were all in one place", "N4", "Rhetorical question"),
    ("Morgan", "probably graduation?? that's like 7 years ago", "B3", "Temporal marker: 7 years since graduation"),
    ("Taylor", "wow. seven years", "N3", "Reflective reaction"),
    ("Morgan", "I know right?? time is a scam", "N5", "Echo agreement"),
    ("Morgan", "oh speaking of time passing — Olive update", "N4", "Topic transition"),
    ("Morgan", "she's almost 4 months now and she's starting to grab things", "S1", "Fact: Olive almost 4 months, grabbing things"),
    ("Taylor", "omg that's a big milestone right??", "N4", "Question about milestone"),
    ("Morgan", "yes!! like she reached out and grabbed Chris's finger yesterday and just held on", "S4", "Life event: Olive grabbing milestone"),
    ("Morgan", "I full on sobbed", "S5", "Emotional: moved by milestone"),
    ("Taylor", "ok now I'M crying", "N3", "Emotional reaction"),
    ("Morgan", "haha join the club. I cry like 4 times a day now", "B2", "Casual embedded fact about postpartum"),
    ("Taylor", "ok can I be honest about something", "N4", "Vulnerability setup"),
    ("Morgan", "of course", "N1", "Encouraging filler"),
    ("Taylor", "I love Berlin. like I really do. the city is incredible and my job is great", "B1", "Context-enabling qualifier"),
    ("Taylor", "but sometimes I get so homesick it physically hurts", "S5", "Emotional: deep homesickness"),
    ("Morgan", "oh Taylor", "N1", "Sympathetic filler"),
    ("Taylor", "like I miss dumb stuff. Target runs. Mexican food that's actually good. English everywhere", "S5", "Emotional: specific things missed about US"),
    ("Taylor", "and I miss having people who really KNOW me, you know?", "S5", "Emotional: missing deep connections"),
    ("Morgan", "yeah I get that. even though I'm in the US I feel that way too sometimes since having Olive", "S5", "Emotional: relates to isolation"),
    ("Morgan", "like my whole world shrank to this tiny apartment and a baby", "S5", "Emotional: world getting smaller"),
    ("Taylor", "god we're both kind of lonely huh", "S5", "Emotional: shared loneliness recognition"),
    ("Morgan", "which is why this visit is gonna be so good", "B1", "Context-enabling: visit importance"),
    ("Taylor", "honestly yeah. I need this", "S5", "Emotional: needs the visit"),
    ("Morgan", "me too. more than you know", "S5", "Emotional: mutual need"),
    ("Taylor", "ok I'm getting emotional and I have to go meet Lukas for brunch lol", "N2", "Farewell with Lukas callback"),
    ("Morgan", "haha go go! tell Lukas I say hi even though we've never met", "N2", "Farewell"),
    ("Taylor", "haha he'll be so confused but I'll explain", "N1", "Amused filler"),
    ("Morgan", "haha", "N1", "Filler"),
    ("Taylor", "ok for real though. February cannot come fast enough", "S5", "Emotional: anticipation"),
    ("Morgan", "seriously. I'm counting the days", "S5", "Emotional: anticipation"),
    ("Taylor", "love you Morgan. I'm so glad I texted you that random Saturday", "S5", "Emotional: gratitude"),
    ("Morgan", "love you too!! I'm so glad you did", "S5", "Emotional: mutual gratitude"),
    ("Taylor", "ok BYE for real this time", "N2", "Farewell"),
    ("Morgan", "bye!! go eat brunch!!", "N2", "Farewell"),
    ("Taylor", "already out the door!!", "N2", "Final farewell"),
]
g6 = [20, 15, 30, 20, 30, 25, 40, 20, 25, 30, 20, 25, 20, 25, 30, 25, 30, 25, 30, 20, 25, 35,
      25, 40, 20, 25, 20, 25, 30, 35, 20, 25, 20, 60, 20, 25, 30, 20, 25, 20, 90, 20, 25, 30,
      25, 30, 25, 30, 25, 30, 20, 25, 30, 20, 25, 30, 20, 15, 25, 30, 20, 25, 20, 25]


# ═══════════════════════════════════════════════════════════════════════════
# BUILD AND WRITE
# ═══════════════════════════════════════════════════════════════════════════
sessions = [
    (1, "2025-11-08", "2025-11-08T19:30:00", g1, s1),
    (2, "2025-11-10", "2025-11-10T14:00:00", g2, s2),
    (3, "2025-11-13", "2025-11-13T11:15:00", g3, s3),
    (4, "2025-11-16", "2025-11-16T10:00:00", g4, s4),
    (5, "2025-11-19", "2025-11-19T20:00:00", g5, s5),
    (6, "2025-11-22", "2025-11-22T09:30:00", g6, s6),
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
    "conversation_id": "conv5",
    "speakers": {
        "Taylor": "Expat in Berlin, product manager at N26, from US, went to University of Oregon",
        "Morgan": "New parent in Portland OR, graphic designer on parental leave, baby Olive (3mo), partner Chris",
    },
    "date_range": {"start": "2025-11-08", "end": "2025-11-22"},
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

out_path = Path("benchmarks/gate_eval/datasets/gate_benchmark_conv5.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"Wrote {len(all_messages)} messages to {out_path}")
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
