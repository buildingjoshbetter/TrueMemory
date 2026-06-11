#!/usr/bin/env python3
"""Phase 3: Conv 2 — Maria & Sam (parent/adult child), 400 messages, 10 sessions."""

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

def build(sess_num: int, date: str, start: str, gaps: list[int], msgs: list[tuple], conv="conv2") -> list[dict]:
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
    """Generate n-1 gaps from a cycling pattern."""
    if pattern is None:
        pattern = [45, 30, 60, 20, 90, 30, 45, 120, 30, 60, 20, 45, 30, 90, 60, 30, 45, 20, 120, 30]
    return [pattern[i % len(pattern)] for i in range(n - 1)]

# ═══════════════════════════════════════════════════════════════════════════
# CONVERSATION 2: Maria & Sam
# Maria = parent (more formal, complete sentences, asks questions, gives advice)
# Sam = adult child (casual, Gen-Z slang, brief, sometimes ignores questions)
# 10 sessions over 4 weeks, 400 messages, 45% noise target
# Noise=180, Signal=132, Borderline=88
# ═══════════════════════════════════════════════════════════════════════════

# Session 1: Check-in, Sam mentions college app stress (40 msgs, 2025-11-05)
s1 = [
    ("Maria", "Hi sweetie, how's your day going?", "N2", "Greeting"),
    ("Sam", "hey mom", "N2", "Greeting response"),
    ("Sam", "its fine", "N1", "Minimal filler"),
    ("Maria", "Just fine? What's going on?", "N1", "Probing question"),
    ("Sam", "nothing lol just tired", "N1", "Deflection filler"),
    ("Maria", "Are you getting enough sleep? You need at least 8 hours.", "N1", "Unsolicited health advice filler"),
    ("Sam", "mom", "N1", "Exasperated filler"),
    ("Maria", "I'm just saying! Your father never slept enough in college and he was always sick.", "N1", "Anecdotal filler"),
    ("Sam", "ok lol", "N1", "Dismissive filler"),
    ("Maria", "How are your classes this semester? You never tell me anything.", "N4", "Meta-complaint transition"),
    ("Sam", "they're good. I'm taking 5 classes this semester which is a lot", "S1", "Course load fact"),
    ("Maria", "Five classes?! That's too many. Which ones?", "N3", "Concerned reaction"),
    ("Sam", "intro to psych, calc 2, english comp, bio, and this elective on film studies", "S1", "Specific course list"),
    ("Maria", "Film studies? That sounds fun at least.", "N1", "Commentary"),
    ("Sam", "yeah its my favorite honestly. the professor is amazing", "S2", "Preference: favorite class"),
    ("Maria", "What's the professor's name?", "N1", "Question"),
    ("Sam", "Dr. Okafor. she did her PhD at NYU on like documentary filmmaking", "S1", "Professor name + background"),
    ("Maria", "Oh that's wonderful. Are you thinking about maybe minoring in film?", "N1", "Follow-up question"),
    ("Sam", "idk maybe. haven't really thought about it", "N1", "Noncommittal filler"),
    ("Maria", "Well you should think about it! You've always loved movies.", "N1", "Advice filler"),
    ("Sam", "yeah", "N1", "Agreement filler"),
    ("Maria", "Oh, I wanted to ask you about Thanksgiving. Are you coming home?", "S1", "Holiday plans question"),
    ("Sam", "yeah I was gonna take the train home Wednesday night", "S3", "Travel plan commitment"),
    ("Maria", "Oh good. Aunt Rosa is coming too, and Uncle Miguel.", "S1", "Guest list: Rosa and Miguel"),
    ("Sam", "nice", "N1", "Minimal filler"),
    ("Maria", "I'm making the tamales you like. The ones with the green chile.", "S1", "Food plan: specific tamale type"),
    ("Sam", "omg yesss", "N3", "Enthusiastic reaction"),
    ("Maria", "Ha! I knew that would get a reaction.", "N1", "Commentary"),
    ("Sam", "lol u know me", "N1", "Filler"),
    ("Maria", "I do. Oh, before I forget — have you started working on your grad school applications?", "S1", "Application status question + fact Sam is applying"),
    ("Sam", "ugh mom not yet. the deadline isn't until January", "B3", "Temporal: application deadline month"),
    ("Maria", "January is closer than you think, Sam.", "N1", "Pressure filler"),
    ("Sam", "I know I know. I'll start soon", "N1", "Deflection"),
    ("Maria", "Which programs are you looking at? We talked about this.", "N4", "Meta-reference to past conversation"),
    ("Sam", "mostly clinical psych programs. Berkeley, UCLA, UCSF", "S1", "Target schools list"),
    ("Maria", "Those are all wonderful schools. Have you talked to Professor Chen about your letter of recommendation?", "S1", "Faculty name + recommendation context"),
    ("Sam", "yeah she said she'd write me one. I need to send her my personal statement tho", "S1", "Rec letter status"),
    ("Maria", "That's great! When is she expecting it?", "N1", "Question"),
    ("Sam", "she said by december 1st ideally", "B3", "Temporal: personal statement due date"),
    ("Maria", "OK well that's coming up fast. Love you sweetie. Call me this weekend?", "N2", "Farewell + request"),
    ("Sam", "love u too. yeah maybe sunday", "N2", "Farewell + temporal"),
]
# 41 msgs, need 40. Let me remove one. Remove msg 8 (Maria's anecdotal filler)
# Actually that's already 41. Let me recount... 1-41 is 41 lines. Hmm.
# Let me remove message 28: "Ha! I knew that would get a reaction." — pure filler
# New count: 40 msgs
s1 = [
    ("Maria", "Hi sweetie, how's your day going?", "N2", "Greeting"),
    ("Sam", "hey mom", "N2", "Greeting response"),
    ("Sam", "its fine", "N1", "Minimal filler"),
    ("Maria", "Just fine? What's going on?", "N1", "Probing question"),
    ("Sam", "nothing lol just tired", "N1", "Deflection filler"),
    ("Maria", "Are you getting enough sleep? You need at least 8 hours.", "N1", "Unsolicited health advice"),
    ("Sam", "mom", "N1", "Exasperated filler"),
    ("Maria", "I'm just saying! Your father never slept enough in college either.", "N1", "Anecdotal filler"),
    ("Sam", "ok lol", "N1", "Dismissive filler"),
    ("Maria", "How are your classes? You never tell me anything.", "N4", "Meta-complaint transition"),
    ("Sam", "they're good. taking 5 this semester which is a lot", "S1", "Course load fact: 5 classes"),
    ("Maria", "Five?! That's too many. Which ones?", "N3", "Concerned reaction"),
    ("Sam", "intro to psych, calc 2, english comp, bio, and a film studies elective", "S1", "Specific course list"),
    ("Maria", "Film studies sounds fun at least.", "N1", "Commentary"),
    ("Sam", "yeah its my fav honestly. the professor is incredible", "S2", "Preference: favorite class"),
    ("Maria", "What's the professor's name?", "N1", "Question"),
    ("Sam", "Dr. Okafor. she did her PhD at NYU on documentary filmmaking", "S1", "Professor name and background"),
    ("Maria", "Oh wonderful. Are you thinking of minoring in film maybe?", "N1", "Suggestion"),
    ("Sam", "idk maybe. haven't thought about it", "N1", "Noncommittal filler"),
    ("Maria", "You should! You've always loved movies.", "N1", "Advice filler"),
    ("Sam", "yeah", "N1", "Agreement filler"),
    ("Maria", "Anyway, I wanted to ask about Thanksgiving. Are you coming home?", "S1", "Holiday plan inquiry"),
    ("Sam", "yeah I'm taking the train home Wednesday night", "S3", "Travel commitment + method"),
    ("Maria", "Oh good! Aunt Rosa and Uncle Miguel are coming too.", "S1", "Thanksgiving guests: Rosa and Miguel"),
    ("Sam", "nice", "N1", "Minimal filler"),
    ("Maria", "I'm making the green chile tamales you love.", "S1", "Specific food plan"),
    ("Sam", "omg yesss 🙌", "N3", "Enthusiastic food reaction"),
    ("Sam", "u always know the way to my heart lol", "N1", "Filler"),
    ("Maria", "Ha! I do know. Oh before I forget — have you started your grad school applications?", "S1", "Application topic + Sam is applying"),
    ("Sam", "ugh not yet mom. deadline isn't til January", "B3", "Temporal: January deadline"),
    ("Maria", "January is closer than you think.", "N1", "Pressure filler"),
    ("Sam", "I knowww. I'll start soon I promise", "N1", "Deflection"),
    ("Maria", "Which programs are you applying to?", "N4", "Follow-up question"),
    ("Sam", "clinical psych programs mostly. Berkeley, UCLA, UCSF", "S1", "Target school list"),
    ("Maria", "Those are excellent. Have you asked Professor Chen about your recommendation letter?", "S1", "Faculty reference: Professor Chen"),
    ("Sam", "yeah she agreed to write one. just need to send her my personal statement first", "S1", "Rec letter status detail"),
    ("Maria", "When does she need it by?", "N1", "Question"),
    ("Sam", "she said December 1st ideally", "B3", "Temporal: statement deadline"),
    ("Maria", "OK that's very soon! Love you sweetie. Call me Sunday?", "N2", "Farewell + request"),
    ("Sam", "love u too ❤️ yeah I'll try", "N2", "Farewell"),
]
g1 = gaps_for(40, [30, 60, 20, 45, 30, 90, 20, 60, 30, 120, 45, 30, 20, 60, 30, 45, 90, 30, 20, 60])


# Session 2: Maria's health scare (45 msgs, 2025-11-10)
s2 = [
    ("Maria", "Sam, can you call me when you get a chance?", "N2", "Request greeting"),
    ("Sam", "whats up? in class rn can u text", "N1", "Response filler"),
    ("Maria", "OK. I went to the doctor today.", "S1", "Doctor visit fact"),
    ("Sam", "oh is everything ok??", "N3", "Concerned reaction"),
    ("Maria", "Well, they found something in my bloodwork that they want to check further.", "S1", "Medical finding"),
    ("Sam", "wait what do u mean", "N3", "Concerned reaction"),
    ("Maria", "My glucose levels were high. The doctor said I might have prediabetes.", "S1", "Diagnosis: prediabetes concern"),
    ("Sam", "oh no mom", "N3", "Worried reaction"),
    ("Maria", "Don't panic, it's not confirmed yet. I have to go back for more tests next week.", "S1", "Follow-up appointment"),
    ("Sam", "ok but like are u feeling ok??", "N1", "Concern question"),
    ("Maria", "I feel fine honestly. I had no symptoms at all. It was just the routine bloodwork.", "S1", "No symptoms, routine discovery"),
    ("Sam", "ok good", "N1", "Relief filler"),
    ("Maria", "The doctor said if it IS prediabetes, I'll need to change my diet and exercise more.", "S1", "Treatment plan if confirmed"),
    ("Sam", "ok well thats manageable right?", "N1", "Reassurance attempt"),
    ("Maria", "Yes, absolutely. It's very common. I just wanted you to know.", "N1", "Reassurance filler"),
    ("Sam", "thx for telling me mom", "N1", "Gratitude filler"),
    ("Maria", "Of course. I didn't want to worry you but I also don't want to keep secrets.", "S5", "Emotional: torn about sharing health news"),
    ("Sam", "no I'm glad u told me. when are the follow up tests?", "N1", "Question"),
    ("Maria", "Next Thursday the 14th.", "S1", "Follow-up test date"),
    ("Sam", "ok keep me posted pls", "N1", "Request filler"),
    ("Maria", "I will. Your father is taking me.", "S1", "Dad is accompanying, fact"),
    ("Sam", "good. tell dad I said hi", "N1", "Filler"),
    ("Maria", "I will. How are you doing with everything? School OK?", "N4", "Topic transition"),
    ("Sam", "yeah its fine. I started working on my personal statement actually", "S1", "Application progress"),
    ("Sam", "figured I should stop procrastinating lol", "N1", "Commentary"),
    ("Maria", "Oh that's wonderful! What are you writing about?", "N3", "Enthusiastic reaction"),
    ("Sam", "my experience volunteering at the youth crisis center last summer. and how it made me want to do clinical psych", "S1", "Personal statement topic + motivation"),
    ("Maria", "That's perfect. That was such meaningful work you did there.", "N5", "Affirming echo"),
    ("Sam", "yeah it really changed my perspective on a lot of things honestly", "S5", "Emotional reflection on volunteer experience"),
    ("Maria", "How long does the statement need to be?", "N1", "Question"),
    ("Sam", "like 500-750 words. its not super long but I want it to be good", "S1", "Statement length requirement"),
    ("Maria", "Do you want me to read it when you're done? Your father could too — he wrote his own for med school.", "B1", "Context: dad went to med school"),
    ("Sam", "yeah that would be great actually", "N1", "Agreement"),
    ("Maria", "OK just send it whenever. No pressure.", "N1", "Filler"),
    ("Sam", "ok cool", "N1", "Filler"),
    ("Maria", "Oh and Sam?", "N4", "Attention getter"),
    ("Sam", "yeah?", "N1", "Response"),
    ("Maria", "I'm really proud of you. The applications, the volunteering, all of it.", "S5", "Emotional: parental pride"),
    ("Sam", "🥺 thx mom", "N1", "Emotional reaction filler"),
    ("Maria", "OK get back to class. Love you.", "N2", "Farewell"),
    ("Sam", "love u ❤️", "N2", "Farewell"),
    ("Maria", "Oh wait — one more thing. Are you eating enough? You looked thin on FaceTime last week.", "N1", "Classic parent concern"),
    ("Sam", "MOM", "N1", "Exasperated filler"),
    ("Maria", "OK OK going now 😂", "N2", "Farewell for real"),
    ("Sam", "bye lol", "N2", "Farewell"),
]
g2 = gaps_for(45, [30, 90, 30, 20, 120, 30, 45, 20, 60, 30, 90, 30, 20, 45, 60, 30, 120, 30, 45, 20])


# Session 3: Thanksgiving prep + family reunion planning (40 msgs, 2025-11-18)
s3 = [
    ("Maria", "Sam are you still coming home Wednesday night?", "N1", "Logistical question"),
    ("Sam", "yeah getting the 6pm train from Davis", "S1", "Specific train time + location"),
    ("Maria", "OK good. Dad will pick you up at the station.", "S1", "Dad picking up, logistics"),
    ("Sam", "cool thx", "N1", "Filler"),
    ("Maria", "I have so much cooking to do this week. The tamales alone take an entire day.", "N1", "Venting filler"),
    ("Sam", "lol I believe it", "N1", "Filler"),
    ("Maria", "Aunt Rosa called today. She's bringing her tres leches cake.", "S1", "Rosa's contribution: tres leches"),
    ("Sam", "oh yessss", "N3", "Excited food reaction"),
    ("Maria", "I know, it's so good. Uncle Miguel is driving them from Fresno.", "S1", "Miguel driving from Fresno"),
    ("Sam", "wait they're driving? that's like 3 hours", "B3", "Temporal: drive time"),
    ("Maria", "Yes but Miguel hates flying. You know how he is.", "B2", "Casual fact: Miguel dislikes flying"),
    ("Maria", "Oh and your cousin Sofia is coming too! She just got back from her semester in Barcelona.", "S1", "Sofia attending + Barcelona semester"),
    ("Sam", "oh nice I haven't seen Sofia in forever", "N1", "Filler"),
    ("Maria", "I think since the reunion two years ago in Monterey.", "B3", "Temporal: last time they met"),
    ("Sam", "yeah sounds right", "N5", "Agreement echo"),
    ("Maria", "She apparently learned to make paella while she was in Spain and wants to cook for us.", "S1", "Sofia's cooking plan"),
    ("Sam", "oh that's cool", "N1", "Filler"),
    ("Maria", "By the way, I got my test results back.", "S1", "Health update setup"),
    ("Sam", "wait already?? what did they say", "N3", "Urgent reaction"),
    ("Maria", "It IS prediabetes. But the doctor said it's very manageable with diet and exercise.", "S4", "Life event: prediabetes confirmed"),
    ("Sam", "oh no mom. are you ok?", "N3", "Concerned reaction"),
    ("Maria", "Yes honey I'm fine. Really. The doctor put me on a meal plan and I need to walk 30 minutes a day.", "S1", "Treatment: meal plan + exercise requirement"),
    ("Sam", "ok well thats not too bad", "N1", "Reassurance attempt"),
    ("Maria", "No it's not. Your father has been very supportive. He's walking with me every morning now.", "S1", "Dad walking with Maria daily"),
    ("Sam", "aw thats sweet", "N1", "Filler"),
    ("Maria", "Yes well he could use the exercise too 😂", "N1", "Humor filler"),
    ("Sam", "lmao", "N1", "Laughter filler"),
    ("Maria", "The main thing I need to cut down on is sugar and refined carbs.", "S1", "Dietary restriction detail"),
    ("Sam", "wait but what about the tamales lol", "B2", "Casual callback: tamale plan vs diet"),
    ("Maria", "Ha! I can still have SOME tamales. Just not ten like last year.", "N1", "Humor filler"),
    ("Sam", "lol fair", "N1", "Filler"),
    ("Maria", "Anyway, I also wanted to talk about the family reunion.", "N4", "Topic transition"),
    ("Sam", "wait theres a reunion?", "N3", "Surprised reaction"),
    ("Maria", "Yes! Abuela's 80th birthday is in March. We're doing a big family gathering.", "S4", "Life event: grandmother's 80th + party"),
    ("Sam", "oh wow 80!! where is it gonna be", "N3", "Reaction"),
    ("Maria", "We're still deciding. Either at Tía Carmen's house in San Jose or we rent a hall.", "S1", "Venue options + Tía Carmen's location"),
    ("Sam", "either sounds good. I can prob come if its a weekend", "B3", "Temporal: weekend availability"),
    ("Maria", "It will be Saturday March 15th most likely.", "S1", "Reunion date"),
    ("Maria", "We're expecting about 40-50 people from all over.", "S1", "Expected attendance"),
    ("Sam", "oh wow big party. ok lmk when its confirmed and I'll block it off", "S3", "Commitment to attend"),
]
g3 = gaps_for(40, [60, 30, 20, 90, 45, 30, 120, 30, 20, 60, 30, 45, 30, 60, 20, 90, 30, 120, 45, 30])


# Session 4: Post-Thanksgiving, Sam's pottery hobby (35 msgs, 2025-11-25)
s4 = [
    ("Sam", "yo mom that was the best thanksgiving", "N2", "Post-holiday greeting"),
    ("Maria", "It really was! I'm so glad everyone could make it.", "N1", "Agreement filler"),
    ("Sam", "sofias paella was actually so good", "S2", "Food preference: Sofia's paella"),
    ("Maria", "It was! She's become quite the cook in Spain.", "N5", "Affirming echo"),
    ("Sam", "also aunt rosas tres leches omg", "S2", "Food preference: Rosa's cake"),
    ("Maria", "Ha I know. She gave me the recipe actually.", "B2", "Casual fact: got the recipe"),
    ("Sam", "oh nice. hey so I wanted to tell u something", "N4", "Topic transition"),
    ("Maria", "What's up?", "N1", "Response"),
    ("Sam", "so my friend Jake has been doing pottery at this studio near campus", "B2", "Casual fact: friend Jake, pottery connection"),
    ("Maria", "Oh?", "N1", "Prompt"),
    ("Sam", "and he took me to a class last week and I actually loved it??", "S1", "New hobby: pottery + temporal"),
    ("Maria", "Really? That's wonderful! What did you make?", "N3", "Enthusiastic reaction"),
    ("Sam", "lol just a really lopsided bowl. it was terrible but so fun", "S1", "First pottery piece"),
    ("Maria", "Oh I'd love to see it!", "N1", "Enthusiasm filler"),
    ("Sam", "I'll send u a pic when it comes out of the kiln", "N1", "Filler"),
    ("Sam", "but yeah I signed up for the 6-week beginner series. it's every Saturday morning", "S1", "Course commitment: 6 weeks, Saturdays"),
    ("Maria", "Oh that's exciting. How much does it cost?", "N1", "Question"),
    ("Sam", "$180 for the series. includes all the materials and kiln time", "S1", "Course cost + what's included"),
    ("Maria", "That's very reasonable. Do you need help paying for it?", "N1", "Offer filler"),
    ("Sam", "nah I got it from my work study money. thanks tho", "B2", "Casual fact: Sam has work-study income"),
    ("Maria", "OK just let me know. I think it's wonderful that you're trying something creative.", "N1", "Supportive filler"),
    ("Sam", "yeah idk theres something really calming about it. like u just focus on the clay and everything else disappears", "S5", "Emotional: pottery as stress relief"),
    ("Maria", "You deserve that. With five classes and applications, you need an outlet.", "N5", "Affirming echo"),
    ("Sam", "yeah exactly", "N1", "Agreement"),
    ("Maria", "Oh, did you send Professor Chen your personal statement yet?", "B1", "Callback to rec letter deadline"),
    ("Sam", "YES I sent it last night actually. she said she'd look at it this week", "S1", "Statement sent + timeline"),
    ("Maria", "Oh wonderful! How do you feel about it?", "N1", "Question"),
    ("Sam", "pretty good honestly. it felt really personal to write about the crisis center stuff", "S5", "Emotional: personal connection to statement content"),
    ("Maria", "I'm sure it's beautiful. You've always been good with words.", "N1", "Supportive filler"),
    ("Sam", "lol ok mom", "N1", "Deflection filler"),
    ("Maria", "What! It's true!", "N1", "Protest filler"),
    ("Sam", "haha ok ok. thanks ❤️", "N1", "Gratitude filler"),
    ("Maria", "How's everything else? Are you eating properly?", "N1", "Classic parent concern"),
    ("Sam", "yes mom I eat like 3 meals a day I promise", "N1", "Exasperated reassurance"),
    ("Maria", "OK good. Love you. Have a good rest of your week.", "N2", "Farewell"),
]
g4 = gaps_for(35, [30, 45, 20, 30, 60, 30, 90, 30, 20, 60, 45, 30, 120, 30, 20, 45, 60, 30, 90, 30])


# Session 5: Application stress + health update (40 msgs, 2025-12-02)
s5 = [
    ("Sam", "mom I'm freaking out", "S5", "Emotional: application stress"),
    ("Maria", "What happened?! Are you OK?", "N3", "Concerned reaction"),
    ("Sam", "yeah sorry I'm fine. its just the applications", "N1", "Clarification filler"),
    ("Maria", "OK you scared me. What about the applications?", "N1", "Follow-up"),
    ("Sam", "berkeley's app is due december 15th and I still haven't finished my research proposal", "S1", "Deadline fact: Berkeley Dec 15 + incomplete proposal"),
    ("Maria", "OK that's two weeks away. That's still time.", "B3", "Temporal: two weeks out"),
    ("Sam", "yeah but I also have finals coming up AND a bio paper due friday", "S1", "Competing deadlines: finals + bio paper"),
    ("Maria", "Take a deep breath. Make a list. Prioritize.", "N1", "Advice filler"),
    ("Sam", "I know I know. im just overwhelmed", "S5", "Emotional: overwhelmed"),
    ("Maria", "Can you talk to any of your professors about extensions?", "N1", "Suggestion"),
    ("Sam", "maybe. Dr. Okafor is pretty flexible usually", "B1", "Callback to film professor"),
    ("Maria", "Ask her. The worst she can say is no.", "N1", "Advice filler"),
    ("Sam", "true", "N1", "Agreement"),
    ("Sam", "also the research proposal is hard bc I want to write about adolescent trauma interventions but idk if thats too broad", "S1", "Research proposal topic"),
    ("Maria", "Have you talked to Professor Chen about it? She's your advisor right?", "B1", "Callback to professor"),
    ("Sam", "yeah shes my academic advisor. I have a meeting with her tomorrow actually", "S1", "Chen is advisor + meeting tomorrow"),
    ("Maria", "Perfect. She can help you narrow it down.", "N1", "Reassurance"),
    ("Sam", "yeah hopefully. how r u btw. hows the prediabetes stuff", "N4", "Topic transition"),
    ("Maria", "Good actually! I've been following the meal plan for two weeks now and I already feel better.", "S1", "Health progress: 2 weeks on plan"),
    ("Maria", "My blood sugar is trending down according to the glucose monitor.", "S1", "Measurable health improvement"),
    ("Sam", "oh thats great mom!!", "N3", "Enthusiastic reaction"),
    ("Maria", "Yes and your father and I walk every morning now. We do about 2 miles along the river trail.", "S1", "Exercise routine detail: 2 miles, river trail"),
    ("Sam", "aw thats cute. couple goals", "N1", "Filler"),
    ("Maria", "Ha! He complains the whole time but he shows up.", "N1", "Humor filler"),
    ("Sam", "lmao classic dad", "N1", "Filler"),
    ("Maria", "My next checkup is in January. The doctor wants to see if the numbers improve enough to avoid medication.", "S1", "Next checkup + medication threshold"),
    ("Sam", "fingers crossed 🤞", "N1", "Support filler"),
    ("Maria", "Thank you sweetie. Oh, how's pottery going?", "N4", "Topic transition"),
    ("Sam", "SO GOOD actually", "N3", "Excited reaction"),
    ("Sam", "I made a mug this week that actually looks like a real mug lol", "S1", "Pottery progress: functional mug"),
    ("Sam", "and the instructor said I have a natural feel for the wheel", "S1", "Instructor feedback"),
    ("Maria", "Oh that's wonderful! Send me a photo.", "N1", "Request filler"),
    ("Sam", "I will when its glazed. I'm doing this blue-green glaze that Jake recommended", "S1", "Glaze detail + Jake callback"),
    ("Maria", "I can't wait to see it. Maybe you can make me something for Christmas?", "N1", "Request filler"),
    ("Sam", "haha maybe!! I was actually thinking about that", "S3", "Considering making gifts"),
    ("Maria", "I would treasure anything you make. OK go study. Love you.", "N2", "Farewell"),
    ("Sam", "ok ok. love u too mom. and seriously glad about the health stuff ❤️", "S5", "Emotional: relief about mom's health"),
    ("Maria", "Thank you baby. You focus on those apps!", "N2", "Farewell"),
    ("Sam", "I will I will byeee", "N2", "Farewell"),
    ("Maria", "Bye! 💕", "N2", "Farewell"),
]
g5 = gaps_for(40, [20, 30, 60, 30, 120, 30, 45, 20, 90, 30, 60, 30, 20, 45, 30, 120, 30, 60, 20, 30])


# Session 6: Sam submits Berkeley app! (35 msgs, 2025-12-15)
s6 = [
    ("Sam", "MOM", "N1", "Excitement opener"),
    ("Sam", "I DID IT", "S4", "Life event: submitted application"),
    ("Maria", "What?! Tell me!", "N3", "Excited reaction"),
    ("Sam", "I submitted the Berkeley application", "S1", "Specific school: Berkeley submitted"),
    ("Maria", "OH SAM!! 🎉🎉", "N3", "Celebration reaction"),
    ("Maria", "I'm so proud of you!", "N1", "Pride filler"),
    ("Sam", "lol thanks it feels so good to have it done", "S5", "Emotional: relief"),
    ("Maria", "How do you feel about it? The research proposal and everything?", "N1", "Question"),
    ("Sam", "actually really good. Professor Chen helped me narrow it down to adolescent PTSD interventions specifically", "S1", "Final research topic: adolescent PTSD"),
    ("Maria", "Oh that's so focused and meaningful.", "N5", "Affirming echo"),
    ("Sam", "yeah and my personal statement ties into it perfectly with the crisis center experience", "B1", "Connection between statement and proposal"),
    ("Maria", "When will you hear back?", "N1", "Question"),
    ("Sam", "they said decisions come out in March", "S1", "Decision timeline: March"),
    ("Maria", "March! That's a long wait.", "N3", "Reaction to timeline"),
    ("Sam", "I know 😩 but at least the hard part is done", "N1", "Filler"),
    ("Sam", "UCLA is due January 5th and UCSF is January 15th so I still have two more", "S1", "Remaining deadlines"),
    ("Maria", "But you can mostly reuse material right?", "N1", "Practical question"),
    ("Sam", "yeah basically. each school has slightly different essay prompts tho", "S1", "Application format detail"),
    ("Maria", "Well you'll get through it. The hard part was starting.", "N1", "Encouragement filler"),
    ("Sam", "true. oh btw I finished my pottery series!!", "S1", "Pottery course completed"),
    ("Sam", "I made you a bowl and a little planter for christmas 🎄", "S1", "Christmas gifts: bowl + planter"),
    ("Maria", "Sam!! Oh my heart. I can't wait.", "S5", "Emotional: touched by gift"),
    ("Sam", "lol they're not perfect but I tried", "N1", "Self-deprecating filler"),
    ("Maria", "They'll be perfect because you made them.", "N1", "Supportive filler"),
    ("Sam", "ok ur gonna make me cry in the library rn", "S5", "Emotional reaction"),
    ("Maria", "Ha! Sorry. When are you coming home for Christmas?", "N1", "Question"),
    ("Sam", "dec 20th. driving down with Jake actually hes going to Sacramento too", "S1", "Travel plan: Dec 20, driving with Jake to Sacramento"),
    ("Maria", "Jake the pottery friend?", "B1", "Callback to Jake"),
    ("Sam", "yeah he lives in Sac. well his parents do", "S1", "Jake's hometown: Sacramento"),
    ("Maria", "Oh how convenient! Is he a good driver? 😂", "N1", "Humor filler"),
    ("Sam", "lol yes mom he has a prius", "B2", "Casual fact: Jake's car"),
    ("Maria", "OK good. I'm so excited to see you. The house is already decorated.", "N1", "Excitement filler"),
    ("Sam", "can't wait ❤️", "N1", "Filler"),
    ("Maria", "Love you sweetie. Go celebrate tonight! You deserve it.", "N2", "Farewell"),
    ("Sam", "haha we're getting boba later. love u too!", "B2", "Casual celebration plan + farewell"),
]
g6 = gaps_for(35, [10, 30, 20, 30, 30, 45, 30, 120, 30, 20, 60, 30, 45, 30, 20, 90, 30, 45, 30, 60])


# Session 7: Holiday check-in, Sam's pottery progress (40 msgs, 2025-12-28)
s7 = [
    ("Maria", "Did you get home safely?", "N2", "Post-visit greeting"),
    ("Sam", "yeah jake just dropped me off. back in davis", "S1", "Travel update: back in Davis"),
    ("Maria", "Good. It was so wonderful having you home.", "N1", "Sentiment filler"),
    ("Sam", "it was really nice. thanks for everything mom", "N1", "Gratitude filler"),
    ("Maria", "I'm using the bowl you made me every morning for my oatmeal. I love it.", "S2", "Preference: using Sam's bowl daily"),
    ("Sam", "omg really?? that makes me so happy", "S5", "Emotional: joy about gift being used"),
    ("Maria", "Yes! And the planter is on the kitchen windowsill with the basil plant.", "S1", "Planter placement detail"),
    ("Sam", "perf 🌿", "N1", "Filler"),
    ("Maria", "Your aunt Rosa said she wants to commission you to make her some serving bowls.", "S1", "Rosa wants pottery commissions"),
    ("Sam", "lmao I'm not that good yet", "N1", "Self-deprecating filler"),
    ("Maria", "She was serious! She'll pay you.", "N1", "Emphasis filler"),
    ("Sam", "haha ok tell her I'll think about it", "N1", "Deflection"),
    ("Maria", "I will. How are the other applications coming?", "N4", "Topic transition"),
    ("Sam", "UCLA is basically done. submitting tomorrow", "S1", "UCLA app nearly complete + timeline"),
    ("Maria", "Oh wonderful! And UCSF?", "N1", "Follow-up"),
    ("Sam", "still working on that one. the essay prompt is weird. they want you to describe a clinical scenario", "S1", "UCSF essay prompt type"),
    ("Maria", "That sounds challenging. Can you use your crisis center experience?", "B1", "Callback to crisis center"),
    ("Sam", "yeah thats what im doing. writing about this one situation where I helped a teenager who was having a panic attack", "S1", "Specific scenario being used"),
    ("Maria", "Oh Sam. That must have been so difficult.", "N3", "Empathetic reaction"),
    ("Sam", "it was. but also like the most meaningful thing I've ever done honestly", "S5", "Emotional: deep meaning from volunteer work"),
    ("Sam", "she ended up coming back the next week and asked for me specifically", "S1", "Follow-up outcome: teen returned"),
    ("Maria", "See? You have a gift for this.", "N5", "Affirming echo"),
    ("Sam", "thanks mom 🥺", "N1", "Emotional filler"),
    ("Maria", "Have you thought about what happens if you get into more than one program?", "N1", "Strategic question"),
    ("Sam", "yeah I mean Berkeley is my top choice by far. the clinical psych program there is ranked like top 5 nationally", "S2", "Preference: Berkeley is top choice + ranking"),
    ("Maria", "And it's close enough that you could come home on weekends sometimes.", "N1", "Practical note"),
    ("Sam", "true. plus I already know the Bay Area from visiting Sofia", "B2", "Casual fact: visited Bay Area via cousin"),
    ("Maria", "Exactly. Oh speaking of, we finalized the reunion details.", "N4", "Topic transition"),
    ("Maria", "It's going to be at Tía Carmen's house in San Jose. March 15th.", "S1", "Reunion venue and date confirmed"),
    ("Sam", "oh nice. her house is huge right?", "N1", "Question filler"),
    ("Maria", "Yes perfect for a big gathering. We're expecting about 45 family members.", "S1", "Expected attendance number"),
    ("Sam", "wow thats a lot of primos", "N1", "Filler"),
    ("Maria", "Ha! Yes it is. Abuela is so excited. She keeps calling me to add people to the list.", "B2", "Casual fact: grandmother actively planning"),
    ("Sam", "lol classic abuela", "N1", "Filler"),
    ("Maria", "She wants you to bring your pottery to show everyone.", "S1", "Abuela wants to see pottery"),
    ("Sam", "omg no thats embarrassing", "N3", "Embarrassed reaction"),
    ("Maria", "She's proud of you! We all are.", "N1", "Supportive filler"),
    ("Sam", "ok ok fine. maybe I'll bring one piece lol", "S3", "Agreeing to bring pottery"),
    ("Maria", "Good. OK I'll let you go. Finish that UCSF application! Love you.", "N2", "Farewell"),
    ("Sam", "on it rn!! love u ❤️", "N2", "Farewell"),
]
g7 = gaps_for(40, [30, 45, 20, 60, 30, 90, 30, 20, 60, 45, 30, 120, 30, 20, 45, 60, 30, 90, 30, 20])


# Session 8: Sam submits all apps, pottery studio upgrade (40 msgs, 2026-01-16)
s8 = [
    ("Sam", "all three apps submitted 🎉🎉🎉", "S4", "Life event: all grad apps done"),
    ("Maria", "YESSS!! Berkeley, UCLA, and UCSF?", "N3", "Confirming reaction"),
    ("Sam", "yep all three. I'm freeeee", "S5", "Emotional: relief and freedom"),
    ("Maria", "I'm so proud of you Sam. That was so much work.", "N1", "Pride filler"),
    ("Sam", "it really was lol. like the last month has been insane", "N1", "Filler"),
    ("Maria", "When do you hear back from UCLA and UCSF?", "N1", "Question"),
    ("Sam", "UCLA is late February. UCSF is mid-March", "S1", "Decision timelines for other schools"),
    ("Maria", "OK so Berkeley is March, UCLA is February, UCSF is March. Got it.", "B3", "Temporal summary of all timelines"),
    ("Sam", "yeah its gonna be a stressful couple months lol", "N1", "Filler"),
    ("Maria", "Just try to enjoy the rest of your semester. You've earned it.", "N1", "Advice filler"),
    ("Sam", "yeah I plan to. actually I wanted to tell u something about pottery", "N4", "Topic transition"),
    ("Maria", "Oh what?", "N1", "Prompt"),
    ("Sam", "so the studio offered me a spot in their intermediate wheel-throwing class", "S1", "Course advancement: intermediate level"),
    ("Maria", "Oh how exciting! When does it start?", "N3", "Reaction"),
    ("Sam", "next week. its 8 weeks and we learn glazing techniques and larger forms", "S1", "Course details: 8 weeks, glazing + large forms"),
    ("Maria", "That sounds wonderful. More expensive though?", "N1", "Practical question"),
    ("Sam", "yeah $250 but they gave me a 15% discount cuz I finished the beginner series", "S1", "Cost + discount detail"),
    ("Maria", "Do you want me to pay for it? Consider it a late Christmas gift.", "S3", "Offering to fund the course"),
    ("Sam", "mom u dont have to", "N1", "Deflection"),
    ("Maria", "I want to. You're talented and I want to support that.", "N1", "Insistence filler"),
    ("Sam", "ok fine 😭 thank you", "N1", "Grateful acceptance"),
    ("Maria", "You're welcome! Just keep making me pottery.", "N1", "Humor filler"),
    ("Sam", "deal lol", "N1", "Agreement"),
    ("Sam", "oh also Jake and I might start selling stuff at the Davis farmers market", "S3", "Business plan: farmers market pottery sales"),
    ("Maria", "What?! Really?", "N3", "Surprised reaction"),
    ("Sam", "yeah the instructor connected us with the market organizer. you just need to apply for a booth", "S1", "Market connection + process"),
    ("Maria", "Oh Sam that would be amazing. What would you sell?", "N1", "Question"),
    ("Sam", "mugs and bowls mostly. maybe some planters. we'd split a booth, like $50 each per weekend", "S1", "Products + booth cost split"),
    ("Maria", "That's very entrepreneurial of you.", "N5", "Affirming echo"),
    ("Sam", "lol idk about that but it could be fun", "N1", "Deflection"),
    ("Maria", "I think it's wonderful. Your father will be thrilled.", "N1", "Supportive filler"),
    ("Sam", "haha yeah probably. ok I gotta go to class", "N2", "Farewell"),
    ("Maria", "OK sweetie. Oh wait — how's my health update?", "N4", "Topic transition"),
    ("Sam", "oh yeah!! how was the January checkup?", "B1", "Callback to health checkup"),
    ("Maria", "Great news actually! My glucose levels dropped significantly. Doctor said no medication needed for now.", "S4", "Life event: health improvement, no meds"),
    ("Sam", "MOM THATS AMAZING", "N3", "Excited reaction"),
    ("Maria", "Yes! The walking and diet changes are working. Just need to keep it up.", "S1", "Treatment effectiveness confirmed"),
    ("Sam", "im so happy. seriously. like so relieved", "S5", "Emotional: relief about mom's health"),
    ("Maria", "Me too sweetie. OK go to class for real. Love you!", "N2", "Farewell"),
    ("Sam", "love u!! ❤️❤️", "N2", "Farewell"),
]
g8 = gaps_for(40, [30, 20, 60, 30, 45, 30, 120, 30, 60, 30, 20, 45, 30, 60, 30, 90, 30, 20, 120, 30])


# Session 9: Reunion prep + waiting for decisions (45 msgs, 2026-02-20)
s9 = [
    ("Maria", "Good morning sweetie! How are you?", "N2", "Greeting"),
    ("Sam", "morning mom", "N2", "Greeting"),
    ("Sam", "I'm ok. kinda anxious honestly", "S5", "Emotional: anxiety about decisions"),
    ("Maria", "About the applications?", "N1", "Question"),
    ("Sam", "yeah UCLA decisions should be coming out soon. like any day now", "S1", "UCLA decision timing"),
    ("Maria", "Oh how exciting. And nerve-wracking I'm sure.", "N5", "Empathetic echo"),
    ("Sam", "yeah lol I check my email like every 5 minutes", "N1", "Behavioral filler"),
    ("Maria", "I remember when your father was waiting for his med school acceptance. He was a wreck.", "B2", "Casual fact: dad's med school anxiety"),
    ("Sam", "lol some things are genetic I guess", "N1", "Humor filler"),
    ("Maria", "Ha! True. Listen, whatever happens, I'm proud of you no matter what.", "N1", "Supportive filler"),
    ("Sam", "thx mom ❤️", "N1", "Gratitude filler"),
    ("Maria", "So the reunion is in less than a month. March 15th at Tía Carmen's.", "B3", "Temporal: reunion approaching"),
    ("Sam", "yeah I have it on my calendar", "N1", "Acknowledgment"),
    ("Maria", "Good. I'm coordinating the food with Rosa and Carmen. We need to figure out seating too.", "S1", "Coordination details"),
    ("Sam", "lol u r such a planner mom", "N1", "Commentary"),
    ("Maria", "Someone has to be! Abuela keeps trying to invite more people. We're up to 50 now.", "S1", "Updated guest count: 50"),
    ("Sam", "lmao 50?? is there even room", "N3", "Reaction"),
    ("Maria", "Carmen's backyard is huge. Plus we're renting tables and chairs.", "S1", "Venue logistics: outdoor + rentals"),
    ("Sam", "ok fair. what do u need me to do", "N1", "Question"),
    ("Maria", "Can you help with the music? You always have good taste.", "S3", "Assigning Sam a role: music"),
    ("Sam", "oh yeah I can make a playlist", "S3", "Accepting music responsibility"),
    ("Maria", "Perfect. A mix of everything — some old school for Abuela, some modern stuff for you kids.", "S2", "Music preference guidelines"),
    ("Sam", "lol 'you kids.' I'm 22 mom", "N1", "Protest filler"),
    ("Maria", "You'll always be my kid. Oh, are you bringing anyone? A friend? A... special someone?", "N1", "Nosy parent question"),
    ("Sam", "MOM", "N1", "Exasperated filler"),
    ("Maria", "What?! I'm just asking!", "N1", "Innocent protest filler"),
    ("Sam", "no I'm coming solo lol. I'll drive with sofia actually, she said she'd pick me up from the train station", "S1", "Travel plan: train + Sofia pickup"),
    ("Maria", "Oh that's perfect. You two can catch up on the way.", "N1", "Filler"),
    ("Sam", "yeah. hey can I ask u something about abuela", "N4", "Topic transition"),
    ("Maria", "Of course.", "N1", "Response"),
    ("Sam", "do u think she'd like a pottery piece from me? like as a birthday gift", "S3", "Gift idea: pottery for grandmother"),
    ("Maria", "Oh Sam she would LOVE that. She talks about your pottery all the time.", "N3", "Enthusiastic reaction"),
    ("Sam", "wait she does?? I only showed her one piece", "N3", "Surprised reaction"),
    ("Maria", "I sent her photos of everything you've made. She shows all her friends.", "B2", "Casual fact: Maria shares Sam's work with abuela"),
    ("Sam", "omg mom 😭", "S5", "Emotional: touched"),
    ("Maria", "What?! I'm a proud mother!", "N1", "Defensive filler"),
    ("Sam", "ok well I'll make her something special. maybe a vase", "S3", "Decision: making abuela a vase"),
    ("Maria", "She'll treasure it forever. Oh, one more thing — the farmers market. How's that going?", "N4", "Topic transition"),
    ("Sam", "oh so we did our first weekend last Saturday! Jake and I split a booth", "S4", "Life event: first farmers market"),
    ("Maria", "How did it go?!", "N3", "Excited reaction"),
    ("Sam", "we sold 6 mugs and 3 bowls!! made like $180 total", "S1", "Sales data: items + revenue"),
    ("Maria", "That's incredible for your first time!", "N3", "Reaction"),
    ("Sam", "right?? people really liked the glazes. this one woman bought 3 mugs as gifts", "S1", "Customer detail"),
    ("Maria", "I'm beaming right now. OK I have to go start dinner. Love you so much.", "N2", "Farewell"),
    ("Sam", "love u too mom ❤️ I'll let u know the SECOND I hear from UCLA", "N2", "Farewell + promise"),
]
g9 = gaps_for(45, [30, 20, 60, 30, 90, 30, 45, 20, 60, 30, 120, 30, 20, 45, 60, 30, 90, 30, 20, 60])


# Session 10: UCLA acceptance!! + reunion excitement (40 msgs, 2026-02-28)
s10 = [
    ("Sam", "MOM MOM MOM", "N1", "Excitement opener"),
    ("Sam", "CALL ME RN", "N1", "Urgent filler"),
    ("Maria", "What?! What happened?! Is everything OK?!", "N3", "Panicked reaction"),
    ("Sam", "YES OMG YES", "N1", "Excitement filler"),
    ("Sam", "I GOT INTO UCLA", "S4", "Life event: UCLA acceptance"),
    ("Maria", "WHAT", "N3", "Shocked reaction"),
    ("Maria", "SAM", "N1", "Excitement echo"),
    ("Maria", "OH MY GOD!! 🎉🎉🎉", "N3", "Celebration reaction"),
    ("Sam", "I KNOW I'M LITERALLY CRYING RN", "S5", "Emotional: crying from joy"),
    ("Maria", "I'm crying too!! Oh honey I'm so proud of you!", "S5", "Emotional: shared joy"),
    ("Sam", "the email came like 20 minutes ago. I screamed in the library", "B3", "Temporal + setting detail"),
    ("Maria", "Ha!! Tell me everything. What did the letter say?", "N1", "Prompt"),
    ("Sam", "ok so its for the Clinical Psychology PhD program. full admission with funding!!", "S1", "Program + funding status"),
    ("Maria", "FUNDING?!", "N3", "Shocked reaction"),
    ("Sam", "yeah they're offering a full tuition waiver plus a $32,000 annual stipend", "S1", "Financial package: tuition waiver + stipend amount"),
    ("Maria", "Oh Sam. Oh my God. That's... that's everything.", "S5", "Emotional: overwhelmed by financial aid"),
    ("Sam", "I know. I literally cant believe it", "N1", "Filler"),
    ("Sam", "its a 5-year program. first 2 years are coursework and the last 3 are research and clinical practice", "S1", "Program structure: 5 years, course/research split"),
    ("Maria", "And they're PAYING you to do it. Your father is going to lose his mind.", "N1", "Filler"),
    ("Sam", "lol tell him!! I tried calling but he didn't pick up", "N1", "Filler"),
    ("Maria", "He's in surgery right now. I'll tell him the SECOND he's out.", "B2", "Casual fact: dad is a surgeon, currently in OR"),
    ("Maria", "Oh honey, have you heard from Berkeley or UCSF yet?", "N4", "Question transition"),
    ("Sam", "not yet. berkeley is early march and ucsf is mid-march", "B3", "Temporal: remaining decision dates"),
    ("Maria", "Well you already have UCLA! With funding! Whatever else happens is just a bonus.", "N1", "Reassurance"),
    ("Sam", "yeah honestly even if I don't get in anywhere else I'm SO happy with UCLA", "S2", "Preference: happy with UCLA outcome"),
    ("Maria", "The campus is beautiful. When would you start?", "N1", "Question"),
    ("Sam", "fall semester. September", "S1", "Start date: September"),
    ("Maria", "That's exciting. You'd be in Los Angeles!", "N3", "Reaction"),
    ("Sam", "I know!! I've always wanted to live in LA honestly", "S2", "Preference: wants to live in LA"),
    ("Maria", "We'll need to help you find an apartment. Does the program have housing?", "N1", "Practical question"),
    ("Sam", "theres grad student housing yeah. I need to look into it", "S1", "Housing option: grad student housing available"),
    ("Maria", "We'll figure it all out. For now just CELEBRATE.", "N1", "Encouragement"),
    ("Sam", "lol jake and I are going out tonight for sure", "B2", "Casual celebration plan with Jake"),
    ("Maria", "Good! You deserve it. Oh Sam, wait until Abuela hears at the reunion.", "B1", "Callback to reunion: sharing news there"),
    ("Sam", "omg she's gonna lose it", "N3", "Reaction"),
    ("Maria", "She's going to tell EVERYONE. Her grandbaby the doctor. Well, psychologist.", "N1", "Humor filler"),
    ("Sam", "lmaooo not a doctor yet mom", "N1", "Correction filler"),
    ("Maria", "Yet! I'm telling all my friends too. My coworker Diana's daughter just got rejected from 3 programs and I feel bad but also—", "B2", "Casual fact: Maria's coworker Diana, social comparison"),
    ("Sam", "MOM don't be that person lol", "N1", "Protest filler"),
    ("Maria", "OK OK I'll be tasteful about it. LOVE YOU SO MUCH. ❤️🎉", "N2", "Farewell"),
]
g10 = gaps_for(40, [5, 30, 20, 10, 30, 20, 20, 30, 30, 60, 30, 120, 30, 20, 90, 30, 45, 30, 20, 60])


# ═══════════════════════════════════════════════════════════════════════════
# BUILD AND WRITE
# ═══════════════════════════════════════════════════════════════════════════
sessions = [
    (1,  "2025-11-05", "2025-11-05T18:30:00", g1,  s1),
    (2,  "2025-11-10", "2025-11-10T10:15:00", g2,  s2),
    (3,  "2025-11-18", "2025-11-18T19:00:00", g3,  s3),
    (4,  "2025-11-25", "2025-11-25T16:30:00", g4,  s4),
    (5,  "2025-12-02", "2025-12-02T20:00:00", g5,  s5),
    (6,  "2025-12-15", "2025-12-15T14:00:00", g6,  s6),
    (7,  "2025-12-28", "2025-12-28T11:00:00", g7,  s7),
    (8,  "2026-01-16", "2026-01-16T17:00:00", g8,  s8),
    (9,  "2026-02-20", "2026-02-20T09:00:00", g9,  s9),
    (10, "2026-02-28", "2026-02-28T15:30:00", g10, s10),
]

all_messages = []
for num, date, start, gaps, msgs in sessions:
    assert len(gaps) == len(msgs) - 1, \
        f"Session {num}: {len(gaps)} gaps for {len(msgs)} msgs (need {len(msgs)-1})"
    all_messages.extend(build(num, date, start, gaps, msgs))

cat_counts = Counter(m["category"] for m in all_messages)
noise_total = sum(v for k, v in cat_counts.items() if k.startswith("N"))
signal_total = sum(v for k, v in cat_counts.items() if k.startswith("S"))
border_total = sum(v for k, v in cat_counts.items() if k.startswith("B"))
speaker_counts = Counter(m["speaker"] for m in all_messages)

output = {
    "conversation_id": "conv2",
    "speakers": {
        "Maria": "Sam's mother, caring and sometimes overbearing, formal communication style",
        "Sam": "College student (Davis), applying to clinical psych PhD programs, pottery hobby",
    },
    "date_range": {"start": "2025-11-05", "end": "2026-02-28"},
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

out_path = Path("benchmarks/gate_eval/datasets/gate_benchmark_conv2.json")
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
    print(f"  Session {num}: {len(sm)} msgs | N={n} S={s} B={b}")
