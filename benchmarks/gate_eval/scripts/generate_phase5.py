#!/usr/bin/env python3
"""Phase 5: Conv 4 — Pat & Riley (coworkers becoming friends), 400 messages, 8 sessions.

Lowest noise conversation (40%) — most substantive. Pat is analytical, starting
a side business (analytics SaaS). Riley is supportive, uses humor, training for
Portland Marathon. Professional tone early, progressively casual.

Target distribution:
  Noise:      ~160 (40%)
  Signal:     ~140 (35%)
  Borderline: ~100 (25%)
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


def build(sess_num: int, date: str, start: str, gaps: list[int], msgs: list[tuple], conv="conv4") -> list[dict]:
    ts = make_timestamps(start, gaps)
    out = []
    for i, (spk, txt, cat, note) in enumerate(msgs):
        out.append({
            "id": f"{conv}_s{sess_num:02d}_{i+1:03d}",
            "conversation_id": conv,
            "session": f"session_{sess_num}",
            "session_date": date,
            "speaker": spk,
            "recipient": "Riley" if spk == "Pat" else "Pat",
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
# CONVERSATION 4: Pat & Riley
# Pat = analytical coworker, starting analytics SaaS, asks "what do you think"
# Riley = supportive, humorous, marathon runner, deflects serious with jokes
# 8 sessions over 3 weeks (Nov 3 – Nov 28, 2025), 400 messages, 40% noise
# Noise=160, Signal=140, Borderline=100
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# SESSION 1: First real non-work conversation (50 msgs, 2025-11-03 Mon)
# Team offsite. Discover shared interests. Still professional-ish.
# Pat mentions frustration with project lead Vanessa. Riley mentions running.
# ═══════════════════════════════════════════════════════════════════════════
s1 = [
    ("Pat",   "Hey, did you end up going to that breakout session on data pipelines?", "N2", "Greeting + offsite question"),
    ("Riley", "Yeah! Actually it was pretty good. The speaker was from Datadog.", "S1", "Speaker company fact"),
    ("Pat",   "Oh nice, I skipped it for the leadership one. Kinda wish I hadn't.", "B2", "Casual preference about session choice"),
    ("Riley", "Haha was it bad?", "N1", "Question filler"),
    ("Pat",   "Not bad exactly, just very corporate-speak. Lots of 'synergy' and 'alignment.'", "B2", "Casual opinion embedded in conversation"),
    ("Riley", "Oh god, the alignment talk. I swear every offsite has one of those.", "B2", "Casual observation about offsite patterns"),
    ("Pat",   "Right? Anyway, I think we're in the same group for the afternoon thing.", "B3", "Temporal: afternoon schedule"),
    ("Riley", "Oh yeah the team-building exercise? I saw the list.", "B1", "Context-enabling: references offsite schedule"),
    ("Pat",   "Yeah. Should be interesting at least.", "N1", "Filler"),
    ("Riley", "Honestly I'm just glad to be out of the office. Vanessa's been on a tear lately.", "S1", "Mentions Vanessa + work climate"),
    ("Pat",   "Oh my god, tell me about it. She pulled me into a 'strategy session' Friday that was basically her rethinking the entire Q4 roadmap.", "S1", "Vanessa event: rethinking Q4 roadmap"),
    ("Riley", "Wait, again? Didn't she just finalize the roadmap like two weeks ago?", "B3", "Temporal: roadmap was finalized 2 weeks prior"),
    ("Pat",   "Yep. Two weeks. The ink wasn't even dry.", "N5", "Confirming echo"),
    ("Riley", "That's wild. I feel like she does this every quarter.", "B2", "Casual observation about Vanessa's behavior pattern"),
    ("Pat",   "She does. It's exhausting. What do you think about the way she handles sprint planning in general?", "S2", "Pat's analytical probing about management style"),
    ("Riley", "Honestly? I think she means well but she changes direction so fast that nobody can actually execute.", "S2", "Assessment of Vanessa's management"),
    ("Pat",   "That's exactly it. The team can't build momentum when the target keeps moving.", "B2", "Casual elaboration on team dynamics"),
    ("Riley", "Yeah. Anyway, at least the food here is good.", "N4", "Topic deflection"),
    ("Pat",   "Ha, true. The coffee is actually decent for once.", "N1", "Filler"),
    ("Riley", "Right? I've had like three cups already. I'm going to be wired for the afternoon session.", "B2", "Casual fact about coffee consumption"),
    ("Pat",   "Same. So what do you usually do on weekends? I realized we've worked together for like 8 months and I barely know anything about you outside of standups.", "S1", "Duration of working together: 8 months"),
    ("Riley", "Ha, that's so true. Honestly I've been training a lot lately. I'm doing the Portland Marathon in April.", "S1", "Portland Marathon goal, April"),
    ("Pat",   "Wait, really? That's awesome. Have you done a marathon before?", "N1", "Follow-up question"),
    ("Riley", "Nope, this would be my first. I've done a bunch of halfs though. The last one was the Seattle Half in September, did a 1:42.", "S1", "First marathon, Seattle Half in Sept, time 1:42"),
    ("Pat",   "Is that good? I honestly know nothing about running.", "B2", "Casual admission: Pat knows nothing about running"),
    ("Riley", "Ha, yeah it's decent. I'm trying to go sub-3:30 for the full.", "S1", "Marathon time goal: sub-3:30"),
    ("Pat",   "What do you think about the training load? Like how much time does it actually take?", "N1", "Analytical question"),
    ("Riley", "Right now I'm doing about 40 miles a week. Five days running, two rest days. It's a lot honestly.", "S1", "Training volume: 40 mi/week, 5 days running"),
    ("Pat",   "Wow, that's serious commitment. I could barely run a mile without dying.", "B2", "Casual fact: Pat doesn't run"),
    ("Riley", "Haha it's not for everyone. What about you? Any hobbies outside of work?", "N1", "Question filler"),
    ("Pat",   "I've been reading a lot about startups lately, actually. Like obsessively.", "S2", "Interest: startup ecosystem"),
    ("Riley", "Oh interesting. Like what kind of stuff?", "N1", "Question"),
    ("Pat",   "Mostly analytics and data tooling. I keep seeing gaps in the market that our team runs into every day.", "B2", "Casual embedded observation about market"),
    ("Riley", "Huh. Are you thinking about building something?", "N1", "Question"),
    ("Pat",   "Maybe? I don't know. It's just an idea at this point. I read this article on Indie Hackers about a guy who built a $30K MRR analytics tool as a solo founder.", "S1", "Reference: Indie Hackers article, $30K MRR"),
    ("Riley", "That's cool. You definitely have the brain for it.", "N1", "Supportive filler"),
    ("Pat",   "Thanks. I'll probably just keep thinking about it for another year and never do anything, knowing me.", "S5", "Self-aware emotional disclosure about inaction tendency"),
    ("Riley", "Ha, that's literally everyone with a side project idea. Join the club.", "N1", "Normalizing filler"),
    ("Pat",   "Fair point.", "N1", "Agreement filler"),
    ("Riley", "Hey I think they're calling us back for the afternoon thing.", "N4", "Meta-conversation transition"),
    ("Pat",   "Oh right. Which room are we in?", "N1", "Question"),
    ("Riley", "Conference B I think. Second floor.", "B1", "Location detail"),
    ("Pat",   "Cool, I'll head over. Nice actually talking to you outside of standup for once.", "N2", "Farewell + meta"),
    ("Riley", "Ha, same. We should do this more often.", "N2", "Farewell"),
    ("Pat",   "Definitely.", "N1", "Agreement filler"),
    ("Riley", "Oh hey, random question — do you know if the offsite dinner tonight is mandatory?", "B1", "Context: offsite dinner"),
    ("Pat",   "I think it's 'strongly encouraged,' which at this company means yes.", "B2", "Casual fact about company culture"),
    ("Riley", "Ugh. Fine. At least it's at that Italian place on 4th. Lucca's.", "S1", "Dinner location: Lucca's on 4th"),
    ("Pat",   "Oh nice, I've actually heard good things about that place.", "N1", "Filler"),
    ("Riley", "Yeah their pasta is supposed to be incredible. OK see you in Conference B.", "B2", "Casual food fact + location callback"),
]
g1 = gaps_for(50, [45, 30, 60, 20, 90, 30, 45, 120, 30, 60, 20, 45, 30, 90, 60, 30, 45, 20, 120, 30])


# ═══════════════════════════════════════════════════════════════════════════
# SESSION 2: Work drama escalates (50 msgs, 2025-11-06 Thu)
# Vanessa wants to pivot project 2 weeks before deadline. Pat vents.
# Riley offers perspective. Pat first mentions side business idea (analytics SaaS).
# ═══════════════════════════════════════════════════════════════════════════
s2 = [
    ("Pat",   "Are you in the office today?", "N2", "Opening question"),
    ("Riley", "Yeah I'm at my desk. What's up?", "N1", "Response"),
    ("Pat",   "Did you see the email from Vanessa?", "S1", "Reference to Vanessa email"),
    ("Riley", "Which one? She sent like 4 this morning.", "B2", "Casual fact: Vanessa sent 4 emails"),
    ("Pat",   "The one about pivoting the dashboard project. She wants to completely rethink the metrics layer.", "S1", "Vanessa pivot: rethink metrics layer"),
    ("Riley", "Oh that one. Yeah I read it twice trying to figure out what she actually wants.", "B1", "Context-enabling: Riley's confusion about the email"),
    ("Pat",   "The demo to the VP is in two weeks, Riley. TWO WEEKS.", "S1", "VP demo in two weeks, deadline pressure"),
    ("Riley", "I know. It's... ambitious.", "N1", "Diplomatic filler"),
    ("Pat",   "It's insane is what it is. We scoped this thing out in August. The architecture is already built.", "S1", "Project scoped in August, architecture complete"),
    ("Riley", "Yeah I was there for the design review. We spent like a whole sprint just on the schema.", "B3", "Temporal: design review was a full sprint"),
    ("Pat",   "Exactly. And now she wants to swap out the entire aggregation pipeline because she read some blog post about 'real-time analytics.'", "S1", "Vanessa's reason: blog post about real-time analytics"),
    ("Riley", "Lol wait, a blog post? Please tell me you're exaggerating.", "N3", "Reaction"),
    ("Pat",   "I wish I was. She literally linked it in the email. It was a Medium article.", "S1", "Source: Medium article"),
    ("Riley", "Oh no. The Medium article pipeline. Classic Vanessa.", "B1", "Context-enabling: establishes Vanessa pattern"),
    ("Pat",   "What do you think I should do? Push back or just try to make it work?", "S2", "Pat asking for advice, analytical framing"),
    ("Riley", "Honestly? Push back. But tactfully. Frame it as risk to the demo timeline.", "S2", "Riley's strategic advice"),
    ("Pat",   "Yeah that's what I was thinking. I need to quantify the cost of the pivot.", "S2", "Pat's analytical approach"),
    ("Riley", "There you go. She responds to numbers. If you can show her it adds 3 weeks of work, she'll probably back down.", "B1", "Context-enabling: tactical advice about Vanessa"),
    ("Pat",   "True. She does respond to data better than opinions.", "B2", "Casual insight: Vanessa responds to data"),
    ("Riley", "Meanwhile I'm just over here trying to finish the code review backlog before standup tomorrow.", "B3", "Temporal: code review before tomorrow's standup"),
    ("Pat",   "Ugh, the code review backlog. I have like 6 PRs waiting too.", "B2", "Casual fact: Pat has 6 PRs in queue"),
    ("Riley", "Sprint planning is going to be a blast this week.", "B3", "Temporal: sprint planning this week"),
    ("Pat",   "Can't wait.", "N1", "Sarcastic agreement"),
    ("Riley", "Anyway. You'll figure it out with Vanessa. You always do.", "N1", "Supportive filler"),
    ("Pat",   "Thanks. Hey, actually — remember how I mentioned at the offsite that I've been reading about analytics tools?", "B1", "Callback to offsite conversation"),
    ("Riley", "Yeah, the Indie Hackers stuff?", "B1", "Callback to specific reference"),
    ("Pat",   "So I've actually been thinking about it more seriously. Like, what if I actually built something?", "S3", "Decision-adjacent: considering building a product"),
    ("Riley", "Oh wow, really? What would it be?", "N3", "Interested reaction"),
    ("Pat",   "An analytics SaaS tool. Specifically for mid-size teams that outgrow Google Analytics but can't afford Amplitude or Mixpanel.", "S1", "Business idea: analytics SaaS for mid-size teams"),
    ("Riley", "Oh that's actually a real gap. We dealt with that exact problem at my last company.", "S2", "Validation: Riley experienced the same gap"),
    ("Pat",   "Right?? That's what got me thinking. Every mid-size company I've worked at has this exact problem.", "S2", "Further validation of market gap"),
    ("Riley", "Do you have a name for it yet?", "N1", "Question"),
    ("Pat",   "Not yet. I'm still in the 'is this actually viable' phase. I've been doing customer interviews on the side.", "S1", "Current phase: customer interviews"),
    ("Riley", "Wait you're already doing customer interviews? That's not 'just an idea' anymore.", "N3", "Reaction to progress"),
    ("Pat",   "Ha, I guess not. I've talked to like 6 people so far. Mostly product managers and growth leads.", "S1", "Interview count: 6, target personas: PMs and growth leads"),
    ("Riley", "And what are they saying?", "N1", "Question"),
    ("Pat",   "Every single one said the same thing: they need something between GA and the enterprise tools. Something that does event analytics without a $50K annual contract.", "S1", "Customer interview finding: need for mid-tier pricing"),
    ("Riley", "Dude that's really promising. Have you thought about pricing?", "N1", "Question"),
    ("Pat",   "Not in detail yet. But the sweet spot seems to be somewhere around $30-60 per month per seat.", "S1", "Preliminary pricing range: $30-60/seat/month"),
    ("Riley", "That sounds right. Low enough to expense without VP approval.", "B2", "Casual pricing insight"),
    ("Pat",   "Exactly. The 'just put it on the company card' price point.", "B2", "Casual pricing strategy note"),
    ("Riley", "Ha! That's literally a business strategy. I love it.", "N3", "Reaction"),
    ("Pat",   "Thanks for actually taking this seriously, by the way. I haven't told many people.", "S5", "Emotional: vulnerability about sharing idea"),
    ("Riley", "Of course. I think you should go for it honestly.", "N1", "Supportive filler"),
    ("Pat",   "We'll see. One existential work crisis at a time.", "B2", "Casual: Pat's self-awareness about priorities"),
    ("Riley", "Lol fair. OK I really need to go look at those PRs. Talk later?", "N2", "Farewell"),
    ("Pat",   "Yeah for sure. And thanks for the Vanessa advice. I'll write up the risk analysis tonight.", "S3", "Commitment: writing risk analysis"),
    ("Riley", "Smart. Good luck with it. Let me know how it goes.", "N2", "Farewell"),
    ("Pat",   "Will do. Thanks Riley.", "N2", "Farewell"),
    ("Riley", "👍", "N1", "Emoji filler"),
]
g2 = gaps_for(50, [30, 60, 30, 45, 20, 90, 30, 60, 30, 20, 120, 45, 30, 60, 30, 45, 20, 90, 30, 60])


# ═══════════════════════════════════════════════════════════════════════════
# SESSION 3: Marathon update + side business MVP + wedding talk (55 msgs, 2025-11-10 Mon)
# Riley: Portland Marathon training update (April 12th). Pat: validated idea,
# wants to build MVP. Mutual friends' wedding: Jason & Elena, Napa Valley, Dec 14.
# ═══════════════════════════════════════════════════════════════════════════
s3 = [
    ("Riley", "Happy Monday. How'd the Vanessa thing go?", "N2", "Greeting + callback"),
    ("Pat",   "Actually really well! I sent her the risk analysis Friday and she backed off the full pivot.", "S1", "Vanessa outcome: backed off pivot"),
    ("Riley", "Oh nice! So no last-minute architecture rework?", "N1", "Clarifying question"),
    ("Pat",   "She still wants some changes but nothing that blows up the timeline. We compromised on adding two new metric views instead of redoing the whole pipeline.", "S1", "Compromise: two new metric views"),
    ("Riley", "See, I told you. Numbers work on her.", "B1", "Context-enabling: callback to own earlier advice"),
    ("Pat",   "You were right. I even put together a Gantt chart showing the impact and she was like 'oh I didn't realize it was that much work.'", "B2", "Casual fact: used a Gantt chart"),
    ("Riley", "Lmao of course she didn't. Managers never do.", "B2", "Casual insight about management"),
    ("Pat",   "Ha. True. So how was your weekend? Did you get your long run in?", "N4", "Topic transition to running"),
    ("Riley", "YES and it was a good one actually. Did 16 miles Saturday along the river trail.", "S1", "Long run: 16 miles, river trail"),
    ("Pat",   "Sixteen miles?? That's like... a lot.", "N3", "Impressed reaction"),
    ("Riley", "Ha yeah. It's my longest run so far in this training cycle. The Portland Marathon is April 12th so I need to peak around late March.", "S1", "Marathon date: April 12th, peak timing"),
    ("Pat",   "How do you even train for something like that? Is there like a plan you follow?", "N1", "Curious question"),
    ("Riley", "Yeah I'm doing the Pfitzinger 18/55 plan. It's an 18-week program with 5 runs per week, building up to 55 miles in the peak week.", "S1", "Training plan: Pfitzinger 18/55, 18 weeks, 55 peak miles"),
    ("Pat",   "That's incredibly structured. I respect the data-driven approach.", "B2", "Casual fact: Pat appreciates data-driven methods"),
    ("Riley", "Ha, you would appreciate the data side. I track everything in Strava — splits, heart rate, elevation.", "S1", "Tracking: uses Strava"),
    ("Pat",   "Of course you do. What do you think about the mental side of it? Like how do you stay motivated on the bad days?", "S2", "Pat's analytical probing of motivation"),
    ("Riley", "Honestly? Some days I don't. I just go anyway and hate every second of it. And then I feel amazing after.", "S5", "Emotional: honest about struggle + payoff"),
    ("Pat",   "That's kind of inspiring actually.", "N1", "Filler"),
    ("Riley", "Don't tell anyone I said something inspiring. I have a brand to maintain.", "B2", "Casual: Riley's self-image as humor-only"),
    ("Pat",   "Ha, your secret is safe with me.", "N1", "Filler"),
    ("Riley", "So what about you? How's the analytics thing going? Have you done more interviews?", "N4", "Topic transition"),
    ("Pat",   "OK so this is actually exciting. I've now talked to 12 people total and the pattern is really clear.", "S1", "Interview count: 12 total"),
    ("Pat",   "I feel like I've validated the core hypothesis. There's a real gap between free tools and enterprise solutions.", "S3", "Decision: considers idea validated"),
    ("Riley", "12 interviews is no joke. So what's the next step?", "N1", "Question"),
    ("Pat",   "I want to build an MVP. Nothing fancy — just event tracking, a simple dashboard, and basic funnels.", "S1", "MVP scope: event tracking, dashboard, funnels"),
    ("Riley", "Are you going to build it yourself or find a cofounder?", "N1", "Question"),
    ("Pat",   "Myself for now. I can handle the backend and I found this React template that'll save time on the frontend.", "S1", "Solo build, React template for frontend"),
    ("Riley", "When do you think you can have something working?", "N1", "Question"),
    ("Pat",   "If I'm disciplined about it, maybe 6-8 weeks? I've been doing about 2 hours a night after work.", "S1", "Timeline: 6-8 weeks, 2 hours/night"),
    ("Riley", "Two hours a night PLUS the day job? That's intense.", "N3", "Reaction"),
    ("Pat",   "Yeah it's a lot. But honestly it doesn't feel like work. It's the first thing in a long time that actually energizes me.", "S5", "Emotional: energized by side project"),
    ("Riley", "That's how you know it's the right thing. When it doesn't feel like a slog.", "B2", "Casual philosophy about passion"),
    ("Pat",   "What do you think about the name 'Clearview Analytics'?", "S1", "Proposed business name: Clearview Analytics"),
    ("Riley", "Hmm. It's fine but there's a facial recognition company called Clearview AI. Might cause confusion.", "B1", "Context-enabling: naming conflict knowledge"),
    ("Pat",   "Oh good call. Back to the drawing board on that one.", "N1", "Agreement"),
    ("Riley", "What about something shorter? Like Beacon or Tally or Signal?", "B2", "Casual name suggestions"),
    ("Pat",   "Hmm, those are interesting. I'll think about it.", "N1", "Consideration"),
    ("Riley", "Hey totally different topic — did you get Jason's wedding invite?", "N4", "Topic transition to wedding"),
    ("Pat",   "Yes! Jason and Elena, right? December 14th in Napa Valley.", "S1", "Wedding: Jason & Elena, Dec 14, Napa Valley"),
    ("Riley", "Yeah! Are you going?", "N1", "Question"),
    ("Pat",   "Definitely. I've known Jason since college. How do you know them?", "S1", "Pat knows Jason from college"),
    ("Riley", "Elena and I worked together at my first job out of school. She was my desk neighbor for two years.", "S1", "Riley's connection to Elena: first job, desk neighbors"),
    ("Pat",   "Oh that's funny. I didn't realize we had mutual friends outside of work.", "B2", "Casual observation: shared social circles"),
    ("Riley", "Small world! I'm excited for the wedding honestly. Napa in December is gorgeous.", "B2", "Casual opinion: Napa in December"),
    ("Pat",   "Yeah I heard they booked this vineyard called Domaine Chandon. It's supposed to be beautiful.", "S1", "Venue: Domaine Chandon vineyard"),
    ("Riley", "Oh I've been there for a tasting. It's stunning.", "B2", "Casual fact: Riley has visited the venue"),
    ("Pat",   "Nice. Have you thought about a gift yet?", "N1", "Question"),
    ("Riley", "Not yet. They're registered at Crate & Barrel and Williams-Sonoma I think.", "S1", "Wedding registries: Crate & Barrel, Williams-Sonoma"),
    ("Pat",   "OK cool. I'll look at those. What do you think about going in on something together? Might be nice to get them something bigger.", "S3", "Proposing joint gift idea"),
    ("Riley", "Oh that's a great idea actually. Let me pull up the registry.", "N1", "Agreement filler"),
    ("Pat",   "Yeah let's figure it out closer to the date. No rush.", "N1", "Filler"),
    ("Riley", "For sure. We can look at the registry together sometime this week.", "B3", "Temporal: planning to look this week"),
    ("Riley", "Sounds good. OK I need to hop on my 2pm. Talk later?", "N2", "Farewell"),
    ("Pat",   "Yeah go. And good luck on your next long run.", "N2", "Farewell"),
    ("Riley", "Thanks! 🏃", "N2", "Farewell with emoji"),
]
g3 = gaps_for(55, [30, 60, 45, 20, 90, 30, 45, 30, 120, 30, 20, 60, 45, 30, 90, 30, 20, 60, 120, 30])


# ═══════════════════════════════════════════════════════════════════════════
# SESSION 4: Project crisis + first customer interest (50 msgs, 2025-11-13 Thu)
# Vanessa changes requirements again. Pat & Riley bond over frustration.
# Pat's side business gets first interest from a potential customer.
# ═══════════════════════════════════════════════════════════════════════════
s4 = [
    ("Pat",   "I'm going to lose it.", "N1", "Frustration opener"),
    ("Riley", "Uh oh. Vanessa?", "N3", "Concerned reaction"),
    ("Pat",   "Who else. She called an emergency meeting this morning to change the requirements for the demo AGAIN.", "S1", "Vanessa: emergency meeting, changed demo requirements"),
    ("Riley", "You're kidding. The demo is next Thursday.", "B3", "Temporal: demo next Thursday"),
    ("Pat",   "I am not kidding. She wants us to add a real-time notification system now. Real-time. In one week.", "S1", "New requirement: real-time notifications in one week"),
    ("Riley", "That's... not feasible.", "N3", "Reaction"),
    ("Pat",   "I know! I told her that. She said 'I believe in this team's ability to deliver under pressure.'", "S1", "Vanessa's response: 'ability to deliver under pressure'"),
    ("Riley", "Ah yes, the classic 'I believe in you' which actually means 'I don't care about your concerns.'", "B2", "Casual translation of management speak"),
    ("Pat",   "Exactly.", "N5", "Agreement echo"),
    ("Riley", "Did anyone else push back?", "N1", "Question"),
    ("Pat",   "Marcus tried but she steamrolled him. You know how she gets in that mode.", "S1", "Coworker Marcus tried to push back"),
    ("Riley", "Yeah. Once Vanessa decides something, reasoning with her is like arguing with a wall.", "B2", "Casual characterization of Vanessa"),
    ("Pat",   "The worst part is I actually think real-time notifications IS a good feature. Just not with a 7-day timeline.", "B2", "Casual nuanced take on feature vs timeline"),
    ("Riley", "Totally. It's a Q1 thing, not a 'surprise, we're doing this in a sprint' thing.", "B3", "Temporal: should be Q1 not this sprint"),
    ("Pat",   "What do you think the VP will even care about in the demo? Honestly?", "N1", "Analytical question"),
    ("Riley", "The numbers. They always care about the numbers. Usage metrics, adoption rate, that kind of stuff.", "B1", "Context-enabling: Riley's knowledge of VP priorities"),
    ("Pat",   "Right. Which we HAVE. The dashboard already shows all of that.", "S1", "Fact: dashboard already has usage metrics"),
    ("Riley", "Then just make that the star of the show. The notification thing can be 'coming soon' on a roadmap slide.", "S2", "Tactical advice"),
    ("Pat",   "Smart. Yeah. I'll frame it that way.", "S3", "Decision: adopting Riley's demo framing strategy"),
    ("Riley", "You're going to be fine. The demo will be great.", "N1", "Supportive filler"),
    ("Pat",   "Thanks. I needed to hear that honestly.", "S5", "Emotional: vulnerability about needing reassurance"),
    ("Riley", "Of course. That's what work friends are for, right?", "N1", "Filler"),
    ("Pat",   "Ha, are we work friends now?", "B1", "Context-enabling: defining relationship"),
    ("Riley", "I mean we text outside of Slack now so I think that's legally binding.", "B2", "Casual fact about communication patterns"),
    ("Pat",   "Lol fair enough.", "N1", "Filler"),
    ("Riley", "How's the side project going btw? The analytics thing.", "N4", "Topic transition"),
    ("Pat",   "Oh, actually something really cool happened.", "N4", "Teaser opener"),
    ("Riley", "What?", "N1", "Prompt"),
    ("Pat",   "One of the people I interviewed — a Head of Growth at this Series B startup called Meridian — reached out and said she'd pay for early access.", "S4", "Life event: first potential customer interest"),
    ("Riley", "Wait, seriously?", "N3", "Surprised reaction"),
    ("Pat",   "Yeah! Her exact words were 'if you build this, we'd be your first customer. We're spending $3K a month on Mixpanel and barely using 20% of it.'", "S1", "Customer quote: $3K/month Mixpanel, 20% usage"),
    ("Riley", "Dude. That's huge. That's not just validation, that's someone with their wallet out.", "B2", "Casual assessment of business progress"),
    ("Pat",   "I know. Her name is Asha Patel. She runs a team of like 8 growth people.", "S1", "Customer name: Asha Patel, 8-person growth team"),
    ("Riley", "Have you thought about what you'd charge her?", "N1", "Question"),
    ("Pat",   "I'm thinking something like $200 a month for early access with a commitment to feedback. Basically design partner pricing.", "S1", "Early access pricing: $200/month with feedback"),
    ("Riley", "That's smart. Design partner model. Lock in the feedback loop.", "B2", "Casual: naming the business strategy"),
    ("Pat",   "Exactly. What do you think about offering a free month first though? Lower the barrier.", "S2", "Pat seeking opinion on pricing strategy"),
    ("Riley", "Hmm. I'd say no. If they're already willing to pay, charging from day one establishes value. Free stuff gets deprioritized.", "B2", "Casual pricing philosophy"),
    ("Pat",   "That's a really good point actually. OK, no free tier.", "S3", "Decision: no free tier for early access"),
    ("Riley", "Look at us doing business strategy during a work crisis.", "N1", "Meta-commentary humor"),
    ("Pat",   "Multi-tasking at its finest.", "N1", "Humor agreement"),
    ("Riley", "I do want to say though — I think you should really go for this. The fact that you have someone ready to pay before you've even built it is a huge signal.", "B1", "Context-enabling: Riley's business assessment"),
    ("Pat",   "Thanks. That actually means a lot.", "S5", "Emotional: gratitude for support"),
    ("Riley", "OK but first, survive the Vanessa demo. One crisis at a time.", "N1", "Practical humor"),
    ("Pat",   "Right. Priorities. Demo first, startup empire second.", "N1", "Humor agreement"),
    ("Riley", "That's the spirit. Hey I'm going for a run at lunch, want to grab coffee after?", "B3", "Temporal: lunch plans"),
    ("Pat",   "Yeah that sounds good. 1:30?", "B3", "Temporal: specific time"),
    ("Riley", "Perfect. Meet at the Blue Bottle downstairs?", "S1", "Coffee spot: Blue Bottle downstairs"),
    ("Pat",   "Works for me. See you then.", "N2", "Farewell"),
    ("Riley", "👍", "N1", "Filler"),
]
g4 = gaps_for(50, [60, 30, 45, 20, 90, 30, 45, 30, 120, 30, 20, 60, 30, 45, 90, 30, 20, 60, 30, 120])


# ═══════════════════════════════════════════════════════════════════════════
# SESSION 5: Getting casual (55 msgs, 2025-11-17 Mon)
# Riley had a bad training run (knee pain). Pat built a landing page.
# Wedding gift discussion for Jason/Elena.
# ═══════════════════════════════════════════════════════════════════════════
s5 = [
    ("Riley", "ugh monday", "N2", "Greeting + filler"),
    ("Pat",   "That bad already?", "N1", "Filler response"),
    ("Riley", "my long run yesterday was brutal. my knee started hurting around mile 14 and I had to cut it short at 16.", "S1", "Knee pain at mile 14, cut short at 16"),
    ("Pat",   "Oh no. Is it serious?", "N3", "Concerned reaction"),
    ("Riley", "idk. it's this dull ache on the outside of my left knee. could be IT band stuff.", "S1", "Symptom: outside left knee, possible IT band"),
    ("Pat",   "Have you seen a PT or anything?", "N1", "Question"),
    ("Riley", "not yet. I'm doing the runner thing where I pretend it's fine and hope it goes away.", "S5", "Emotional: avoidance behavior about injury"),
    ("Pat",   "Ha, I feel like that's not the recommended approach.", "B1", "Context-enabling: acknowledging avoidance isn't ideal"),
    ("Riley", "it is not. but my friend Nadia is a PT and she said I should foam roll and take a few easy days.", "S1", "Friend Nadia is a PT, advice: foam roll + easy days"),
    ("Pat",   "You should listen to Nadia.", "N1", "Advice filler"),
    ("Riley", "yeah yeah. it's just stressful because the marathon is 5 months out and I can't afford to miss training weeks.", "S5", "Emotional: training schedule anxiety"),
    ("Pat",   "What do you think about cross-training? Like swimming or biking? Lower impact.", "S2", "Pat's analytical suggestion"),
    ("Riley", "honestly yeah that's probably smart. there's a pool at the gym I never use.", "S3", "Considering cross-training: pool at gym"),
    ("Pat",   "There you go. Data says cross-training reduces injury risk anyway.", "B2", "Casual fact about cross-training benefits"),
    ("Riley", "lol 'the data says.' you're such an analyst even about running.", "S2", "Preference: Riley's perception of Pat as analytical"),
    ("Pat",   "I can't help it. It's a curse.", "N1", "Self-deprecating filler"),
    ("Riley", "anyway how was your weekend? did you work on the side project?", "N4", "Topic transition"),
    ("Pat",   "OK so I kind of went on a tear.", "N4", "Teaser opener"),
    ("Riley", "oh?", "N1", "Prompt"),
    ("Pat",   "I built a landing page. Like a real one. With a waitlist form and everything.", "S4", "Life event: built landing page"),
    ("Riley", "wait seriously?? when did you do this?", "N3", "Surprised reaction"),
    ("Pat",   "Saturday night. I started at like 8pm and looked up and it was 2am.", "B3", "Temporal: built it Saturday 8pm-2am"),
    ("Riley", "lol classic flow state. what's it look like?", "N1", "Question"),
    ("Pat",   "Pretty clean actually. I used this Tailwind template and customized it. Want to see?", "B2", "Casual fact: Tailwind template"),
    ("Riley", "obviously yes send it", "N1", "Enthusiastic filler"),
    ("Pat",   "metricflow.io — that's the name I landed on.", "S1", "Business name decided: MetricFlow, domain: metricflow.io"),
    ("Riley", "MetricFlow. That's actually really good. Clean and descriptive.", "B2", "Casual opinion on business name"),
    ("Pat",   "Thanks! I bought the domain at like 1am. Only $12. I took that as a sign.", "B2", "Casual fact: domain cost $12"),
    ("Riley", "haha $12 domain = destiny", "N1", "Humor filler"),
    ("Pat",   "Exactly. I also reached back out to Asha at Meridian to tell her I'm moving forward.", "S1", "Contacted Asha, moving forward"),
    ("Riley", "and??", "N1", "Prompt"),
    ("Pat",   "She's in. She wants to be the first design partner. We set up a call for next week to define her requirements.", "S4", "Asha confirmed as design partner, requirements call next week"),
    ("Riley", "Pat. You have a customer before you have a product. That's like the dream.", "B1", "Context-enabling: framing the significance"),
    ("Pat",   "I know. I keep waiting for reality to kick in.", "S5", "Emotional: disbelief at progress"),
    ("Riley", "don't let imposter syndrome win. this is legit.", "N1", "Supportive filler"),
    ("Pat",   "Thanks. Seriously.", "N1", "Gratitude filler"),
    ("Riley", "ok totally different topic. the Jason and Elena wedding gift. we said we'd go in together right?", "B1", "Callback to joint gift idea"),
    ("Pat",   "Yeah! Did you look at the registries?", "N1", "Question"),
    ("Riley", "I did. OK so there's a Le Creuset Dutch oven on the Williams-Sonoma one. The 5.5 quart in 'flame' orange. It's $380.", "S1", "Gift option: Le Creuset Dutch oven, 5.5qt, flame orange, $380"),
    ("Pat",   "Oh that's a great gift. Everyone loves Le Creuset.", "B2", "Casual opinion on gift"),
    ("Riley", "right? and it's the kind of thing nobody buys for themselves because it's so expensive.", "B2", "Casual reasoning for gift choice"),
    ("Pat",   "Perfect wedding gift logic. I'm in. So $190 each?", "S3", "Decision: splitting Le Creuset, $190 each"),
    ("Riley", "yep. I'll order it this week and we can wrap it together if you want.", "S3", "Action plan: Riley ordering"),
    ("Pat",   "Sounds perfect. Should we get it shipped to one of us or bring it to Napa?", "N1", "Logistics question"),
    ("Riley", "hmm let me check if they have gift wrapping. might be easier to ship it direct.", "B1", "Context-enabling: shipping logistics consideration"),
    ("Pat",   "Good call. Oh hey, are you driving to Napa for the wedding?", "N1", "Question"),
    ("Riley", "yeah I was going to. it's only like 90 minutes from here. you want to carpool?", "S3", "Carpool offer to Napa"),
    ("Pat",   "Oh that would be amazing actually. I wasn't looking forward to the drive alone.", "S5", "Emotional: relief about not driving alone"),
    ("Riley", "cool, we'll figure out logistics closer to December 14th.", "B3", "Temporal: wedding date callback"),
    ("Pat",   "Thanks Riley. For the ride and for... I don't know, being a good friend I guess.", "S5", "Emotional: friendship acknowledgment"),
    ("Riley", "oh god are we having a moment? quick someone make a sarcastic comment", "N1", "Humor deflection"),
    ("Pat",   "Ha. Moment over.", "N1", "Humor response"),
    ("Riley", "good. ok I gotta go ice this knee. talk later nerd", "N2", "Farewell"),
    ("Pat",   "Later. Take care of that knee!", "N2", "Farewell"),
    ("Riley", "🧊🦵", "N1", "Emoji filler"),
]
g5 = gaps_for(55, [30, 90, 30, 20, 60, 45, 30, 120, 30, 45, 20, 60, 30, 90, 30, 20, 45, 60, 30, 120])


# ═══════════════════════════════════════════════════════════════════════════
# SESSION 6: Riley's promotion + work-life balance (50 msgs, 2025-11-20 Thu)
# Riley gets promoted to Senior Engineer. Pat is happy. Work-life balance
# discussion. Pat's business: 3 beta signups.
# ═══════════════════════════════════════════════════════════════════════════
s6 = [
    ("Riley", "OK SO", "N1", "Dramatic opener"),
    ("Riley", "you're not gonna believe this", "N4", "Teaser"),
    ("Pat",   "what happened??", "N3", "Excited reaction"),
    ("Riley", "I just got out of my 1:1 with Diane", "B1", "Context: Diane is Riley's manager"),
    ("Pat",   "and??", "N1", "Prompt"),
    ("Riley", "she promoted me. Senior Engineer. Effective January 1st.", "S4", "Life event: promoted to Senior Engineer, Jan 1st"),
    ("Pat",   "RILEY", "N3", "Excited reaction"),
    ("Pat",   "That's incredible!! Congratulations!!", "N3", "Celebration reaction"),
    ("Riley", "haha thanks. I'm kind of in shock honestly", "S5", "Emotional: shock at promotion"),
    ("Pat",   "You absolutely deserve it. You've been performing at senior level for months.", "N5", "Affirming echo"),
    ("Riley", "that's actually what Diane said. she said the promo committee flagged me for the July cycle but she pushed for the December one instead.", "S1", "Promo details: flagged in July, Diane pushed for December"),
    ("Pat",   "Good for Diane. A manager who actually advocates for their people.", "B2", "Casual opinion on Diane's management"),
    ("Riley", "right? she's the anti-Vanessa lol", "B1", "Callback comparison to Vanessa"),
    ("Pat",   "Ha! That she is. So does this come with a raise?", "N1", "Practical question"),
    ("Riley", "yeah about 15%. plus RSU refresh. I don't have the exact numbers yet but Diane said it's significant.", "S1", "Raise: ~15% + RSU refresh"),
    ("Pat",   "That's great. What do you think about the new scope? Like are the expectations going to change a lot?", "S2", "Pat's analytical question about role change"),
    ("Riley", "yeah that's the thing. senior here means more ownership of technical direction and mentoring junior devs.", "S1", "Senior role: technical direction + mentoring"),
    ("Pat",   "You'll be great at the mentoring part. You're already basically doing that.", "B1", "Context-enabling: Riley already mentoring informally"),
    ("Riley", "thanks. the technical direction part scares me a little if I'm being honest", "S5", "Emotional: anxiety about increased responsibility"),
    ("Pat",   "Why?", "N1", "Question"),
    ("Riley", "like what if I make the wrong call on architecture and it costs the team months?", "S5", "Emotional: fear of high-stakes decision making"),
    ("Pat",   "That's just engineering though. Every decision carries risk. You make good calls.", "N1", "Reassurance"),
    ("Riley", "ha you sound like a leadership offsite right now", "B2", "Casual: Riley's humor-deflection pattern"),
    ("Pat",   "OK fair. But seriously, you've got good instincts. I've seen your code reviews.", "S2", "Pat's opinion: Riley has good instincts"),
    ("Riley", "thanks Pat. ok enough about my ego", "N4", "Topic deflection"),
    ("Pat",   "Wait before we move on — how are you feeling about work-life balance with the new role?", "S2", "Pat's probing question about balance"),
    ("Riley", "honestly that's what I'm most worried about. I already feel stretched between work and marathon training. adding more responsibility on top...", "S5", "Emotional: worried about balance"),
    ("Pat",   "Yeah I've been thinking about that a lot too. Like the side business takes up all my nights and the day job takes all my energy.", "B2", "Casual admission about time allocation"),
    ("Riley", "it's that thing where you have two things you care about and neither gets 100% of you", "B2", "Casual philosophy on divided attention"),
    ("Pat",   "Exactly. Sometimes I wonder if I should just pick one and go all in.", "S2", "Pat considering focus trade-off"),
    ("Riley", "but which one would you pick?", "N1", "Question"),
    ("Pat",   "Honestly? The business. If I'm being really honest with myself.", "S3", "Preference: would choose business over day job"),
    ("Riley", "then maybe that's your answer eventually", "S2", "Riley's indirect advice"),
    ("Pat",   "Maybe. Not yet though. I need more traction first.", "B1", "Context-enabling: Pat's timing criteria"),
    ("Riley", "smart. ok NOW can we talk about something else? this got too real", "N4", "Topic escape"),
    ("Pat",   "Ha, fine. How's the business update though? Real quick.", "N4", "Transition back"),
    ("Riley", "lol you're not great at 'something else' are you", "N1", "Humor"),
    ("Pat",   "I really am not. OK so: I have 3 beta signups on the MetricFlow waitlist. Including Asha.", "S1", "Business update: 3 beta signups, including Asha"),
    ("Riley", "three signups! when did those come in?", "N3", "Reaction"),
    ("Pat",   "Two came through the landing page organically. One from a LinkedIn post I did about analytics tooling.", "S1", "Signup sources: 2 organic, 1 LinkedIn"),
    ("Riley", "wait you did a LinkedIn post? mr 'I haven't told many people'?", "B1", "Callback to Pat's earlier vulnerability"),
    ("Pat",   "Ha, I didn't mention MetricFlow by name. Just shared thoughts on the mid-market analytics gap. But I put the link in the comments.", "S1", "LinkedIn strategy: thought leadership + link in comments"),
    ("Riley", "sneaky. I like it", "N1", "Approval filler"),
    ("Pat",   "Thanks. The call with Asha's team is tomorrow actually. I'm nervous.", "S5", "Emotional: nervous about customer call"),
    ("Riley", "you'll crush it. just remember — she already wants to pay you. you have the leverage.", "B1", "Context-enabling: reframing leverage"),
    ("Pat",   "True. I'm going to practice my demo tonight. Run through it a few times.", "S3", "Commitment: practicing demo"),
    ("Pat",   "OK I need to go prep for that. Celebrate your promotion tonight! You earned it.", "N2", "Farewell"),
    ("Riley", "oh I plan to. me and my foam roller are having a party", "B2", "Casual: Riley's recovery routine"),
    ("Pat",   "Lol. Congrats again, Senior Engineer.", "N2", "Farewell with title"),
    ("Riley", "🎉 thanks friend", "N2", "Farewell"),
]
g6 = gaps_for(50, [20, 30, 60, 30, 20, 45, 30, 90, 30, 60, 20, 45, 30, 120, 30, 60, 20, 45, 30, 90])


# ═══════════════════════════════════════════════════════════════════════════
# SESSION 7: Pre-Thanksgiving (45 msgs, 2025-11-24 Mon)
# Holiday plans. Riley's training over holidays. Pat's business: pricing at
# $49/month. Wedding is 3 weeks away.
# ═══════════════════════════════════════════════════════════════════════════
s7 = [
    ("Pat",   "happy almost turkey day", "N2", "Holiday greeting"),
    ("Riley", "happy almost turkey day to you too. any plans?", "N2", "Greeting + question"),
    ("Pat",   "yeah I'm going to my sister's place in Portland. She and her husband host every year.", "S1", "Thanksgiving: sister's in Portland"),
    ("Riley", "oh nice! Portland. I love that city. That's where the marathon is too right?", "B1", "Callback to Portland Marathon"),
    ("Pat",   "Ha, true. Maybe I'll scout the course for you while I'm there.", "N1", "Humor filler"),
    ("Riley", "lol please do. I think it goes through downtown and then along the waterfront.", "S1", "Marathon route: downtown + waterfront"),
    ("Pat",   "What about you? What are your Thanksgiving plans?", "N1", "Question"),
    ("Riley", "going to my parents' place in Eugene. about a 2 hour drive from here.", "S1", "Riley's Thanksgiving: parents in Eugene, 2hr drive"),
    ("Pat",   "Oh nice. Big family thing?", "N1", "Question"),
    ("Riley", "yeah pretty big. my parents, my brother Caleb and his wife, my aunt and uncle, a few cousins. like 12 people total.", "S1", "Family details: brother Caleb + wife, 12 people"),
    ("Pat",   "That sounds really nice.", "N1", "Filler"),
    ("Riley", "it is. my mom goes way overboard on the food. like there's always enough for 30 people even when there's 12.", "B2", "Casual fact: Riley's mom's cooking habits"),
    ("Pat",   "Ha, that's the mark of a good Thanksgiving.", "N1", "Filler"),
    ("Riley", "what about your sister? does she cook?", "N1", "Question"),
    ("Pat",   "Yeah she's actually a really great cook. Her husband Dave is too. They do a smoked turkey which is incredible.", "S1", "Sister's husband Dave, smoked turkey tradition"),
    ("Riley", "ooh smoked turkey sounds amazing", "N3", "Reaction"),
    ("Pat",   "It really is. They also do this sweet potato situation with bourbon and pecans that's life-changing.", "B2", "Casual food detail"),
    ("Riley", "ok stop I'm literally getting hungry", "N1", "Filler"),
    ("Pat",   "Ha sorry. How are you going to handle training over the holiday? Taking time off?", "N4", "Topic transition"),
    ("Riley", "I actually need to keep it going. My plan has me doing a 14-miler on Thanksgiving morning before dinner.", "S1", "Training plan: 14 miles Thanksgiving morning"),
    ("Pat",   "A 14 mile run before Thanksgiving dinner. You're a different breed.", "N3", "Reaction"),
    ("Riley", "lol it's actually perfect. I get to eat absolutely everything guilt-free.", "B2", "Casual: running-eating justification"),
    ("Pat",   "OK that's actually brilliant strategy.", "N1", "Filler"),
    ("Riley", "how's the knee btw?", "N1", "Self-question wait — wrong direction"),
    ("Pat",   "Wait I was going to ask you that. How's the knee?", "N1", "Question with callback"),
    ("Riley", "lol oh yeah. it's actually a lot better. Nadia gave me some exercises and I've been religious about them.", "S1", "Knee update: better, doing Nadia's exercises"),
    ("Pat",   "Oh good. Crisis averted?", "N1", "Question"),
    ("Riley", "I think so. I did 18 miles yesterday with zero pain.", "S1", "Long run: 18 miles, no pain"),
    ("Pat",   "That's awesome. OK business update since I know you're dying to ask.", "N4", "Topic transition"),
    ("Riley", "I was literally about to ask lol. go", "B1", "Context-enabling: mutual interest in business updates"),
    ("Pat",   "So the call with Asha went really well. We defined the core feature set together.", "S1", "Asha call: defined core features"),
    ("Pat",   "And I've been thinking a lot about pricing. I think I'm going to go with $49 per month per workspace.", "S1", "Pricing decision: $49/month per workspace"),
    ("Riley", "$49. that's below the 'needs a purchase order' threshold. smart.", "B2", "Casual pricing insight: purchase order threshold"),
    ("Pat",   "Exactly. Self-serve signup, credit card, no sales team needed.", "S1", "Business model: self-serve, no sales team"),
    ("Riley", "when are you aiming to launch?", "N1", "Question"),
    ("Pat",   "I want to have the MVP ready by early January. Beta launch mid-January.", "S1", "Timeline: MVP early Jan, beta mid-Jan"),
    ("Riley", "that's ambitious. but I think you can do it.", "N1", "Supportive filler"),
    ("Pat",   "Thanks. Oh hey — the wedding is in 3 weeks. Did you order the Le Creuset?", "B3", "Temporal: wedding in 3 weeks"),
    ("Riley", "YES. ordered it last week. getting shipped to my place. it's enormous btw.", "S1", "Le Creuset ordered, shipping to Riley"),
    ("Pat",   "Ha. We should coordinate wrapping it.", "N1", "Logistics"),
    ("Riley", "I have wrapping paper. come over the week before and we'll do it.", "B3", "Temporal: wrapping plan week before wedding"),
    ("Pat",   "Deal. Alright, I should finish up some stuff before EOD. Happy Thanksgiving, Riley.", "N2", "Farewell"),
    ("Riley", "happy thanksgiving Pat! eat all the sweet potato bourbon pecan stuff for me", "N2", "Farewell with callback"),
    ("Pat",   "Oh I will. Take care of that knee. 🦃", "N2", "Farewell"),
    ("Riley", "🦃🏃", "N1", "Emoji filler"),
]
g7 = gaps_for(45, [45, 30, 60, 20, 90, 45, 30, 120, 30, 20, 60, 30, 45, 30, 90, 30, 60, 20, 45, 120])


# ═══════════════════════════════════════════════════════════════════════════
# SESSION 8: Post-Thanksgiving deep conversation (55 msgs, 2025-11-28 Fri)
# Deep talk about goals and life direction. Riley opens up about why running
# matters. Pat decides to go part-time to focus on business. Most personal
# session. Talk about the upcoming wedding.
# ═══════════════════════════════════════════════════════════════════════════
s8 = [
    ("Riley", "hey. how was thanksgiving?", "N2", "Greeting"),
    ("Pat",   "so good. my sister outdid herself. how was yours?", "N1", "Response + question"),
    ("Riley", "really nice actually. the 14 miler beforehand was rough but then I ate my weight in mashed potatoes so it balanced out", "B2", "Casual callback to Thanksgiving run"),
    ("Pat",   "haha the runner's bargain", "N1", "Humor filler"),
    ("Riley", "exactly. hey can I tell you something kind of personal?", "N4", "Meta-conversation setup"),
    ("Pat",   "of course. what's up?", "N1", "Response"),
    ("Riley", "so being home with my family this week... it was great but it also made me think about a lot of stuff.", "S5", "Emotional: reflective mood from family time"),
    ("Pat",   "in what way?", "N1", "Prompt"),
    ("Riley", "like my brother Caleb has this whole life figured out. he's married, they just bought a house in Bend, he's a VP at his company. and I'm over here like... running in circles. literally.", "S5", "Emotional: comparing self to brother, feeling behind"),
    ("Pat",   "first of all, running in circles is very efficient. it's a loop.", "N1", "Humor to lighten"),
    ("Riley", "lol ok thank you for the geometry lesson", "N1", "Humor response"),
    ("Pat",   "but seriously, comparison is the thief of joy and all that. your life isn't less valid because it looks different from Caleb's.", "B2", "Casual philosophical take on comparison"),
    ("Riley", "yeah I know. it's just... sometimes I wonder what I'm actually working toward, you know? like I have this marathon goal and I have the new senior title but at the end of the day I come home to an empty apartment and eat takeout alone.", "S5", "Emotional: loneliness, questioning direction"),
    ("Riley", "you know what's funny? running is actually the thing that keeps me sane. like when I'm at mile 15 and everything hurts, there's no room to think about any of the other stuff. it's just me and the road.", "S5", "Emotional: running as mental health outlet"),
    ("Pat",   "I think the marathon isn't just about the time or the distance for you, is it?", "B1", "Context-enabling: Pat reading Riley's deeper motivation"),
    ("Riley", "no. honestly it's about proving to myself that I can commit to something hard and follow through. I've never been great at that.", "S5", "Emotional: deep vulnerability about commitment"),
    ("Pat",   "Riley, you literally just got promoted because you're great at following through.", "B1", "Callback to promotion"),
    ("Riley", "ha. work stuff is different though. it's the personal stuff I suck at.", "S5", "Emotional: distinction between work and personal follow-through"),
    ("Pat",   "I think the fact that you can even articulate that means you're more self-aware than you give yourself credit for.", "N1", "Supportive response"),
    ("Riley", "ok STOP being wise. it's my turn. what about you? what do you actually want?", "N4", "Topic reversal"),
    ("Pat",   "honestly? I'd go all in on MetricFlow. Like quit my job and build this thing full time.", "S2", "Core desire: full-time on MetricFlow"),
    ("Riley", "so why don't you?", "N1", "Challenge"),
    ("Pat",   "fear, mostly. the golden handcuffs. health insurance. the whole 'what if it fails' thing.", "S5", "Emotional: fear of leaving stable job"),
    ("Riley", "ok those are real concerns. but what if it succeeds?", "N1", "Counter-question"),
    ("Pat",   "then it changes everything.", "S5", "Emotional: weight of possibility"),
    ("Riley", "look you don't have to quit tomorrow. but have you thought about going part-time?", "B1", "Context-enabling: Riley suggesting the middle path"),
    ("Pat",   "actually... yeah. Diane approved a part-time arrangement for someone on the infrastructure team last year. Three days a week, prorated salary and benefits.", "S1", "Precedent: Diane approved 3-day/week part-time before"),
    ("Riley", "Pat. that's your answer. three days at work, two days on MetricFlow, keep your benefits.", "B1", "Context-enabling: Riley framing the plan"),
    ("Pat",   "I know. I've been running the numbers and I could make it work financially, especially with the savings I have.", "S1", "Financial feasibility: has savings"),
    ("Riley", "then do it. this is the most sure I've ever heard you about anything.", "B1", "Context-enabling: Riley reading Pat's certainty"),
    ("Pat",   "you're right. I'm going to talk to my manager in January. Ask for the part-time arrangement starting in February.", "S3", "Decision: ask for part-time in January, start February"),
    ("Riley", "January. February. written in stone now because I witnessed it.", "B3", "Temporal anchoring of the commitment"),
    ("Pat",   "ha. I guess it is. thanks for pushing me on this.", "S5", "Emotional: gratitude for being pushed"),
    ("Riley", "that's what friends are for. even ones who deflect with humor.", "N1", "Self-aware humor"),
    ("Pat",   "especially those ones.", "N1", "Affirming filler"),
    ("Riley", "ok this got very deep for a Friday morning after thanksgiving. the wedding! it's in two weeks!", "B3", "Meta-commentary + temporal: wedding in 2 weeks"),
    ("Pat",   "oh right! I need to figure out what to wear. Is it formal?", "N1", "Logistics question"),
    ("Riley", "the invite said 'cocktail attire.' so like... nicer than jeans but you don't need a tux.", "S1", "Dress code: cocktail attire"),
    ("Pat",   "navy blazer, dark jeans, nice shoes. you know, the silicon valley formal.", "B2", "Casual outfit description"),
    ("Riley", "haha 'silicon valley formal.' that's painfully accurate. I'm going with a dress probably.", "B2", "Casual outfit detail"),
    ("Pat",   "we should coordinate the carpool. what time do you want to leave?", "N1", "Logistics question"),
    ("Riley", "ceremony is at 4 right? so maybe leave by 2? gives us time for traffic and finding parking.", "S1", "Ceremony time: 4pm, leaving at 2pm"),
    ("Pat",   "sounds good. your car or mine?", "N1", "Question"),
    ("Riley", "mine. I have a Subaru Outback. it's ugly but reliable.", "B2", "Casual fact: Riley drives Subaru Outback"),
    ("Pat",   "perfect road trip car. alright, see you Monday.", "N2", "Farewell"),
]
g8 = gaps_for(45, [30, 45, 60, 20, 120, 30, 90, 30, 20, 60, 45, 30, 120, 30, 60, 20, 45, 30, 90, 30])


# ═══════════════════════════════════════════════════════════════════════════
# BUILD AND WRITE
# ═══════════════════════════════════════════════════════════════════════════
sessions = [
    (1, "2025-11-03", "2025-11-03T10:30:00", g1, s1),
    (2, "2025-11-06", "2025-11-06T09:15:00", g2, s2),
    (3, "2025-11-10", "2025-11-10T09:45:00", g3, s3),
    (4, "2025-11-13", "2025-11-13T10:00:00", g4, s4),
    (5, "2025-11-17", "2025-11-17T08:30:00", g5, s5),
    (6, "2025-11-20", "2025-11-20T14:00:00", g6, s6),
    (7, "2025-11-24", "2025-11-24T11:00:00", g7, s7),
    (8, "2025-11-28", "2025-11-28T09:00:00", g8, s8),
]

all_messages = []
for num, date, start, gaps, msgs in sessions:
    assert len(gaps) == len(msgs) - 1, \
        f"Session {num}: {len(gaps)} gaps for {len(msgs)} msgs (need {len(msgs)-1})"
    all_messages.extend(build(num, date, start, gaps, msgs))

# Verify total count
total = len(all_messages)
assert total == 400, f"Expected 400 messages, got {total}"

# Compute statistics
cat_counts = Counter(m["category"] for m in all_messages)
noise_total = sum(v for k, v in cat_counts.items() if k.startswith("N"))
signal_total = sum(v for k, v in cat_counts.items() if k.startswith("S"))
border_total = sum(v for k, v in cat_counts.items() if k.startswith("B"))
speaker_counts = Counter(m["speaker"] for m in all_messages)

output = {
    "conversation_id": "conv4",
    "speakers": {
        "Pat": "Analytical coworker, starting analytics SaaS (MetricFlow), asks 'what do you think'",
        "Riley": "Supportive coworker, Senior Engineer, training for Portland Marathon, humor as armor",
    },
    "date_range": {"start": "2025-11-03", "end": "2025-11-28"},
    "message_count": total,
    "category_distribution": {
        "noise": {k: v for k, v in sorted(cat_counts.items()) if k.startswith("N")},
        "signal": {k: v for k, v in sorted(cat_counts.items()) if k.startswith("S")},
        "borderline": {k: v for k, v in sorted(cat_counts.items()) if k.startswith("B")},
        "totals": {
            "noise": noise_total,
            "signal": signal_total,
            "borderline": border_total,
            "noise_pct": round(noise_total / total * 100, 1),
            "signal_pct": round(signal_total / total * 100, 1),
            "borderline_pct": round(border_total / total * 100, 1),
        },
    },
    "speaker_balance": {
        k: {"count": v, "pct": f"{v / total * 100:.1f}%"}
        for k, v in speaker_counts.items()
    },
    "messages": all_messages,
}

out_path = Path("benchmarks/gate_eval/datasets/gate_benchmark_conv4.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"✓ Wrote {total} messages to {out_path}")
print(f"  Noise:      {noise_total} ({noise_total/total*100:.1f}%)")
print(f"  Signal:     {signal_total} ({signal_total/total*100:.1f}%)")
print(f"  Borderline: {border_total} ({border_total/total*100:.1f}%)")
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

# Speaker balance check
for spk, cnt in speaker_counts.items():
    pct = cnt / total * 100
    assert pct <= 65, f"Speaker {spk} has {pct:.1f}% of messages (max 65%)"
    print(f"  {spk}: {cnt} ({pct:.1f}%) — {'OK' if pct <= 65 else 'OVER LIMIT'}")
