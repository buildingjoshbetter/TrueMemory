#!/usr/bin/env python3
"""Generate 200 benchmark questions for GateLoCoMo. Reads actual message IDs from conversation files."""

import json, re
from pathlib import Path

def load_conv(conv_id):
    """Load all messages for a conversation, merging parts if needed."""
    base = Path("benchmarks/gate_eval/datasets")
    msgs = []
    if conv_id == "conv1":
        for part in ["gate_benchmark_conv1_part1.json", "gate_benchmark_conv1_part2.json"]:
            with open(base / part) as f:
                msgs.extend(json.load(f)["messages"])
    else:
        n = conv_id.replace("conv", "")
        with open(base / f"gate_benchmark_conv{n}.json") as f:
            msgs.extend(json.load(f)["messages"])
    return msgs

def find_msg(msgs, pattern, speaker=None, category_prefix=None):
    """Find message IDs matching a content pattern."""
    results = []
    for m in msgs:
        if pattern.lower() in m["content"].lower():
            if speaker and m["speaker"] != speaker:
                continue
            if category_prefix and not m["category"].startswith(category_prefix):
                continue
            results.append(m["id"])
    return results

def find_first(msgs, pattern, speaker=None):
    r = find_msg(msgs, pattern, speaker)
    return r[0] if r else f"MISSING:{pattern[:30]}"

def find_all(msgs, *patterns, speaker=None):
    ids = []
    for p in patterns:
        r = find_msg(msgs, p, speaker)
        if r:
            ids.append(r[0])
    return ids

# Load all conversations
convs = {cid: load_conv(cid) for cid in ["conv1", "conv2", "conv3", "conv4", "conv5"]}

questions = []
qid = 0

def q(conv_id, question, gold_answer, evidence_patterns, difficulty="easy", category="fact", speaker=None):
    global qid
    qid += 1
    msgs = convs[conv_id]
    evidence = []
    for p in evidence_patterns:
        found = find_msg(msgs, p, speaker)
        if found:
            evidence.append(found[0])
    questions.append({
        "id": f"q_{conv_id}_{qid:03d}",
        "question": question,
        "conversation_id": conv_id,
        "gold_answer": gold_answer,
        "evidence_messages": evidence,
        "difficulty": difficulty,
        "category": category,
    })

# ═══════════════════════════════════════════════════════════════════════════
# CONV 1: Alex & Jordan (40 questions)
# ═══════════════════════════════════════════════════════════════════════════

# Facts (easy)
q("conv1", "Where does Alex work now?",
  "Alex works at Anthropic as a Senior ML Infrastructure Engineer.",
  ["title is Senior ML Infrastructure Engineer", "I GOT IT"])

q("conv1", "How much is Alex's base salary at the new job?",
  "Alex's base salary at Anthropic is $245k.",
  ["base is $245k"])

q("conv1", "What's Alex's total compensation at Anthropic?",
  "Alex's total comp is about $380k including RSUs over 4 years.",
  ["total comp is like $380k", "RSUs over 4 years"])

q("conv1", "Where is Jordan's new apartment?",
  "Jordan's apartment is on Balboa St between 6th and 7th Ave in the Inner Richmond neighborhood.",
  ["Balboa St, between 6th and 7th", "inner Richmond"])

q("conv1", "How much does Jordan pay in rent?",
  "Jordan pays $2,400 per month.",
  ["$2,400 a month"])

q("conv1", "What does Priya do for work?",
  "Priya is a product designer at Figma.",
  ["product designer at Figma"])

q("conv1", "Where did Alex and Priya go on their first date?",
  "They went to a wine bar called Wildhawk in Noe Valley.",
  ["wine bar in Noe Valley called Wildhawk"])

q("conv1", "What kind of music does Priya like?",
  "Priya is into indie music, specifically Japanese Breakfast and Mitski.",
  ["Japanese Breakfast, Mitski"], category="preference")

q("conv1", "Where is Alex's new office?",
  "Anthropic's office is on Mission St near Embarcadero, about a 20 minute bike ride from Alex's place.",
  ["Mission St near Embarcadero", "20 min bike"])

q("conv1", "What trail did Jordan hike at Point Reyes?",
  "Jordan did the Tomales Point trail, about 12 miles round trip.",
  ["tomales point trail", "12 miles"])

# Decisions
q("conv1", "When did Alex decide to give notice at Stripe?",
  "Alex planned to give notice on Monday, with a start date at Anthropic of January 6th.",
  ["telling Derek on Monday", "start date is January 6th"], category="decision")

q("conv1", "Is Jordan considering a career change?",
  "Yes, Jordan is thinking about switching to product management and applied for an associate PM role at Google.",
  ["I've been thinking about product management", "Google has one"], difficulty="medium", category="decision")

# Temporal
q("conv1", "When did Alex start at Anthropic?",
  "Alex's start date was January 6th.",
  ["start date is January 6th", "January 6th"], category="temporal")

q("conv1", "When did Jordan move into the new apartment?",
  "Jordan moved in on December 1st.",
  ["move in December 1st"], category="temporal")

q("conv1", "How long was Jordan's apartment search?",
  "Jordan searched for about two and a half months, since September, viewing about 15 places.",
  ["been looking since September", "15 places"], difficulty="medium", category="temporal")

# Emotional
q("conv1", "How was Alex feeling about the career change?",
  "Alex was nervous about interviewing but ultimately thrilled about the offer. Alex said they haven't been this excited about anything in a long time.",
  ["freaking out honestly", "haven't been this excited"], difficulty="medium", category="emotional")

q("conv1", "Was Alex unhappy at Stripe?",
  "Yes, Alex was very unhappy. Alex said the culture had gotten corporate and felt done with the place, especially under manager Derek.",
  ["culture at Stripe has just gotten so corporate", "I'm so done with this place"], difficulty="medium", category="emotional")

# Medium difficulty (connecting multiple messages)
q("conv1", "What was Alex's first project at Anthropic?",
  "Alex's first project was building model evaluation infrastructure — benchmarks and monitoring systems.",
  ["model evaluation infrastructure", "benchmarks and monitoring"], difficulty="medium")

q("conv1", "Who is Alex's manager at Anthropic and what's their background?",
  "Alex's manager is Ava, who previously worked at DeepMind.",
  ["manager is this woman named Ava", "DeepMind"], difficulty="medium")

q("conv1", "What features does Jordan's apartment have?",
  "Jordan's apartment is a corner unit with lots of natural light, hardwood floors, laundry in the building, and a small balcony off the bedroom.",
  ["corner unit", "hardwood floors, laundry", "balcony"], difficulty="medium")

q("conv1", "What happened with Jordan's heater issue?",
  "The heater was broken since Jordan moved in. After emailing the landlord three times and considering filing a city complaint, it was eventually fixed.",
  ["heater's broken", "emailed the landlord three times", "finally"], difficulty="medium")

# Hard (synthesis across sessions)
q("conv1", "How did Alex's career trajectory evolve over the conversation?",
  "Alex went from being unhappy at Stripe under manager Derek, to actively looking at ML/AI roles, interviewing at Anthropic (with help from Jordan's friend Lisa), getting the offer ($245k base, $380k total), and starting in January as Senior ML Infra Engineer.",
  ["culture at Stripe", "Anthropic has a role", "I GOT IT", "base is $245k", "first project is building"], difficulty="hard")

q("conv1", "How serious did Alex's relationship with Priya get over time?",
  "Alex and Priya matched on Hinge, had a first date at Wildhawk in Noe Valley, started seeing each other every weekend, Priya met Alex's parents, and by session 12 they'd been together about 3 months with Alex calling Priya before parents when sharing the job news.",
  ["matched on Hinge", "met my parents", "month and a half", "first person I called"], difficulty="hard")

q("conv1", "What career development was Jordan pursuing?",
  "Jordan took a Reforge PM course, started exploring product management as a career, applied to a Google PM role on the Cloud team, and got a phone screen interview.",
  ["Reforge", "Google has one", "phone screen"], difficulty="hard")

# Preferences
q("conv1", "What restaurant does Alex think has the best ramen in the city?",
  "Alex thinks Mensho on Valencia has the best ramen, particularly praising the chashu and their truffle tonkotsu.",
  ["Mensho", "best ramen in the city", "chashu"], category="preference")

q("conv1", "What's Jordan's opinion of the Richmond neighborhood?",
  "Jordan thinks the Richmond is underrated and loves the neighborhood, especially the restaurants on Clement St.",
  ["Richmond is underrated", "Clement St"], difficulty="medium", category="preference")

# Comparison
q("conv1", "Where does Jordan work?",
  "Jordan works at Dropbox.",
  ["first month at Dropbox"], category="fact")

q("conv1", "Who was Derek and why did Alex dislike him?",
  "Derek was Alex's manager at Stripe, a PM with no engineering background who came from McKinsey. He called team meetings to criticize performance and used consulting buzzwords.",
  ["PM named Derek", "McKinsey", "be more agile"], difficulty="medium")

# More facts
q("conv1", "What were the details of Jordan's housewarming party?",
  "Jordan planned a housewarming for Saturday the 25th with 15-20 people, featuring a taco bar and asking Alex to bring a JBL speaker.",
  ["Saturday the 25th", "15-20 people", "taco bar", "JBL"], difficulty="medium")

q("conv1", "What trip were Alex and friends planning?",
  "Alex and Priya were planning a group cabin trip to Tahoe for Presidents Day weekend, staying in a Truckee cabin that sleeps 8 at $400/night.",
  ["Tahoe", "Truckee", "$400"], difficulty="medium")

q("conv1", "How did Alex and Priya meet?",
  "They matched on the dating app Hinge about two weeks before their first date.",
  ["matched on Hinge"])

q("conv1", "What was Alex's interview process at Anthropic like?",
  "It was a full day with five rounds: system design, two coding rounds, behavioral, and a team lunch with the hiring manager.",
  ["five interview rounds", "system design, two coding"], difficulty="medium")

q("conv1", "Who helped Alex prepare for the Anthropic interview?",
  "Jordan's friend Lisa, who works on the research team at Anthropic, sent Alex interview tips about the system design round.",
  ["Lisa actually sent me some tips", "system design round is the hardest"], difficulty="medium")

q("conv1", "What hobby is Priya into?",
  "Priya is really into rock climbing and offered to take Alex to her gym, Dogpatch Boulders.",
  ["rock climbing", "Dogpatch Boulders"])

q("conv1", "When did Alex join Stripe?",
  "Alex joined Stripe in 2022, right after their layoffs.",
  ["2022 right after their layoffs"], category="temporal")

q("conv1", "What was the raise Alex got by switching to Anthropic?",
  "It was almost a 40% raise from Stripe.",
  ["40% raise"])

q("conv1", "Where did Alex and Priya go on their second date?",
  "They went to Sushi Sato in Japantown.",
  ["Sushi Sato", "Japantown"])

q("conv1", "What did Priya get promoted to?",
  "Priya was promoted to Senior Product Designer at Figma.",
  ["promoted at Figma", "Senior Product Designer"], difficulty="medium")

q("conv1", "Who was confirmed for the Tahoe trip?",
  "The confirmed attendees were Alex, Jordan, Priya, Anil (Jordan's coworker), and Mike. Sarah couldn't make it due to a wedding.",
  ["you me Priya Anil and Mike", "Sarah can't make it"], difficulty="medium")


# ═══════════════════════════════════════════════════════════════════════════
# CONV 2: Maria & Sam (40 questions)
# ═══════════════════════════════════════════════════════════════════════════

q("conv2", "What grad programs is Sam applying to?",
  "Sam is applying to clinical psychology PhD programs at Berkeley, UCLA, and UCSF.",
  ["clinical psych programs", "Berkeley, UCLA, UCSF"])

q("conv2", "What health issue was Maria diagnosed with?",
  "Maria was diagnosed with prediabetes based on high glucose levels in routine bloodwork.",
  ["prediabetes", "glucose levels were high"], category="life_event")

q("conv2", "How is Maria managing her prediabetes?",
  "Maria follows a meal plan, cuts down on sugar and refined carbs, and walks 2 miles daily with Sam's father along the river trail.",
  ["meal plan", "walk", "2 miles", "river trail"], difficulty="medium")

q("conv2", "What is Sam's research proposal topic?",
  "Sam's research proposal is on adolescent PTSD interventions, narrowed down with help from Professor Chen.",
  ["adolescent PTSD interventions", "Professor Chen"])

q("conv2", "Who is Professor Chen?",
  "Professor Chen is Sam's academic advisor who is writing a recommendation letter.",
  ["academic advisor", "write me one"])

q("conv2", "What school is Sam attending?",
  "Sam attends UC Davis.",
  ["Davis", "train from Davis"])

q("conv2", "What new hobby did Sam pick up?",
  "Sam started doing pottery after friend Jake took them to a class at a studio near campus.",
  ["pottery", "friend Jake"], category="life_event")

q("conv2", "How much did Sam's pottery course cost?",
  "The 6-week beginner series was $180 including materials and kiln time, then the intermediate course was $250 with a 15% discount.",
  ["$180 for the series", "$250", "15% discount"], difficulty="medium")

q("conv2", "What happened at Sam's first farmers market?",
  "Sam and Jake split a booth and sold 6 mugs and 3 bowls, making $180 total on their first weekend.",
  ["6 mugs and 3 bowls", "$180 total"], category="life_event")

q("conv2", "What did Sam make Maria for Christmas?",
  "Sam made a handmade pottery bowl and a little planter.",
  ["bowl and a little planter"])

q("conv2", "What school did Sam get into?",
  "Sam got accepted to UCLA's Clinical Psychology PhD program with full funding.",
  ["GOT INTO UCLA", "full admission with funding"], category="life_event")

q("conv2", "What's the financial package for Sam's PhD?",
  "UCLA offered full tuition waiver plus a $32,000 annual stipend for a 5-year program.",
  ["tuition waiver", "$32,000 annual stipend", "5-year program"], difficulty="medium")

q("conv2", "When is the family reunion?",
  "The reunion for Abuela's 80th birthday is on Saturday March 15th at Tía Carmen's house in San Jose.",
  ["March 15th", "Tía Carmen's house", "San Jose"], difficulty="medium", category="temporal")

q("conv2", "How many people are expected at the reunion?",
  "About 45-50 family members are expected.",
  ["45", "50"])

q("conv2", "Who is coming to Thanksgiving?",
  "Aunt Rosa (bringing tres leches cake), Uncle Miguel (driving from Fresno), and cousin Sofia (back from Barcelona).",
  ["Aunt Rosa", "Uncle Miguel", "Sofia"], difficulty="medium")

q("conv2", "What did Sam's personal statement focus on?",
  "Sam wrote about volunteering at a youth crisis center and how it inspired pursuing clinical psychology.",
  ["youth crisis center", "clinical psych"])

q("conv2", "What is Sam's father's profession?",
  "Sam's father is a surgeon.",
  ["surgery right now", "med school"])

q("conv2", "How many classes is Sam taking this semester?",
  "Sam is taking 5 classes: intro to psych, calc 2, English comp, bio, and a film studies elective.",
  ["5 this semester", "intro to psych, calc 2"])

q("conv2", "Who is Sam's favorite professor?",
  "Dr. Okafor, the film studies professor who did her PhD at NYU on documentary filmmaking.",
  ["Dr. Okafor", "NYU"], category="preference")

q("conv2", "Did Maria's health improve?",
  "Yes, Maria's glucose levels dropped significantly after following the meal plan and exercise routine, and her doctor said no medication was needed.",
  ["glucose levels dropped", "no medication needed"], difficulty="medium", category="life_event")

q("conv2", "What did cousin Sofia learn in Barcelona?",
  "Sofia learned to make paella during her semester in Spain and wanted to cook it for the family.",
  ["paella", "Spain"])

q("conv2", "How is Sam getting home for Christmas?",
  "Sam is driving down on December 20th with friend Jake, who has a Prius and lives in Sacramento.",
  ["dec 20th", "driving", "Jake", "Prius", "Sacramento"], difficulty="medium", category="temporal")

q("conv2", "What are Sam's top school choices?",
  "Berkeley is Sam's top choice because the clinical psych program is ranked top 5 nationally and it's close enough for weekend visits home.",
  ["Berkeley is my top choice", "ranked like top 5"], category="preference")

q("conv2", "When do the grad school decisions come out?",
  "Berkeley decisions come out in March, UCLA in late February, and UCSF in mid-March.",
  ["decisions come out in March", "late February", "mid-March"], difficulty="medium", category="temporal")

q("conv2", "What pottery did Sam make at the intermediate level?",
  "Sam progressed to making functional mugs and was learning glazing techniques and larger forms in the 8-week intermediate course.",
  ["mug this week", "intermediate", "glazing techniques"], difficulty="medium")

q("conv2", "Who is paying for Sam's intermediate pottery course?",
  "Maria offered to pay for it as a late Christmas gift.",
  ["pay for it", "late Christmas gift"], category="decision")

q("conv2", "What is Sam making as a gift for Abuela?",
  "Sam decided to make a handmade pottery vase for Abuela's 80th birthday.",
  ["vase", "birthday gift"], category="decision")

q("conv2", "How was Sam feeling about the application process?",
  "Sam was very stressed and overwhelmed, juggling finals, a bio paper, and the Berkeley application deadline simultaneously.",
  ["freaking out", "overwhelmed", "finals coming up"], difficulty="medium", category="emotional")

q("conv2", "What did Sam write about for UCSF?",
  "Sam wrote about a specific scenario from the crisis center where they helped a teenager having a panic attack, who later came back and asked for Sam specifically.",
  ["panic attack", "came back the next week"], difficulty="medium")

q("conv2", "What is the UCLA program structure?",
  "It's a 5-year program: first 2 years are coursework and the last 3 are research and clinical practice. Starting September.",
  ["5-year", "2 years are coursework", "3 are research"], difficulty="medium")

q("conv2", "How does Maria show she cares about Sam's pottery?",
  "Maria uses the bowl Sam made every morning for oatmeal, put the planter on the kitchen windowsill, and sent photos of Sam's work to Abuela who shows all her friends.",
  ["bowl you made me every morning", "windowsill", "sent her photos"], difficulty="hard")

q("conv2", "How has Sam's pottery journey progressed?",
  "Sam went from making a lopsided bowl in a beginner class to completing the 6-week series, making Christmas gifts, advancing to intermediate wheel-throwing, and selling at the Davis farmers market with Jake.",
  ["lopsided bowl", "6-week beginner", "intermediate", "farmers market"], difficulty="hard")

q("conv2", "What was Maria's health journey over the conversation?",
  "Maria went from discovering high glucose in routine bloodwork, to being diagnosed with prediabetes, starting a meal plan and daily walks with her husband, and eventually getting good news that glucose dropped enough to avoid medication.",
  ["glucose levels were high", "prediabetes", "meal plan", "dropped significantly"], difficulty="hard")

q("conv2", "When is the Berkeley application due?",
  "The Berkeley application was due December 15th.",
  ["berkeley's app is due december 15th"], category="temporal")

q("conv2", "How many people has Sam's grandmother invited to the reunion?",
  "The count grew to about 50 people as Abuela kept trying to invite more.",
  ["50 now", "keeps trying to invite more"])

q("conv2", "What food is Maria making for Thanksgiving?",
  "Maria is making green chile tamales that Sam loves.",
  ["green chile tamales"])

q("conv2", "Who is Sam's pottery friend?",
  "Jake, who goes to school near Sam's campus and whose parents live in Sacramento.",
  ["friend Jake", "Sac"])

q("conv2", "What subjects is Sam studying besides psych?",
  "Sam is taking calc 2, English comp, bio, and film studies in addition to intro to psych.",
  ["calc 2, english comp, bio", "film studies"])

q("conv2", "What happened when Sam submitted the Berkeley application?",
  "Sam was thrilled and relieved. Professor Chen had helped narrow the research proposal to adolescent PTSD interventions specifically.",
  ["submitted the Berkeley application", "adolescent PTSD interventions"], difficulty="medium", category="emotional")


# ═══════════════════════════════════════════════════════════════════════════
# CONV 3: Dev & Casey (40 questions)
# ═══════════════════════════════════════════════════════════════════════════

q("conv3", "How much were the flights to Japan?",
  "The flights were $890 round trip per person on ANA, direct from LAX.",
  ["$890 round trip on ANA"])

q("conv3", "When is the Japan trip?",
  "March 15 to March 29 — two full weeks.",
  ["March 15 to 29", "two full weeks"], category="temporal")

q("conv3", "What's the total budget for the Japan trip?",
  "About $4,780 total — flights $1,780, accommodation ~$1,370, and $1,850 for food/transport/activities.",
  ["$1780", "$1370", "$1850"], difficulty="medium")

q("conv3", "Where are Dev and Casey staying in Tokyo?",
  "An Airbnb in Shibuya, walking distance to Shibuya Crossing, at $95/night for 8 nights.",
  ["Airbnb in Shibuya", "$95/night"])

q("conv3", "What is the ryokan they booked in Kyoto?",
  "Kumo no Ue — 'above the clouds' — for 4 nights at $120/night including breakfast.",
  ["Kumo no Ue", "$120/night"])

q("conv3", "What cooking class did Dev book in Japan?",
  "A traditional kaiseki cuisine class in Kyoto on March 18th afternoon, $80 per person.",
  ["kaiseki cuisine", "March 18th", "$80"])

q("conv3", "Who is Gerald?",
  "Gerald is a name partner at Casey's law firm who yelled at Casey in front of the team over a comma in a filing.",
  ["Gerald", "comma in a filing"])

q("conv3", "What bootcamp is Casey doing?",
  "Designlab, a 6-month fully online UX design bootcamp starting January 6th, costing $6,500.",
  ["Designlab", "6 months", "$6,500"])

q("conv3", "Who is Casey's Designlab mentor?",
  "A senior UX designer at Spotify.",
  ["senior UX designer at Spotify"], difficulty="medium")

q("conv3", "How much does the kitchen renovation cost?",
  "Marco from Roma Construction quoted $18,000 with a 6-week timeline.",
  ["$18k", "Marco", "Roma Construction", "6 week"])

q("conv3", "What countertops did they choose?",
  "Calacatta-look quartz with white and grey veining, about $55-75 per square foot installed.",
  ["Calacatta-look quartz", "$55-75"], category="decision")

q("conv3", "How long did Dev's ramen broth take?",
  "12 hours — Dev started at 6am with pork neck bones.",
  ["12 hours", "6am", "pork neck bones"], difficulty="medium")

q("conv3", "What was Casey's salary at the law firm?",
  "Casey was making $95,000 with $1,400/month in student loan payments.",
  ["$95k", "$1,400/month"])

q("conv3", "How much law school debt does Casey have?",
  "Casey went to law school for 3 years and has $180,000 in debt.",
  ["$180k in debt"])

q("conv3", "What's Designlab's placement rate?",
  "89% placement within 6 months of graduating, with average starting salary in the $70-85k range for UX.",
  ["89%", "$70-85k"])

q("conv3", "Who did Dev hire for the renovation?",
  "Marco from Roma Construction — $18k quote, 6-week timeline, 4.9 stars on Yelp with 85 reviews.",
  ["Marco", "Roma Construction", "4.9 stars"], category="decision")

q("conv3", "When did Casey give notice at the firm?",
  "Casey gave two weeks notice with a last day of December 31st.",
  ["two weeks notice", "December 31st"], category="temporal")

q("conv3", "How did Casey's managing partner react to the resignation?",
  "Janet was actually nice about it, said she could tell Casey had been unhappy, and said the door was open if Casey ever wanted to come back.",
  ["Janet", "really nice about it", "door was open"], difficulty="medium")

q("conv3", "How did Gerald react to Casey quitting?",
  "Gerald just said 'noted' and went back to his laptop.",
  ["noted", "went back to his laptop"])

q("conv3", "What did Dev cook for the dinner party?",
  "Dev hosted a 5-course dinner party for 8 people, including a 48-hour miso marinade dish.",
  ["5-course dinner", "8 people", "48-hour miso"], difficulty="medium")

q("conv3", "What Japan Rail Pass did Dev buy?",
  "A 14-day JR Pass for $420 total for both of them, covering unlimited bullet trains.",
  ["Japan Rail Pass", "14-day", "$420"])

q("conv3", "How many nights in each Japanese city?",
  "4 nights in Kyoto, 8 nights in Tokyo, 2 nights in Osaka.",
  ["4 nights Kyoto, 8 nights Tokyo, 2 nights Osaka"])

q("conv3", "What's the Designlab curriculum structure?",
  "First 2 months are foundations (design thinking, user research, wireframing), then 2 months of UI design, then 2 months of portfolio building.",
  ["design thinking, user research", "UI design", "portfolio"])

q("conv3", "What suitcase did Dev recommend?",
  "An Away carry-on for $275 with a 20% off code.",
  ["Away carry-on", "$275", "20% off"])

q("conv3", "How did Casey's career change unfold?",
  "Casey went from being burned out at the law firm under Gerald, to researching UX design bootcamps, finding Designlab, deciding to quit, giving notice (last day Dec 31), and enrolling in the January 6th cohort.",
  ["burned out", "Designlab", "quit", "December 31st", "January 6th"], difficulty="hard")

q("conv3", "How did the renovation planning progress?",
  "Dev researched kitchen renovation costs ($15-20k range), got three contractor quotes, chose Marco from Roma Construction at $18k, selected Calacatta quartz countertops and subway tile backsplash, with work starting mid-January.",
  ["$15k-$20k", "three contractor quotes", "Marco", "Calacatta", "mid-January"], difficulty="hard")

q("conv3", "What's Dev's cooking progression?",
  "Dev started with NYT Cooking pasta recipes, progressed to making ramen completely from scratch (12-hour tonkotsu broth, homemade tare, chashu pork belly), and eventually hosted a full 5-course dinner party for 8 people.",
  ["pistachio pesto", "ramen from scratch", "12 hours", "5-course dinner"], difficulty="hard")

q("conv3", "What ingredients did Dev use for the ramen?",
  "Pork neck bones, dried shiitake, kombu, niboshi (dried sardines), ginger, garlic; plus a tare of soy sauce, mirin, sake, and brown sugar; and 2 lbs of braised pork belly for chashu.",
  ["pork neck bones", "niboshi", "tare", "pork belly"], difficulty="medium")

q("conv3", "How much money does Casey have saved?",
  "Casey has about $12,000 in savings, which would cover the bootcamp plus about 4 months of loan payments.",
  ["$12k in savings"])

q("conv3", "What freelance work is Casey considering during the bootcamp?",
  "Casey plans to do freelance legal research on a platform called LawClerk.",
  ["LawClerk", "freelance"])

q("conv3", "When does the kitchen renovation start?",
  "Marco said he could start mid-January, finishing by end of February — before the Japan trip.",
  ["mid-January", "end of February"], category="temporal")

q("conv3", "What noodles did Dev use for the ramen?",
  "Sun Noodle brand from the Japanese market on Sawtelle.",
  ["Sun Noodle", "Sawtelle"])

q("conv3", "What backsplash style did they consider?",
  "Dev suggested classic subway tile, and Casey liked the idea of herringbone subway tile.",
  ["subway tile", "herringbone"])

q("conv3", "How did Casey feel about quitting law?",
  "Casey was emotional and relieved — crying happy tears. Casey had been unhappy for months, feeling that 3 years of law school and $180k in debt led to getting screamed at about commas.",
  ["cry again but in a good way", "$180k in debt", "commas"], difficulty="medium", category="emotional")

q("conv3", "What was the Osaka accommodation?",
  "A hostel-hotel hybrid near Dotonbori for 2 nights at $65/night.",
  ["Dotonbori", "$65/night"])

q("conv3", "What did Dev offer to help with Casey's career change?",
  "Dev offered to cover a bigger share of rent for 6 months and handle more of the trip costs so Casey could focus on loan payments.",
  ["bigger share of rent", "6 months", "focus on the loan payments"], difficulty="medium", category="emotional")

q("conv3", "How many contractor quotes did Dev get?",
  "Three: Mike's Remodeling at $22k, a guy named Steve from Thumbtack at $14k (sketchy reviews), and Marco from Roma Construction at $18k.",
  ["$22k", "Steve", "$14k", "Marco", "$18k"], difficulty="medium")

q("conv3", "What was the travel fund status?",
  "Dev and Casey had $3,200 saved in their travel fund, with Dev adding $200/month since August.",
  ["$3200 in the travel fund", "$200/month since August"], difficulty="medium")

q("conv3", "What kind of sink were they considering?",
  "A Belfast sink, which runs about $400-600 for a good one.",
  ["Belfast sink", "$400-600"])


# ═══════════════════════════════════════════════════════════════════════════
# CONV 4: Pat & Riley (40 questions)
# ═══════════════════════════════════════════════════════════════════════════

q("conv4", "What marathon is Riley training for?",
  "The Portland Marathon on April 12th.",
  ["Portland Marathon", "April 12th"])

q("conv4", "What's Riley's marathon goal time?",
  "Sub-3:30 for the full marathon.",
  ["sub-3:30"])

q("conv4", "How many miles per week is Riley running?",
  "About 40 miles per week, five days running with two rest days.",
  ["40 miles a week", "five days"])

q("conv4", "What's the name of Pat's side business?",
  "MetricFlow (originally considered Clearview Analytics).",
  ["metricflow", "Clearview Analytics"], difficulty="medium")

q("conv4", "What does MetricFlow do?",
  "It's an analytics SaaS tool for mid-size teams that outgrow Google Analytics but can't afford enterprise tools like Amplitude.",
  ["analytics SaaS", "mid-size teams", "Google Analytics"])

q("conv4", "How much will MetricFlow cost?",
  "Pat settled on $49 per month per workspace.",
  ["$49 per month"])

q("conv4", "Who is Asha Patel?",
  "Asha Patel is Head of Growth at a Series B startup called Meridian. She was one of Pat's customer interviews and became the first design partner.",
  ["Asha Patel", "Meridian", "first design partner"], difficulty="medium")

q("conv4", "Who is Vanessa?",
  "Vanessa is Pat and Riley's project lead at work who keeps changing requirements, including trying to pivot the dashboard project two weeks before a VP demo.",
  ["Vanessa", "pivoting the dashboard", "two weeks"])

q("conv4", "What wedding are Pat and Riley attending?",
  "Jason and Elena's wedding on December 14th at Domaine Chandon in Napa Valley.",
  ["Jason and Elena", "December 14th", "Domaine Chandon"])

q("conv4", "What wedding gift did Pat and Riley get?",
  "They went in together on a Le Creuset Dutch oven (5.5 quart in 'flame' orange) from the Williams-Sonoma registry, splitting $190 each.",
  ["Le Creuset Dutch oven", "5.5 quart", "$190 each"], difficulty="medium", category="decision")

q("conv4", "What happened with Riley's knee?",
  "Riley developed IT band pain around mile 14 of a long run. A PT friend named Nadia recommended foam rolling and easy days.",
  ["IT band", "mile 14", "Nadia"], difficulty="medium")

q("conv4", "Did Riley's knee injury get better?",
  "Yes, Nadia gave exercises that Riley followed religiously, and Riley did 18 miles with zero pain.",
  ["18 miles", "zero pain"], difficulty="medium")

q("conv4", "What training plan is Riley using?",
  "The Pfitzinger 18/55 plan — an 18-week program with 5 runs per week building up to peak mileage.",
  ["Pfitzinger 18/55"])

q("conv4", "What app does Riley use to track runs?",
  "Riley tracks everything in Strava — splits, heart rate, elevation.",
  ["Strava"])

q("conv4", "What promotion did Riley get?",
  "Riley was promoted to Senior Engineer effective January 1st, with about a 15% raise plus RSU refresh.",
  ["Senior Engineer", "January 1st", "15%"], category="life_event")

q("conv4", "Who promoted Riley?",
  "Diane, who pushed it through early — Riley was originally flagged for the July cycle.",
  ["Diane", "July cycle"], difficulty="medium")

q("conv4", "How many beta signups does MetricFlow have?",
  "3 beta signups including Asha, with two coming organically through the landing page and one from a LinkedIn post.",
  ["3 beta signups", "landing page", "LinkedIn"], difficulty="medium")

q("conv4", "What did Asha say about being a customer?",
  "She said 'if you build this, we'd be your first customer' and mentioned spending $3K/month on tools that don't do what they need.",
  ["first customer", "$3K a month"])

q("conv4", "How did Pat handle Vanessa's pivot request?",
  "Pat wrote up a risk analysis and sent it to Vanessa on Friday. She backed off the full pivot and they compromised on adding two new metric views.",
  ["risk analysis", "backed off", "two new metric views"], difficulty="medium", category="decision")

q("conv4", "What's Pat's plan for going part-time?",
  "Pat decided to talk to their manager in January about a part-time arrangement starting in Q1, similar to what was approved for someone on the infrastructure team.",
  ["part-time", "January", "infrastructure team"], category="decision")

q("conv4", "Where is the wedding ceremony?",
  "At a vineyard called Domaine Chandon in Napa Valley. Ceremony is at 4pm.",
  ["Domaine Chandon", "4"])

q("conv4", "How does Riley know the wedding couple?",
  "Riley worked with Elena at a first job out of school — they were desk neighbors for two years.",
  ["Elena and I worked together", "desk neighbor"], difficulty="medium")

q("conv4", "How does Pat know the couple?",
  "Pat has known Jason since college.",
  ["known Jason since college"])

q("conv4", "What's Pat's Thanksgiving plan?",
  "Going to their sister's place in Portland, where she and her husband Dave do a smoked turkey every year.",
  ["sister's place in Portland", "smoked turkey"])

q("conv4", "What's Riley's Thanksgiving plan?",
  "Going to parents' place in Eugene with brother Caleb, his wife, and other family — about 12-15 people.",
  ["Eugene", "brother Caleb", "12-15 people"], difficulty="medium")

q("conv4", "How did Pat validate the business idea?",
  "Pat interviewed 12 people total — mostly product managers and growth leads — and every single one identified the same gap between free tools and enterprise solutions.",
  ["12 people", "product managers", "same gap"], difficulty="medium")

q("conv4", "What pricing did early customer interviews suggest?",
  "The sweet spot seemed to be $30-60 per month per seat.",
  ["$30-60 per month"])

q("conv4", "What's Pat's MVP timeline?",
  "Pat wants the MVP ready by early January with a beta launch mid-January.",
  ["early January", "mid-January"], category="temporal")

q("conv4", "What Italian restaurant do Pat and Riley know?",
  "Lucca's, on 4th street — a team dinner spot.",
  ["Lucca's", "4th"])

q("conv4", "Where did Vanessa get her pivot idea from?",
  "A Medium article about real-time analytics that she linked in the email.",
  ["Medium article"])

q("conv4", "What cross-training did Riley consider for the knee?",
  "Riley decided to try swimming at the gym pool.",
  ["pool at the gym"], category="decision")

q("conv4", "How has Pat's side business evolved?",
  "Pat went from a vague idea reading Indie Hackers, to customer interviews (12 total), validating the hypothesis, building a landing page (metricflow.io), getting first beta signups, signing Asha as design partner, and deciding to go part-time to focus on it.",
  ["Indie Hackers", "12 people", "landing page", "Asha", "part-time"], difficulty="hard")

q("conv4", "How did Pat and Riley's friendship develop?",
  "They went from professional coworkers of 8 months to genuine friends through bonding over Vanessa's management style, sharing personal goals, attending a wedding together, and supporting each other's ambitions.",
  ["8 months", "Vanessa", "carpool", "support"], difficulty="hard")

q("conv4", "What was the full Vanessa saga at work?",
  "Vanessa tried to pivot the dashboard project 2 weeks before a VP demo, wanted to replace the aggregation pipeline based on a Medium article, called an emergency meeting to add a real-time notification system with one week left, and steamrolled objections from team members like Marcus.",
  ["pivoting", "two weeks", "Medium article", "real-time notification", "Marcus"], difficulty="hard")

q("conv4", "What career tradeoff is Pat facing?",
  "Pat is torn between a stable job and the side business, ultimately deciding to go part-time in Q1 to pursue MetricFlow more seriously.",
  ["stable job", "the business", "part-time"], difficulty="medium", category="emotional")

q("conv4", "What did Riley say about the promotion's implications?",
  "Riley noted that senior means more ownership of technical direction and mentoring junior engineers, not just a title bump.",
  ["ownership of technical direction", "mentoring"], category="preference")

q("conv4", "What dress code is the wedding?",
  "Cocktail attire — nicer than jeans but no tux needed.",
  ["cocktail attire"])


# ═══════════════════════════════════════════════════════════════════════════
# CONV 5: Taylor & Morgan (40 questions)
# ═══════════════════════════════════════════════════════════════════════════

q("conv5", "Where does Taylor live?",
  "Taylor lives in Berlin, Germany, in the Kreuzberg neighborhood.",
  ["Berlin", "Kreuzberg"])

q("conv5", "How long has Taylor been in Berlin?",
  "About 3 years.",
  ["3 years ago"], category="temporal")

q("conv5", "Where does Taylor work?",
  "Taylor works at N26, a fintech/mobile banking company, as a product manager.",
  ["N26", "product manager"])

q("conv5", "What's the name of Morgan's baby?",
  "Olive, a girl, about 3 months old.",
  ["Olive", "3 months"])

q("conv5", "Who is Morgan's partner?",
  "Chris, they've been together about 5 years and aren't married yet.",
  ["Chris", "5 years", "not married"])

q("conv5", "How much is Taylor's rent in Berlin?",
  "About $900/month for a 2-bedroom in Kreuzberg.",
  ["900", "2-bedroom", "Kreuzberg"])

q("conv5", "How much do Morgan and Chris pay for their place?",
  "About $2,200/month for a 2-bedroom in Portland.",
  ["$2,200", "Portland"])

q("conv5", "Where did Taylor and Morgan go to college?",
  "University of Oregon.",
  ["UO"])

q("conv5", "Who is Lukas?",
  "Lukas is a German photographer Taylor is seeing. They met at a bar in Friedrichshain.",
  ["Lukas", "photographer", "Friedrichshain"])

q("conv5", "When is Taylor visiting Portland?",
  "February 14th through the 21st.",
  ["February 14th", "21st"], category="temporal")

q("conv5", "How much are flights from Berlin to Portland?",
  "About 700 euros round trip connecting through Amsterdam, about 14 hours total travel.",
  ["700 euros", "Amsterdam", "14 hour"])

q("conv5", "Where is Taylor staying during the visit?",
  "At Morgan and Chris's place — they have a spare room.",
  ["stay with us", "spare room"], category="decision")

q("conv5", "What's the famous club in Berlin Taylor mentioned?",
  "Berghain, in an old power plant in Friedrichshain.",
  ["Berghain", "power plant", "Friedrichshain"])

q("conv5", "What's Kreuzberg like?",
  "It's the artsy multicultural neighborhood with tons of Turkish food and street art everywhere.",
  ["artsy multicultural", "Turkish food", "street art"], category="preference")

q("conv5", "What did Morgan do before having Olive?",
  "Morgan was a graphic designer at an agency downtown in Portland.",
  ["graphic designer", "agency"])

q("conv5", "How much vacation does Taylor get?",
  "30 days a year.",
  ["30 days"])

q("conv5", "What's happening at N26 that worries Taylor?",
  "There are rumors of layoffs, and since Taylor is on a work visa, losing the job would mean about 3 months to find a new one or leave Germany.",
  ["layoffs", "work visa", "3 months"], difficulty="medium", category="emotional")

q("conv5", "What baby milestone did Olive recently hit?",
  "Olive did her first real smile, and at almost 4 months she started grabbing things — she reached out and grabbed Chris's finger.",
  ["first real smile", "grabbed Chris's finger"], difficulty="medium", category="life_event")

q("conv5", "What's Olive's sleep schedule like?",
  "She wakes up every 2-3 hours. Morgan and Chris do shifts but it's still brutal.",
  ["every 2-3 hours", "shifts"])

q("conv5", "Where did Taylor and Morgan's friend group end up?",
  "Kelsey is in NYC working in publishing, Dave is still in Eugene brewing beer, and Priya is doing a PhD at MIT in computational neuroscience.",
  ["Kelsey", "NYC", "publishing", "Dave", "Eugene", "beer", "Priya", "MIT"], difficulty="medium")

q("conv5", "What coffee shop did they go to in college?",
  "Tailored — they went every morning before class.",
  ["Tailored"])

q("conv5", "What house did they live in during college?",
  "A house on 18th Street during junior year.",
  ["18th"])

q("conv5", "What class did they bond over in college?",
  "Economics 301 — intermediate microeconomics, junior year.",
  ["econ 301", "intermediate micro"])

q("conv5", "What Portland spots does Morgan recommend?",
  "Powell's Books (massive, half-day trip), Voodoo Doughnuts, Canard restaurant, Alberta Street neighborhood, the Japanese Garden, and Forest Park for hiking.",
  ["Powell's Books", "Voodoo Doughnuts", "Canard", "Alberta Street"], difficulty="medium", category="preference")

q("conv5", "What's the weather like in Portland in February?",
  "Rain — just constant rain.",
  ["rain", "constant rain"])

q("conv5", "Is Morgan going back to work?",
  "Morgan is unsure — on parental leave and doesn't know when or if they'll go back to the design agency.",
  ["parental leave", "idk when or if I'll go back"], category="emotional")

q("conv5", "How long was Chris's parental leave?",
  "Chris went back to work after 6 weeks, leaving Morgan mostly alone during the day.",
  ["6 weeks", "mostly just me during the day"])

q("conv5", "Is Taylor learning German?",
  "Yes, taking classes twice a week, but struggling because everyone in Berlin speaks English.",
  ["twice a week", "everyone in Berlin speaks English"])

q("conv5", "When do Berlin clubs open?",
  "They don't open until midnight and go until Monday morning.",
  ["midnight", "Monday morning"])

q("conv5", "What reunion dinner is Morgan planning?",
  "Morgan wants to get some of the old college crew together for dinner while Taylor is visiting.",
  ["old crew together", "dinner"], category="decision")

q("conv5", "What's Taylor's flight itinerary?",
  "Arriving Portland February 14th at 4:15pm, departing the 21st at 10am, connecting through Amsterdam.",
  ["February 14th", "4:15pm", "21st", "10am"], difficulty="medium")

q("conv5", "How is Morgan emotionally handling new parenthood?",
  "Morgan opened up about postpartum struggles, feeling isolated, and the brutal sleep deprivation, though also experiencing joyful moments like Olive's first smile.",
  ["isolated", "sleep deprivation", "first real smile"], difficulty="medium", category="emotional")

q("conv5", "Does Taylor feel homesick?",
  "Yes, Taylor sometimes feels lonely and homesick in Berlin despite loving the city.",
  ["lonely", "homesick"], category="emotional")

q("conv5", "How did Taylor end up in Berlin?",
  "Taylor was at a startup in San Francisco for 2 years after school, then the N26 opportunity came up kind of randomly.",
  ["startup in San Francisco", "2 years", "randomly"], difficulty="medium")

q("conv5", "How long had Taylor and Morgan been out of touch?",
  "About 5 years — they were close in college but drifted apart.",
  ["5 years"])

q("conv5", "What activities are Taylor and Morgan planning?",
  "Powell's Books, Voodoo Doughnuts, possibly the Japanese Garden, Forest Park hike, Alberta Street, a reunion dinner, and a drive to the coast.",
  ["Powell's", "Voodoo", "Japanese Garden", "Forest Park", "coast"], difficulty="medium", category="decision")

q("conv5", "What is computational neuroscience?",
  "That's what their college friend Priya is studying for her PhD at MIT.",
  ["computational neuroscience", "MIT"])

q("conv5", "When was Olive born?",
  "About 3 months before the conversation, which takes place in November 2025 — so roughly August 2025.",
  ["3 months old"], category="temporal")

q("conv5", "What class did Professor Huang teach?",
  "The context references Professor Huang's 'terrible lectures' from their college days.",
  ["Professor Huang"])

q("conv5", "What job did Taylor have before N26?",
  "Taylor worked at a startup in San Francisco for about 2 years after graduating from University of Oregon.",
  ["startup in San Francisco", "2 years"])


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATE AND WRITE
# ═══════════════════════════════════════════════════════════════════════════

# Collect all valid message IDs
all_msg_ids = set()
for msgs in convs.values():
    for m in msgs:
        all_msg_ids.add(m["id"])

# Validate evidence messages exist
missing = 0
for q_item in questions:
    valid_evidence = [eid for eid in q_item["evidence_messages"] if eid in all_msg_ids]
    invalid = [eid for eid in q_item["evidence_messages"] if eid not in all_msg_ids]
    if invalid:
        missing += len(invalid)
    q_item["evidence_messages"] = valid_evidence

# Count per conversation
from collections import Counter
conv_counts = Counter(q_item["conversation_id"] for q_item in questions)
diff_counts = Counter(q_item["difficulty"] for q_item in questions)
cat_counts = Counter(q_item["category"] for q_item in questions)

output = {
    "benchmark": "GateLoCoMo",
    "version": "1.0",
    "total_questions": len(questions),
    "distribution": {
        "by_conversation": dict(conv_counts),
        "by_difficulty": dict(diff_counts),
        "by_category": dict(cat_counts),
    },
    "questions": questions,
}

out_path = Path("benchmarks/gate_eval/datasets/gate_benchmark_questions.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"✓ Wrote {len(questions)} questions to {out_path}")
print(f"  By conversation: {dict(conv_counts)}")
print(f"  By difficulty: {dict(diff_counts)}")
print(f"  By category: {dict(cat_counts)}")
print(f"  Evidence IDs removed (not found): {missing}")
empty_evidence = sum(1 for q_item in questions if not q_item["evidence_messages"])
print(f"  Questions with empty evidence: {empty_evidence}")
