# Choice-point spot check

40 points sampled at random from 787 (seed 7).

ONE question per point, and it is not about insight:

    Was this a real choice -- could this person, at this moment, realistically
    have taken at least two of the listed alternatives?

Write `VERDICT: real` or `VERDICT: not-real` on the line provided. Add
`NOTE:` freely. Do not judge whether any explanation is interesting; that is
what the forecast scoring is for, and a human judging it is how the last five
designs became uninterpretable.

Score the filled sheet with `python choice_points.py --score-spot-check`.

---

## 1. `bloomfield:#south_america-faragrotech:Letelier:688`  (test, 2024-12-03, blueberry_size)

```
[676] Rodriguez: Hola Equipo! Can someone help me check if HF uploaded a disc yesterday? It should have two small scans.
[677] Rodriguez: <@U0190EQ43E1> please your help
[678] McLafferty: Hi Francisco, yes these two scans were uploaded yesterday. Hortifrut / ARM 4-L / L21 / 2024-11-29 <https://dashboard.bloomfield.ai/?customerID=c62af268-6dba-4ee5-9482-7452f918b2f4&amp;ranchID=8b215153-f449-4179-a722-3b81afac07d5&amp;year=2024&amp;blockID=60a2b6aa-61eb-4b4a-bab1-903719c181c4&amp;scanID=770af7cc-c0fd-48fc-bd74-07497533f534&amp;feature=blueberry&amp;mapDimension=Per+Foot&amp;chartDimension=Per+Rowside&amp;ranchPageView=blockDetail&amp;measure=DetectionFact.avgCountPerMeter&amp;pointID=69e10b75-e2bc-302b-9d66-90388c51aa66>
[679] McLafferty: Hortifrut / ARM 4-L / L17 / 2024-11-29 <https://dashboard.bloomfield.ai/?customerID=c62af268-6dba-4ee5-9482-7452f918b2f4&amp;ranchID=8b215153-f449-4179-a722-3b81afac07d5&amp;year=2024&amp;blockID=11a1fd64-503b-4bc8-ae00-05a551998d58&amp;scanID=9486557f-db96-4844-b7ae-c823f800f412&amp;feature=blueberry&amp;mapDimension=Per+Foot&amp;chartDimension=Per+Rowside&amp;ranchPageView=blockDetail&amp;measure=DetectionFact.avgCountPerMeter&amp;pointID=282f4cfb-071a-3739-93e6-0c3bc1848f34>
[680] Rodriguez: Hi Laura, i saw those, but there should be other two scans with date yestarday
[681] McLafferty: Oh hmm I don't see anything else. Should be the same date 11/29? <@UV3K9KHPA> Can you check if there was a different upload yesterday from Hortifrut?
[682] kaufmann: No; just L17 and L21 yesterday.
[683] McLafferty: If you have any more information to share, <@U0677JT1HSL> what blocks are they looking for? Is the disk still in the smasher?
[684] Rodriguez: <@U073RDABE6M> please your help with the info
[685] Letelier: Hi all, yesterday we scanned ESP-2C-27 and SL-I-26 but we could not see them
[686] McLafferty: Is the disk still in the Hortifrut smasher? <@U073RDABE6M>
[687] Letelier: All the information it's completed
```

**Person:** Letelier  
**Situation:** McLafferty asked if the disk is still in the smasher, after Letelier said the info was complete.  
**Why it was open:** Letelier could have answered the question about the disk, ignored it, or conceded, but chose to re-raise the request to check the disks.

| | action | would look like | gives up |
|---|---|---|---|
|  | `CONCEDE` | Accept that the disk might be the issue and wait. | Immediate resolution of the missing scans. |
|  | `HOLD` | Reiterate the request to check the disks. | Addressing the 'disk in smasher' question directly. |
|  | `IGNORE` | Say nothing to the smasher question and wait. | Clarifying the disk status. |
| **<-- taken** | `RE_RAISE` | Bring up the missing scans again without new info. | Providing new information to help. |

**What they actually wrote:** Please can you check those disks?

VERDICT: 
NOTE: 

---

## 2. `bloomfield:#labeling:McLafferty:84`  (dev, 2024-02-26, purple_threshold)

```
[72] McLafferty: These look like very young vines to me, so I think they should be labeled. It is difficult to tell, with them being so thin, but because of the blue training tape holding them up, I think these are valid trunk labels.
[73] McLafferty: For the height of the labels, when we had done dormant season vine mapping we told them to label the trunk until it split into the two cordons or canes, so that is probably the guideline that they are still going off of? But this is inconsistent it seems. Hmm as long as we are detecting the trunk is this an issue?
[74] Berger: Thanks! I was unsure about these young vines because they are positioned rather close to the older ones (untypically close I would say) so I was wondering whether it's the young vine or some additional growth. With regards to your second question, the height of the trunk label might be pretty inconsistent in other images as well, and since we are training on tiles, for future performance improvement, we might want to make sure that all pieces that really belong to trunk (or what we want to consider as trunk) are labelled, because otherwise we introduce some confusion to the model
[75] McLafferty: That is a good point these young vines are planted so close. I have to ask <@U04M1Q0KGDT> the expert to confirm!
[76] McLafferty: We had said annotations should be below arms of the vine. In <@U058N0UUBU7>'s guidelines, I am not sure exactly how we would change this. Although <@U022UFDHLHG> In some of your examples they have stopped at a little stump, this 'arm' isn't going to be fruiting so we could maybe give them feedback on this?
[77] Prendiville: In the case of this vineyard it looks like they interplanted new vines in between established vines. So if the vine spacing was 10 ft, it's now 5 ft. Definitely look different, and likely will continue to look different for a while, since the new vines will struggle for resources more than the older vines did when they were planted. I've been asked in the past by customers if we can recognize "young vines." This can be complicated since there is an ambiguous phase of establishment where the trunk is establishing from one or more green shoots growing upwards. That may be a digression. As far as ...
[78] Berger: thank a lot <@U04M1Q0KGDT>! <@U058N0UUBU7> can you please add this example to the guidelines for future use? thank you!
[79] Ortega: yes!
[80] Berger: The second question is about cases where `trunk` label is rather small, though we do see a plant there - from the counting perspective we would want to include this to count, but from our definition it's not a `trunk` - should we consider broadening our definition of trunk maybe? or does it make sense to call trunk only the part before the first side growth / trunk split / etc.? I attach some examples of trunk labels / or absense of such
[81] Berger: some other examples where trunk is not labelled, and therefore, the plant won't be counted
[82] McLafferty: Oh this is problematic if the trunk isn't seen it isn't labeled, because we will want to count that vine!
[83] Berger: Since we label trunk as sth before the vine splits into cordons, we need to ensure that during data collection that part is well visible then. And what do you think about the images where some small part of the trunk is labelled / nothing is labelled (though some vertical parts are present in the image), does it make sense?
```

**Person:** McLafferty  
**Situation:** Berger asks if small or unlabelled vertical parts should be considered trunks, impacting vine count.  
**Why it was open:** McLafferty could have held her ground on the current definition or escalated to a formal review.

| | action | would look like | gives up |
|---|---|---|---|
| **<-- taken** | `CONCEDE` | Agree that the definition might need updating. | Maintaining the current trunk definition. |
|  | `HOLD` | Reiterate the current trunk definition. | Addressing the counting problem. |
|  | `ESCALATE` | Suggest a formal review of the trunk definition. | Trying to resolve it in chat. |

**What they actually wrote:** In this last image at least that small part at the 'top' of the trunk I think should be labeled. Maybe we do need to update the definition.

VERDICT: 
NOTE: 

---

## 3. `boeing:customer-airlines:Technical-Pilot-01:44`  (test, 2017-06-05, Instant messages from Boeing employee to (former) 737 Chief Technical Pilot, June 5, 2017, BATES Number TBC-T&I 549015 –)

```
[32] Beth-Pasztor: Hi, [REDACTED] I would appreciate a few minutes of your time, the topic is on Lion Air. Would it be possible to connect today or tomorrow? Please let me know, thanks for your time.
[33] Beth-Pasztor: Sounds good, would 12:30 pacific work for you? Thank you,
[34] Technical-Pilot-01: Morning, just got to Gatwick. First day in sim tomorrow
[35] Mark-Forkner: how were the flights?
[36] Technical-Pilot-01: Copy me in on emails if you dont mind, so that i can keep up to speed with what is going on at home, in particular RTL and wind additive Flight was good, but weird business seat layout on [REDACTED]
[37] Mark-Forkner: do you know if MAX sim in MIA has the overrun and speedbrake warnings activated, or capable of being activated?
[38] Technical-Pilot-01: Not bad, but i would probably choose another airline over their 787 I don't know. But I will fire of an email right now to find out
[39] Mark-Forkner: I already sent one to [REDACTED]
[40] Technical-Pilot-01: Good
[41] Mark-Forkner: Now friggin Lion Air might need a sim to fly the MAX, and maybe because of their own stupidity. I'm scrambling trying to figure out how to unscrew this now! idiots
[42] Technical-Pilot-01: WHAT THE F%$&!!!! But their sister airline is already flying it!
[43] Mark-Forkner: I know I've asked for a webex so we can thru this with the DGCA not sure if this is Lion's fault or DGCA yet
```

**Person:** Technical-Pilot-01  
**Situation:** Mark-Forkner is scrambling to address Lion Air's potential sim need and a training determination on Wednesday.  
**Why it was open:** Technical-Pilot-01 could offer help, give advice, or disengage from the problem.

| | action | would look like | gives up |
|---|---|---|---|
| **<-- taken** | `TRADE` | Offer to go down for a day to help. | His personal time and convenience. |
|  | `HOLD` | Offer advice or express continued frustration. | Directly assisting with the problem. |
|  | `DROP` | Change the subject to something unrelated. | Engaging with Mark-Forkner's immediate problem. |

**What they actually wrote:** Let me know if you need me to go down for a day while im there, not ideal but if we have to we have to

VERDICT: 
NOTE: 

---

## 4. `oppenheimer:32_russian_bomb:J-Oppenheimer:32`  (test, None, 32. The Soviet bomb, 1949)

```
[20] Bush: Not specifically.
[21] J-Oppenheimer: We still don't know if a Hydrogen bomb is technically feasible.
[22] Strauss: My understanding is that Teller first proposed it at Los Alamos.
[23] J-Oppenheimer: His designs have always been wildly impractical. You'd have to deliver by ox-cart not airplane.
[24] Strauss: If it could put us ahead again, Truman needs to know about it. And if there's a possibility that the Russians know about it from a spy at Los Alamos... we've gotta get going.
[25] J-Oppenheimer: There's no proof there was a spy at Los Alamos.
[26] Condon: They put it under the football stadium?
[27] Wilkins: The field's not in use, anymore.
[28] J-Oppenheimer: Just as well.
[29] Strauss: There were reports of espionage from Los Alamos-
[30] J-Oppenheimer: Unsubstantiated reports-
[31] Strauss: I've heard there were Communists on the project- were any of them involved with discussions of the Super?
```

**Person:** J-Oppenheimer  
**Situation:** Strauss asks if Communists on the project were involved in Super discussions, implying a security breach.  
**Why it was open:** Oppenheimer could acknowledge, deny, or ask for specifics regarding the Communist accusation.

| | action | would look like | gives up |
|---|---|---|---|
|  | `CONCEDE` | Acknowledge the possibility of Communists' involvement. | Defending the project's security and vetting. |
| **<-- taken** | `HOLD` | Deny knowingly employing Communists. | Appearing to be evasive or not fully transparent. |
|  | `DROP` | Ask for names of alleged Communists. | Taking a firm stance on the employment policy. |

**What they actually wrote:** We didn't knowingly employ any Communists.

VERDICT: 
NOTE: 

---

## 5. `bloomfield:#ai-vision:Lyubovsky:262`  (dev, 2024-03-07, purple_threshold)

```
[250] Berger: It's NMS from ultralytics that is the current bottleneck. In particular it's this <https://github.com/BloomfieldRobotics/monorepo/blob/73e0fc9ef0b44d8b53ae093785da7aef44f8383f/ai/segmentation/inference/yolo.py#L124|line> of the code used - inside it's using `torchvision.ops.nms` for non-max suppression on boxes
[251] Wolf: <@U053NF4DQV7> what is the general distribution of how often we are seeing the "unknown" flower class compared to a defined stage?
[252] Lyubovsky: Very rare <https://bloomfield.atlassian.net/wiki/spaces/AD/pages/12916949001/Blueberry-flower>
[253] Wolf: Thank you! Cc: <@U02CBGVPZ2L> / <@U06HLLXS5RP> Let's exclude other from any total counts
[254] Colbourn: <@U04E27894J2> <@U06DH8RDJ2K> Just so you see that we're excluding "other" from total counts (this means we'll have to sum all the individual flower counts in our queries).
[255] Colbourn: <@U06DH8RDJ2K> <@U022UFDHLHG> <@U044CULDG80> <@U053NF4DQV7> Christina is starting on flowers 2.0 with the extra flower classes. When someone gets the chance, can you update this schema with those classes so we know which versions they are available on? <https://docs.google.com/spreadsheets/d/1fTdghtoenn02loPH4YS9Gm8wJQW9KfAZ04-DcWN8wTQ/edit#gid=0>
[256] Lyubovsky: <@U044CULDG80> <@U024D0WMJAJ>, correct me if I'm wrong, but do we need to update ELT to filter on different flower classes?
[257] Rovani: yes
[258] McLafferty: If there are new names like blueberry_v6fit, please please add them to this spreadsheet for the front end team, so that we don't run into issues like we did this weekend.
[259] Chi: Thanks <@U053NF4DQV7>! Perhaps we could also add version/class/service names that are in development and just make a note that they're not live yet? That would help frontend development as well
[260] Lyubovsky: <@U06DH8RDJ2K> could you elaborate on that?
[261] Rovani: Maybe we could already add blueberry_v7?
```

**Person:** Lyubovsky  
**Situation:** Chi suggested adding in-development versions to the spreadsheet, and Rovani suggested blueberry_v7.  
**Why it was open:** Lyubovsky could have agreed, refused, or added it with a condition, choosing the latter.

| | action | would look like | gives up |
|---|---|---|---|
|  | `CONCEDE` | Agree to add blueberry_v7 immediately. | Prioritizing v6fit issue resolution. |
|  | `HOLD` | Reiterate the need to resolve v6fit issues first. | Proactive frontend development. |
| **<-- taken** | `TRADE` | Add v7 but with a condition or assumption. | A clear, unconditional addition. |

**What they actually wrote:** I'de want to resolve issues we'd been seeing with v6fit before starting to use v7, though it might make sense to add it

VERDICT: 
NOTE: 

---

## 6. `bloomfield:#ai-vision:Wolf:423`  (dev, 2024-04-15, purple_threshold)

```
[411] McLafferty: Should we still try to get some new labels on these? I guess we can discuss on Monday how to proceed.
[412] Lyubovsky: Other things that can be done: 1. Slower collection (to see if it helps) 2. Adjusting focus (very difficult) 3. From the modeling side, I'm hoping that adding blur during training will help.
[413] McLafferty: I wondered if we can ask them to go slower? Maybe for blocks with so many buds?
[414] Lyubovsky: I'm worried that it might not help. If it's a focus issue (blur is caused by the camera's focus being slightly off rather than motion blur) decreasing the speed won't help much, but having a single pass could give an indication
[415] Wolf: (if it's not a lens focus issue and it's caused by motion blur) we can ask them to slow down.... but we need to be really explicit/prescriptive
[416] Wolf: We should say "drive x kph" "set your exposure to YYY" for optimal results
[417] McLafferty: It's too bad we can't do some testing of this ourselves. Too many unknowns at the moment to try to direct them.
[418] Wolf: <@UBD0F3UH0> ^
[419] Lyubovsky: I'de noticed there were often smaller experiments that I'de run, where there was no effect on performance (ie. jpeg compression / color correction), and it's easy to loose track of them. I created a <https://bloomfield.atlassian.net/wiki/spaces/AD/pages/13053919238/Blueberry+Experiments|page> with links to the run just to have some reference. The page is specific to blueberries, but wanted to see if this was already being done somewhere else / hear any other suggestions.
[420] Ortega: For objects that are *really* far away from the camera but semi-distinguishable to the eye, what approach is preferred for annotation? For example, this example looks like berries but if I had to annotate them, I would not feel confident on if the count is correct. This plant is not a background plant, it's just far away from the camera. The options I see: - Mark as not-clear berry or flower - Ignore - Annotate and guesstimate
[421] Ortega: I didn't write that point ^ down, but there is something similar in this point too <https://dashboard.bloomfield.ai/?customerID=c62af268-6dba-4ee5-9482-7452f918b2f4&amp;ranchID=18cebeab-ecf3-4171-849c-b8bf48748d08&amp;year=2024&amp;feature=blueberry&amp;mapDimension=Per+Foot&amp;chartDimension=Per+Rowside&amp;ranchPageView=blockDetail&amp;measure=DetectionFact.avgCountPerMeter&amp;blockID=36f55022-9491-4b3b-9037-958c801f0795&amp;scanID=afcf37ba-6738-436c-9998-f6195eb6fa2e&amp;pointID=32797e7b-f149-32e9-a4fa-5e2db4db6c3a>
[422] McLafferty: <@U03JL7WMUCQ> <@UBD0F3UH0> <@U053NF4DQV7> This question relates to the thread about blur and how we want to handle situations like this where we are mostly guessing about what we are looking at.
```

**Person:** Wolf  
**Situation:** Ortega asks about annotation strategy for blurry, far-away objects. McLafferty links it to the blur discussion.  
**Why it was open:** Wolf could have offered an opinion, delegated the decision, or ignored the question.

| | action | would look like | gives up |
|---|---|---|---|
| **<-- taken** | `ESCALATE` | Delegate the decision to Ortega's manager. | Making the decision directly or contributing to the discussion. |
|  | `HOLD` | Offer an opinion on the annotation strategy. | Delegating the decision to someone else. |
|  | `DROP` | Ignore the question and move on to another topic. | Addressing the annotation strategy issue. |

**What they actually wrote:** I'm going to rely on <@UBD0F3UH0> here

VERDICT: 
NOTE: 

---

## 7. `oppenheimer:14_chevalier_approach:J-Oppenheimer:4`  (dev, None, 14. The Chevalier approach, 1943)

```
[0] J-Oppenheimer: I'm ashamed to ask.
[1] Haakon-Chevalier: Anything.
[2] J-Oppenheimer: Take Peter.
[3] Haakon-Chevalier: Sure.
```

**Person:** J-Oppenheimer  
**Situation:** Haakon-Chevalier has agreed to take Peter, but J-Oppenheimer wants to clarify the duration.  
**Why it was open:** J-Oppenheimer could accept the initial agreement or press for more specific terms.

| | action | would look like | gives up |
|---|---|---|---|
| **<-- taken** | `HOLD` | Reiterate the request for a longer duration. | Risk making the ask seem more burdensome. |
|  | `CONCEDE` | Accept the initial 'Sure' without further clarification. | Not getting the desired duration for Peter's stay. |
|  | `DROP` | Change the subject without clarifying the duration. | Clarity on Peter's stay, potentially leading to future issues. |

**What they actually wrote:** No, for a while, Hoke. A while.

VERDICT: 
NOTE: 

---

## 8. `bloomfield:#ai-vision:Lyubovsky:500`  (dev, 2024-04-22, purple_threshold)

```
[488] McLafferty: Oh okay so we should wait for the go ahead from you?
[489] Berger: Yes, Ryan pointed out that I need to redeploy dagster as well, I'm on it right now
[490] McLafferty: Thanks for explaining! :grapes:
[491] McLafferty: <@U022UFDHLHG> do you think this will be done today? I see we also have RBF scans to run, thanks!
[492] Berger: Yes, it will be done in the next hour
[493] Berger: <@U0190EQ43E1> <@U06DJRRSTA4> prod dagster is updated, you can start running the `grape_early_mid_season_v2`!
[494] Prendiville: Curly dock/<https://ipm.ucanr.edu/PMG/WEEDS/curly_dock.html|rumex crispus> (weed) is interfering with detections like early-cluster at WPBNORTH for Constellation. Hard to say how much, but we noted a few patches of this at the perimeter of the scan yesterday. The customer seemed curious to know if it would be detected, but even aside from that it is large/high enough to be an occlusion issue. They said "if it is an issue, maybe it will just be more encouragement to manage the weeds better." <https://dashboard.bloomfield.ai/?customerID=be07f1ab-5069-4157-ade0-3858bd032199&amp;year=2023&amp;feat ...
[495] Berger: I'll create a dataset for this scan so that we don't forget to improve the model in general given this weed - but not sure when it gets labelled and taken into account when iterating
[496] Saxena: <@UBD0F3UH0> <@U05GBS03F0W> *Camera calibration question:* if I want to scale the intrinsic matrix to account for an image resize, should I use rectified or un-rectified image dimensions? *Background:* In our points parquet file, we store camera info which contains, among other things, the intrinsic matrix of the camera and the height/width of the image. However, it seems to contain two sets of (height, width): ```{'height': 3008, 'width': 4112, 'rectified_height': 3000, 'rectified_width': 4096 }``` I'm assuming the first two refer to the image dimensions before rectification, and the latter t ...
[497] Saxena: This is the value of the intrinsic matrix: ```array([[2507.609, 0. , 2048. ], [ 0. , 2504.985, 1500. ], [ 0. , 0. , 1. ]])``` so assuming the camera center is at the center of the image, I guess the intrinsic corresponds to the rectified image?
[498] Patadia: You should use rectified img width and height since depth map is generated using rectified images. The height and width are there to ensure the ai pipeline doesn't break since historically they used that keys at some point
[499] Lyubovsky: <@U05GBS03F0W>, running v9FIT vs V9Tiling, heres' the results. TILING: `num_detections=Counter({'flower': 94, 'blueberry': 91})` FIT: `num_detections=Counter({'flower': 91, 'blueberry': 87})` There are a few extra detections identified, but it doesn't solve the underlying issue.
```

**Person:** Lyubovsky  
**Situation:** Lyubovsky has a question about the decision to rely on F1 instead of MAP.  
**Why it was open:** Lyubovsky could have asked the question, dropped it, or tried to find the answer elsewhere.

| | action | would look like | gives up |
|---|---|---|---|
| **<-- taken** | `HOLD` | Ask the question directly to the named individuals. | Trying to find the answer independently. |
|  | `DROP` | Decide not to ask the question at this time. | Getting clarity on the metric decision. |
|  | `IGNORE` | Keep the question to self and continue with other work. | Understanding the rationale behind the metric choice. |

**What they actually wrote:** <@U044CULDG80>, <@U022UFDHLHG> this is kind-of a random question, but I know we've been relying mostly on F1 instead of MAP. Is it documented anywhere why this decision was made?

VERDICT: 
NOTE: 

---

## 9. `bloomfield:#south_america-faragrotech:Letelier:955`  (test, 2024-12-20, blueberry_size)

```
[943] Rodriguez: Hi team! The latest scans from Bluegold - San Pedro are complete but no info is showing in dashboard or table view. An example is San Pedro - M1T4L10 - 2024/12/12
[944] McLafferty: Hi Francisco, This scan is loading for me. Have you tried to wait 2 minutes and then refresh the dashboard?
[945] McLafferty: <https://dashboard.bloomfield.ai/?customerID=dac71a00-0255-4fc9-a0a9-03731619e61e&amp;ranchID=bcdf4e91-83d1-4981-b094-f52573b099d1&amp;year=2024&amp;blockID=5f9ed399-c4d3-46f1-806e-0c7c3ebc8f25&amp;scanID=df83be88-9626-4f4f-8fcb-1c44c5df7c44&amp;feature=blueberry&amp;mapDimension=Per+Foot&amp;chartDimension=Per+Rowside&amp;ranchPageView=blockDetail&amp;measure=DetectionFact.avgCountPerMeter&amp;pointID=414c8441-5d25-3840-ae88-c3a686693e2e>
[946] Rodriguez: Hi Laura! Yes, me and Bluegold tried. Usually when a scan is ready it takes about 1 hour to appear the info in dashboard. It would be good in the future (not a priority) to delay the "complete" untill it is showing the info
[947] McLafferty: That is an interesting point I wonder if we should think about delaying the Completed status. <@U02CBGVPZ2L> <@U07TXUDKJU9> Let's check in on this delay again. I know there were some next steps to fix this that were discussed, I am not clear how much work it might be, and when we can focus on it. Thanks!
[948] Colbourn: That delay should go away when we get to the Point Generation / Batch Processing refactor where we'll track every step in much more detail than we do now.
[949] Letelier: Hi guys, we are in Peru making scans with Pedregal, tomorrow I will update the disks (2) in Chile in the smasher, please run late season model on monday for this, because this is a big potencial customer and we want to show them the results on tuesday. Thanks for your help
[950] McLafferty: <@U06DJRRSTA4> for eyes to run on Monday
[951] Mizerski: Cool once I see them pop up in pg over the weekend I'll run them!
[952] McLafferty: Thank you!!
[953] Letelier: <@U0190EQ43E1> <@U06DJRRSTA4> thanks!!!
[954] Mizerski: 3 of the 8 of these late_season scans should be on the dashboard.
```

**Person:** Letelier  
**Situation:** Mizerski reports 3 of 8 late_season scans are on the dashboard.  
**Why it was open:** Letelier could have just thanked Mizerski, but chose to ask about the status of the remaining scans.

| | action | would look like | gives up |
|---|---|---|---|
|  | `CONCEDE` | Acknowledge and accept the update. | Pushing for the remaining scans. |
| **<-- taken** | `HOLD` | Acknowledge and ask about the remaining scans. | Simply accepting the current status. |
|  | `DROP` | Acknowledge and move on without asking about the rest. | Getting a timeline for the remaining scans. |

**What they actually wrote:** <@U06DJRRSTA4> thanks, I saw them earlier this morning. The rest could be late today?

VERDICT: 
NOTE: 

---

## 10. `oppenheimer:19_tatlock_last:J-Oppenheimer:5`  (dev, None, 19. Last night with Jean, 1943)

```
[0] Tatlock: You left. Not a word. What did you think that would do to me?
[1] J-Oppenheimer: I wrote.
[2] Tatlock: Pages of nothing. Where'd you go?
[3] J-Oppenheimer: I can't tell you.
[4] Tatlock: Why not?
```

**Person:** J-Oppenheimer  
**Situation:** Tatlock asks why J-Oppenheimer can't tell her where he went.  
**Why it was open:** J-Oppenheimer could have continued to refuse, or given a vague answer, but chose to state the reason.

| | action | would look like | gives up |
|---|---|---|---|
|  | `HOLD` | Maintain his refusal without further explanation. | Addressing her direct question, potentially increasing her frustration. |
| **<-- taken** | `CONCEDE` | Tell her the real reason (she's a Communist). | Maintaining a less confrontational stance. |
|  | `TRADE` | Offer a vague or partial reason. | Full honesty, or full evasion. |

**What they actually wrote:** Because you're a Communist.

VERDICT: 
NOTE: 

---

## 11. `bloomfield:#ai-vision:McLafferty:326`  (dev, 2024-03-20, purple_threshold)

```
[314] McLafferty: It looks like we are missing quite a few closed flowers, they are noticeable in this scan. Would be great to hear what you think about this example image <@U058N0UUBU7> Hortifrut / ESP 2-Q / Q1 / 2024-03-19
[315] McLafferty: <https://dashboard.bloomfield.ai/?customerID=c62af268-6dba-4ee5-9482-7452f918b2f4&amp;ranchID=18cebeab-ecf3-4171-849c-b8bf48748d08&amp;year=2024&amp;blockID=36f55022-9491-4b3b-9037-958c801f0795&amp;scanID=f1381799-1b37-4597-8297-8f47f06f4bc1&amp;feature=blueberry&amp;mapDimension=Per+Foot&amp;chartDimension=Per+Rowside&amp;ranchPageView=blockDetail&amp;measure=DetectionFact.avgCountPerMeter&amp;pointID=03470b01-f314-3121-bba1-4ea9d46ed077>
[316] McLafferty: This variety is Milagro, which I don't believe we have done any annotation on? I think the example I shared for a far distance was also Milagro.
[317] Ortega: There are definitely a lot of missing flowers, but this scan seems very blurry. Maybe a combination of water on the camera + moving too fast? Are all the Milagro blocks like this?
[318] McLafferty: Hayden and I were chatting about the blurriness I feel like it's the same in every image across all Hortifrut scans? That you can't zoom in to see the very small features, do you think it's worse here than what you've seen? Did notice what looked like a bit of water at the top of the frame in some images.
[319] McLafferty: Hmm I'll have to take another look, now I'm wondering if it is more blurry, might help me to look at it with some color correction / brightening to see better.
[320] Ortega: I turned up the brightness for this point, I think it does help a bit. But for example what is circled in blue are missed detections, but the blurriness makes it hard to tell what stage of flower / berry is being missed. <https://dashboard.bloomfield.ai/?customerID=c62af268-6dba-4ee5-9482-7452f918b2f4&amp;ranchID=18cebeab-ecf3-4171-849c-b8bf48748d08&amp;year=2024&amp;blockID=36f55022-9491-4b3b-9037-958c801f0795&amp;scanID=f1381799-1b37-4597-8297-8f47f06f4bc1&amp;feature=blueberry&amp;mapDimension=Per+Foot&amp;chartDimension=Per+Rowside&amp;ranchPageView=blockDetail&amp;measure=DetectionFact.av ...
[321] Ortega: It's hard to tell if these detections are being missed bc 1. New variety 2. Blurriness 3. Combination of both?
[322] Lyubovsky: It does look very blurry. Is there a way to ask them to reduce exposure rate? On a side note, maybe they are increasing exposure rate to compensate for dark images on the dashboard/bloomgo?
[323] McLafferty: I wasn't thinking this was related to exposure.. we had asked them to turn it up one month ago, mostly because the plants that were farther away in scans were dark and missing detections.. This is an example from a Milagro scan from a couple weeks ago, at a different distance, do we think this looks better? Hortifrut / ESP 2-Q / Q4 / 2024-03-07
[324] McLafferty: This is an earlier scan that didn't go through batch processing because there is color correction on the dashboard, to compare. Hortifrut / ESP 2-Q / Q9 / 2023-12-05
[325] McLafferty: The raw image is darker..
```

**Person:** McLafferty  
**Situation:** Ortega is asking about blurriness and its causes. McLafferty previously suggested increasing exposure.  
**Why it was open:** McLafferty could have defended the exposure, conceded it was an issue, or dropped it as they did.

| | action | would look like | gives up |
|---|---|---|---|
|  | `HOLD` | Defend the previous exposure increase decision. | Acknowledging blurriness as a problem. |
|  | `CONCEDE` | Agree blurriness is an issue and exposure might be too high. | Defending the prior exposure increase. |
| **<-- taken** | `DROP` | Acknowledge the blurriness but move on without a solution. | Addressing the blurriness question directly. |
|  | `ESCALATE` | Suggest a meeting or formal investigation into blurriness. | Keeping the discussion informal. |

**What they actually wrote:** I noticed the water spot too. Thanks for looking at these I am not sure what to do about the blurry question.

VERDICT: 
NOTE: 

---

## 12. `oppenheimer:10_joining:J-Oppenheimer:15`  (dev, None, 10. Enlisting, 1942)

```
[3] J-Oppenheimer: That's not the point, Lawrence.
[4] Lawrence: What do you have in common with dock workers and farm laborers?
[5] Lomanitz: Plenty-
[6] Lawrence: Right. Everybody out. Now! Not you. What're you doing?!
[7] J-Oppenheimer: It's a trade union-
[8] Lawrence: Full of Communists!
[9] J-Oppenheimer: So? I haven't joined the party-
[10] Lawrence: They won't let me bring you onto the project because of this shit! They won't even let me tell you what the project is-
[11] J-Oppenheimer: I know what the fucking project is, Lawrence! We all heard about Einstein and Szilard's letter to Roosevelt. Warning him the Germans could make a bomb. And I know what it means for the Nazis to have a bomb.
[12] Lawrence: I don't?
[13] J-Oppenheimer: It's not your people they're herding into camps! It's mine!
[14] Lawrence: You think I tell them about your politics? Next time you're coming home from a meeting, take a look in the rear-view mirror. Listen for sounds on your phone line. And stop being so goddamn naive.
```

**Person:** J-Oppenheimer  
**Situation:** Lawrence is warning J-Oppenheimer about surveillance due to his politics, implying his activities are a security risk.  
**Why it was open:** J-Oppenheimer could have accepted the warning or ignored it, but chose to challenge its premise.

| | action | would look like | gives up |
|---|---|---|---|
| **<-- taken** | `HOLD` | Question Lawrence's premise that his activities are relevant to security. | Appear naive or defiant, potentially angering Lawrence further. |
|  | `CONCEDE` | Acknowledge the validity of Lawrence's warning and agree to be more careful. | Admit to a perceived security risk and give up some autonomy. |
|  | `DROP` | Change the subject or remain silent, letting the warning hang. | Leave the impression of either agreement or indifference to the warning. |

**What they actually wrote:** Why would they care what I do?

VERDICT: 
NOTE: 

---

## 13. `bloomfield:#south_america-faragrotech:Rodriguez:80`  (dev, 2024-04-10, purple_threshold)

```
[68] Rodriguez: Let me ask them. But i don't think so
[69] Prendiville: <@U0677JT1HSL> For the last Munger scan we scanned ~28 rows. Rescanning this at 1/4 we'd be collecting ~7 rows. Is that the plan for next time?
[70] Rodriguez: Okay, lets do it on monday. Thanks!
[71] Rodriguez: Hi <@U04M1Q0KGDT>, please let me know when the scan will be done. I need to call Carlos. Thanks!
[72] Rodriguez: Don't worry, just let me know if scan will be done as soon as possible please. And please let Carlos know also. If not possible, i think we will need to schedule for next week
[73] Rodriguez: Hello! Please let me know if there is a delivery date for API. Alejandro already send us the users who need access. Also, it is important that we provide the info of amount/% of plant scanned by row/block. Thanks!
[74] Wolf: <@U0677JT1HSL> Do we know when Smasher will be plugged back in to power?
[75] Rodriguez: Hi Hayden! Thanks for the update. Let's ask alejandro this info today. On Friday I'll schedule a meeting with the data team, do you think Ryan could join?
[76] Theofiledes: <@U0677JT1HSL> they can hit Green camera icon then look at "ROS NODES" section if they are all "ready" then wait for clearer sky . But I would say have them turn it off wait 10 sec then plug it back in .
[77] Ernst: Hi, <@U0677JT1HSL>, I just got off the call with our team member who is working on this. She is working on a PoC to validate our approach, and we worked together to understand the work involved so we could provide an estimate for when this will be ready. The non-technical summary is: - We plan to use AWS cross-account S3 functionality to load data from Bloomfield into an s3 bucket in the Hortifruit AWS account. - We are currently working on a proof-of-concept (PoC) of this with a separate test AWS account we created earlier this week. - We will use the PoC to track all the required changes in  ...
[78] Ernst: This is why I provided the details technical description of what we are doing.
[79] Rodriguez: <@U049HH0LWE4> do we have any update for hortifrut of the API topic? we have a follow up meeting today
```

**Person:** Rodriguez  
**Situation:** Rodriguez receives info from Hayden and asks for news about procedures or testing.  
**Why it was open:** Rodriguez could have just acknowledged the info, but he chose to re-raise a previous topic.

| | action | would look like | gives up |
|---|---|---|---|
|  | `HOLD` | Only acknowledge the info and not ask for more. | Opportunity to get updates on procedures/testing. |
| **<-- taken** | `RE_RAISE` | Bring up the procedures/testing topic again. | Focusing solely on the received info. |
|  | `ESCALATE` | Suggest a meeting to discuss procedures/testing. | A quick, informal update. |

**What they actually wrote:** Hi Hayden! thank you for the info, i´ll send it to the team. Do you have any news about the procedures or testint that was beign made?

VERDICT: 
NOTE: 

---

## 14. `bloomfield:#ai-vision:Rovani:199`  (dev, 2024-02-19, purple_threshold)

```
[187] Wolf: a serious amount of work
[188] Lyubovsky: Added a column comparing percentages. In general, don't think it's quite right to compare total counts due to occlusion + other factors (even though we are performing in a range of +- 30%). Regarding green percentages, we are consistently overcounting by 10% (this is very consistent). Pink &amp; Purple percentages are all over the place (we don't necessarily know that these have to align). Regarding Blue percentages, these are all over the place (+- 50). This is most concerning for me as we expect blue to align with our color classification. In a separate note: 1. It may be that differences be ...
[189] Lyubovsky: Actually, looking at the total percentage difference in blue, it is roughly +- 5 percent, which seems ok, but should be better.
[190] Wolf: Look at blue purple together
[191] Rovani: Awesome data! I copy-pasted the spreadsheet in <https://docs.google.com/spreadsheets/d/17KhP5AncNxNaOw9J_LWkAsP0lEtNwWar6VYBO988n5c/edit?usp=sharing|Google Sheet> so that we can comment and collaborate on it. Looking at percentages of percentages, especially when they are small is tricky so I added the different in percentage points. As <@U03JL7WMUCQ> suggested I regrouped Blue and Purple, as well as Green and Pink percentages, and it is indeed better. Then I assumed that the customer will use this data to prioritize collection, and would like to use it to send labor in the most promising plac ...
[192] Rovani: Of course it is a little weird to compare weeks, but it may represent better how they might use this data. A next step could be to compare the number of collected maduros: - using perfect prioritisation (using their counts) and summing the 3 or 4 best counts - our prioritisation (we sum the ground-truth maduro counts for the weeks identified as best using our counts) - random prioritisation (taking the average maduros count) This metric would show better the impact of our tool. Of course I am assuming a lot on the way they prioritize labor, but we could discuss with them to better understand h ...
[193] Lyubovsky: <@U03JL7WMUCQ> is there a justification for grouping Blue and Purple, as well as Green and Pink percentages?
[194] Wolf: <@U053NF4DQV7> I'm trying to reframe the problem. If we are pretty good at understanding if a berry is pink/green vs. blue/purple that allows us to remove/isolate the issue of correctly assigning true color to maturity stage. I'm trying to decouple the error here.
[195] Lyubovsky: Thanks. That makes sense to decouple the issue. Though looking at the Ventura verasion stages, our boundary between pink and purple doesn't align with the verasion stage boundary. However, the boundary between Purple and Blue does align, which is why I was expecting blue percentages to be correct, but am not sure about purple percentages.
[196] Saxena: In each detection parquet file, there is a `mask` field which is a set of image locations, e.g., ```[[[892, 2583], [891, 2790], [949, 3003], [1210, 3003], [1189, 2590]]]``` Do these refer to the vertices of a polygon? Is there any code in the monorepo to convert these to a full mask of the size of the image?
[197] Saxena: thanks!
[198] Lyubovsky: Inspired by this <https://bloomfieldrobotics.slack.com/archives/C04QS0R4SQJ/p1708335840071129|thread>, are there major risks to getting rid of 2021 data in training sets? Ie. It's probably not good to get rid of the data, yet if it's outdated (ie. we can't get original images) does this make sense? Or if 2021 data had unique characteristics, is there something in 2023 data that we can use to replace it with? This could apply to 2022 data as well.
```

**Person:** Rovani  
**Situation:** Lyubovsky asks about risks of removing 2021 data from training sets.  
**Why it was open:** Rovani could state a position, escalate to a meeting, or suggest a test to resolve the question.

| | action | would look like | gives up |
|---|---|---|---|
|  | `HOLD` | State a firm position on keeping or removing the data. | Flexibility and further investigation into the impact. |
|  | `ESCALATE` | Suggest a meeting to discuss the data removal risks. | Providing an immediate, actionable suggestion. |
| **<-- taken** | `TRADE` | Suggest testing the impact of removing the data. | A definitive answer without further work. |

**What they actually wrote:** I think we can test with/ without 2021 data

VERDICT: 
NOTE: 

---

## 15. `bloomfield:#ai-vision:Ernst:474`  (dev, 2024-04-16, purple_threshold)

```
[462] Rovani: Hi everyone, I just merged the <https://github.com/BloomfieldRobotics/monorepo/pull/685|PR> fixing the bug we were facing with tensorRT weights on smasher. In addition I created tensorRT weights for the `grape_early_mid_season_v1`: - speed for `blueberry_v7` is now *0.59 seconds per point instead of 0.8 seconds per point* - speed for `grape_early_mid_season_v1` is now *1.03 seconds per point instead of 1.8 seconds per point* - impact on precision and recall is negligible As proposed by <@U049HH0LWE4> and <@U053NF4DQV7> we can now define normal and tensorRT weights for each model. On smasher, t ...
[463] Turchin: Which dataset did we use for bencmarking?
[464] Rovani: The ones specified in the github workflow: - for blueberry, the one created to match detection count distribution for htf - for grape, the one <@U022UFDHLHG> created last week Using your benchmark file, we get the following results for blueberry_v7: - speed is now 1.19 secs/ point instead of 1.22 secs/ point
[465] Lyubovsky: Was there a change that we are no longer at 0.89 s for the `benchmark_htf` ?
[466] Rovani: I didn't know we achieved 0.89 s per point for `benchmark_htf`. Is there a PR where this result has been recorded?
[467] Rovani: Or a branch name I could test again?
[468] Rovani: I am rerunning the workflow on this branch
[469] Rovani: 0.89 secs/ point is `Average prediction time` ```"Average prediction time, s: 0.89",```
[470] Lyubovsky: Ah, yes. You are right. I was specifically looking at prediction time
[471] Ernst: <@U022UFDHLHG> the changelog is great stuff! Just to be sure, you have also redeployed the dagster image and image-processing image for this? Or does this still need to be done?
[472] Ernst: It is required for ELT, right? ELT will try to look up the service and we will get a `KeyError` I believe.
[473] Berger: Thank you! For visibility: I'm building the dagster image now (takes 30-60 minutes depending on machine), will deploy it to dagster-dev, and then we will collaborate with Ryan (?) on when to update dagster-prod to include this change and not interrupt any work
```

**Person:** Ernst  
**Situation:** Berger is building a dagster image and plans to deploy to dev, then prod.  
**Why it was open:** Ernst could have simply agreed with Berger's plan or escalated to a more formal review.

| | action | would look like | gives up |
|---|---|---|---|
| **<-- taken** | `HOLD` | Suggest a batch test on dev before moving to prod. | Letting Berger proceed with their planned deployment steps. |
|  | `CONCEDE` | Agree with Berger's plan for dev and prod deployment. | Ensuring a batch test is performed first. |
|  | `ESCALATE` | Require a formal review of the deployment plan. | Quick deployment; might be seen as bureaucratic. |

**What they actually wrote:** You can do a batch test if you would like by re-running a recent HTF scan on sm001 using dev-process-scan code location. If this works I think we are good to move to prod.

VERDICT: 
NOTE: 

---

## 16. `boeing:mcas-redesign-2016:SC-Engineer-01:26`  (dev, 2016-03-09, MCAS Stab Command requirements,)

```
[14] FlightControls-Engineer-01: To fix the low Mach flaps up stalls, we need to bring in up to 3 degrees of nose-down stab at Mach 0.2-Mach 0.5 (this is a conservative estimate based on what I know now).
[15] FlightControls-Engineer-01: I am not proposing any change at the moment to the command magnitude at the higher Mach numbers relative to the rollout configuration.
[16] FlightControls-Engineer-01: [REDACTED] Given the separation between Mach 0.3-0.5 and our critical conditions at Vd/Md, we are thinking that we don't have an issue for mistrim dive recovery with currently proposed MCAS schedule. But I'd like to make it clear we're still discussing that approach.
[17] FlightControls-Engineer-01: Please let me know if you have any questions or whether this would benefit from an in-person discussion. Thanks, [REDACTED]
[18] Loads-Engineer-01: Hi [REDACTED], I have just thought of 0.81PU command limit requirement (objective is 0.6 PU) on the MCAS function from the Static Loads (see below).
[19] Loads-Engineer-01: I don't know enough about this requirement's background but it is going to violate this requirement if the updated MCAS command requires more stab. We are just scoping out the MCAS updates right now but this is something we need to keep in mind. [REDACTED]
[20] Loads-Engineer-01: o This nose down limit is set by static loads and is determined by multiplying the 0.27 deg/s rate by a 3 second runaway, which is 0.81 PU.
[21] Loads-Engineer-01: Loads needs to meet the out of trim characteristics set by FAR section 25.255 which states that the airplane must be good during nose up and down directions during an out-of-trim situation resulting from a three-second movement of the longitudinal trim system.
[22] Loads-Engineer-01: Although the FAR states for the "normal rate", which would be 0.2 deg/s, loads has analyzed the larger mistrim in this requirement. Despite having this larger nose down range, it is intended that MCAS will not purposefully command the nose past 0.6 deg. [REDACTED]
[23] Loads-Engineer-01: Note:[REDACTED] [REDACTED]
[24] SC-Engineer-01: H[REDACTED]
[25] SC-Engineer-01: Thank you for highlighting this issue. It's one that we have been discussing internally within S&C as well since we also have a Boeing and FAR requirement that deals with the maximum mistrim associated with a stab run-away. There are a couple relevant points to the here.
```

**Person:** SC-Engineer-01  
**Situation:** SC-Engineer-01 has outlined their initial thoughts on the MCAS limit issue.  
**Why it was open:** SC-Engineer-01 could have pushed for a meeting or just finished their explanation, but offered both.

| | action | would look like | gives up |
|---|---|---|---|
| **<-- taken** | `HOLD` | Continue to explain their position electronically. | Opportunity for immediate, interactive clarification. |
|  | `ESCALATE` | Immediately schedule an in-person meeting. | Time to fully articulate their position in writing. |
|  | `DROP` | Finish their explanation without offering further discussion. | Ensuring full understanding and resolution of the issue. |

**What they actually wrote:** I'll outline them electronically but am happy to get together with you guys in-person to work through this and to make sure I understand any constraints.

VERDICT: 
NOTE: 

---

## 17. `boeing:flight-controls:FlightControls-Engineer-01:13`  (dev, 2014-05-01, 737 MAX Flight Controls/Pilots Meeting,)

```
[1] FlightControls-Engineer-01: Notes [REDACTED] [REDACTED] [REDACTED] [REDACTED] [REDACTED] [REDACTED] [REDACTED]
[2] SC-Engineer-01: Hi [REDACTED] Thanks for Coordinating the meeting and for the detailed notes! It's nice to have a mechanism to start tying up some of these lose ends. [REDACTED] Stability & Control [REDACTED]
[3] FlightControls-Engineer-02: MCAS - failure effects and annunciation They have released some coord sheets for failure effects ◦ With annunciation, failure is minor ◦ Without annunciation, failure is major Is it okay to not annunciate it, after all, what would the crew do?
[4] FlightControls-Engineer-02: Should it be annunciated with the existing SPEED TRIM FAIL light ◦ The speed trim system is not all that reliable - [REDACTED] MCAS does not operate within the 1.3g flight envelope ◦ So the probability of being in the flight regime and having a failure is [REDACTED]
[5] FlightControls-Engineer-02: Currently, we are stuck with the overhead light to annunciate the failures, as the flight control computers are not connected to the maintenance status message system
[6] FlightControls-Engineer-02: The failures that would cause MCAS to fail are almost all the same as the ones that would cause Speed Trim to fail. At the moment, autoflight plans to use the same light for failures of the additional signal that MCAS uses
[7] FlightControls-Engineer-02: The condition statement could be changed to note that MCAS is failed, but it might not even be necessary to let crews know that MCAS is on the airplane. It is just one of those automatic protection functions.
[8] FlightControls-Engineer-02: Since speed trim runs the trim wheel anyway, crews likely wouldn't distinguish if the wheel was moving for speed trim or MCAS.
[9] FlightControls-Engineer-01: As long as you need. Just let me know how much time you need so I can schedule it with them. Do you think you'll be ready to give this Friday the 16th?
[10] FlightControls-Engineer-01: I've recruited a few others from our group to help out. Is it alright if I let you know this Friday, after we have a better idea of how quickly it's coming together?
[11] FlightControls-Engineer-01: You bet, sounds good. Thanks for all the help on this.
[12] FlightControls-Engineer-01: [REDACTED]
```

**Person:** FlightControls-Engineer-01  
**Situation:** A collective decision was made to present AEG with systems commonalities at a higher level.  
**Why it was open:** They could simply state the new deadline, explain the strategic shift, or challenge the new strategy.

| | action | would look like | gives up |
|---|---|---|---|
| **<-- taken** | `HOLD` | Relay the new strategic decision and adjust the presentation timeline. | Allowing the previous plan to continue. |
|  | `RE_RAISE` | Question the strategic decision and advocate for the original plan. | Aligning with the collective decision. |
|  | `DROP` | Simply state the new deadline without explaining the strategic shift. | Providing context for the change. |

**What they actually wrote:** [REDACTED], Today we collectively decided it would be better strategically to present the AEG with the systems commonalities and differences at a bit of a higher level before we deep dive into the handling qualities issue.

VERDICT: 
NOTE: 

---

## 18. `bloomfield:#ai-vision:McLafferty:413`  (dev, 2024-04-11, purple_threshold)

```
[401] McLafferty: <@U058N0UUBU7> you are saying the example from 4/4 still has too much blur?
[402] Ortega: The blur looks a bit better I think! But still some guessing is going into what is being missed. For example, I think this could be a berry ? Possibly that line between petal fall / green berry. But it hard to know what the correct label would be here.
[403] McLafferty: Thanks! It is really difficult to tell. Not sure what we can do about this.
[404] Lyubovsky: Even with v9, there isn't significant improvements. Instead of detecting 34 flowers, it's detecting 37
[405] McLafferty: Hi <@U03JL7WMUCQ> <@U053NF4DQV7> the example from Francisco is exactly what we've been discussing here, that we are missing closed flowers. We talked about this again this morning, <@U058N0UUBU7> and I said we can look at some of the reviewed images on segments but I think we need to make a dataset from any recent Hortifrut Q block and update the guidelines with closed flower examples for Flipside because from what I've seen they are missing labeling them in the blurry dataset posted above.
[406] Wolf: do you think the examples he shared are too blurry/too dark to expect good performance?
[407] McLafferty: Hmm it's a good question, I'm not sure! The buds are so small it's going to be tough to count each one. But I think we can do better.
[408] McLafferty: And I'm not sure if saying these images are too dark / blur is the best answer ? I don't know how to try to fix that..?
[409] Lyubovsky: I would say that it is. Looking slightly to the top left of the center, there's a cluster with a brown open flower, and a pink open flower. I don't think a person or model could accurately say what the flower counts &amp; stages are there in the cluster. We can tell that the model is performing poorly, and it can do better in simple flower counts, but I don't think we can have flower stages in that region.
[410] McLafferty: Thanks Andrew! It's not the best news but you're right.
[411] McLafferty: Should we still try to get some new labels on these? I guess we can discuss on Monday how to proceed.
[412] Lyubovsky: Other things that can be done: 1. Slower collection (to see if it helps) 2. Adjusting focus (very difficult) 3. From the modeling side, I'm hoping that adding blur during training will help.
```

**Person:** McLafferty  
**Situation:** Lyubovsky suggests slower collection as a potential solution to blur.  
**Why it was open:** McLafferty could have agreed, proposed a test, or questioned the suggestion.

| | action | would look like | gives up |
|---|---|---|---|
| **<-- taken** | `HOLD` | Express agreement with the idea of slower collection. | Proposing alternative solutions or questioning the idea. |
|  | `CONCEDE` | Agree to implement slower collection immediately. | Further investigation into the cause of blur. |
|  | `ESCALATE` | Suggest a test run for slower collection to verify impact. | Immediate implementation of the suggestion. |

**What they actually wrote:** I wondered if we can ask them to go slower? Maybe for blocks with so many buds?

VERDICT: 
NOTE: 

---

## 19. `bloomfield:#south_america-faragrotech:Letelier:214`  (test, 2024-08-09, purple_threshold)

```
[202] Rodriguez: They will also continue doing this kind of trial for the next month
[203] Wolf: they can either start the scan well before the bags and just drive through OR they can 1. position themselves on the bags 2. start the scan wait for appx 30 seconds, 3. scan the test range, 4. stop on the bags, 5. wait for appx 30 sec. 6. stop the scan
[204] Letelier: <@U03JL7WMUCQ> thanks for the solution... When you say not soon we are talking about a week or more? Just to talk with HF and schedule the next meeting about the trials.
[205] McLafferty: <@U0677JT1HSL> The team is working on recovering those past test scans. My hope is they will be available next week.
[206] Ernst: <@U0677JT1HSL> can you share any details about the purpose of saving this data? So the situation is clear: When we talk about deleting old data, we *are only deleting the images*. *All calculated values, such as berry counts, flower counts, and anything produced as part of processing, will NOT be deleted.* The impact of deleting the image data is: - The *images would not be visible* in the dashboard when viewing the scan, *the scan and all calculated data from processing will still be visible on the dashboard*. ◦ You will still be able to view the scan and all of the associated calculated data ...
[207] Ernst: Makes sense! Thanks for the insight!
[208] Letelier: They are asking for wednesday at 8 am Peru time
[209] Deskins: I'm available, after 10am today and all morning tomorrow. Your choice!
[210] Deskins: Yep! I'm available tomorrow
[211] Letelier: Nice!!! Will please send me your email
[212] Letelier: Hi <@UBD0F3UH0> I agree with your summary. This is the lots they want to keep indefinitely:
[213] Letelier: <@U03JL7WMUCQ> about raspberries, no. Hortifrut has not collected anything yet.
```

**Person:** Letelier  
**Situation:** Letelier needs to schedule a meeting about raspberries and has an existing meeting with Hortifrut.  
**Why it was open:** Letelier could ask for a specific time, schedule separately, or try to combine meetings.

| | action | would look like | gives up |
|---|---|---|---|
| **<-- taken** | `HOLD` | Ask if the raspberry meeting can be before the Hortifrut meeting. | Flexibility in scheduling the raspberry meeting. |
|  | `DROP` | Schedule the raspberry meeting separately without linking it to the Hortifrut meeting. | Potential efficiency of back-to-back meetings. |
|  | `ESCALATE` | Suggest a joint meeting with Hortifrut to discuss raspberries. | Maintaining separate discussions. |

**What they actually wrote:** <@U03JL7WMUCQ> is it possible to make this meeting on tuesday, before our meeting with Hortifrut? Thanks

VERDICT: 
NOTE: 

---

## 20. `bloomfield:#ai-vision:Berger:484`  (dev, 2024-04-16, purple_threshold)

```
[472] Ernst: It is required for ELT, right? ELT will try to look up the service and we will get a `KeyError` I believe.
[473] Berger: Thank you! For visibility: I'm building the dagster image now (takes 30-60 minutes depending on machine), will deploy it to dagster-dev, and then we will collaborate with Ryan (?) on when to update dagster-prod to include this change and not interrupt any work
[474] Ernst: You can do a batch test if you would like by re-running a recent HTF scan on sm001 using dev-process-scan code location. If this works I think we are good to move to prod.
[475] Berger: But it's a grape model, I should do it for some BRF scan probably
[476] Ernst: Yes -- we need to ensure that it works for some small grape scan, but I also want to make sure the new dagster image does not break our ability to process blueberry scans. My assumption is that the dagster image did not have any changes that would prevent it from being backward-compatible with the currently deployed (on sm001) batch-processing image.
[477] Ernst: After this I think we can update the prod-process-scan location to use the same image.
[478] Berger: Ok, dev was just updated, so I'm now going to run two experiments: - recent small HTF scan (`28260bbb-14e3-4a5c-8507-38186d8f58da`) using `process_scan_job_on_smasher` - recent small BRF scan (not yet sure which) using `process_scan_job_v2`
[479] Ernst: <@U022UFDHLHG> I will make sure the GPU tasks are running on sm001. I don't think they are right now.
[480] Berger: how do you check this?
[481] Ernst: I usually check via the ECS console here: <https://us-east-2.console.aws.amazon.com/ecs/v2/clusters/blf-ue2-prod-smasher-001/tasks?region=us-east-2>
[482] Ernst: You can manually change the "desired count" for the `blf-ue2-prod-smasher-001-gpu-service` service. I just set it to 6.
[483] Berger: And where do you set it?
```

**Person:** Berger  
**Situation:** Ernst provided a link to check ECS console, but not the specific setting Berger asked about.  
**Why it was open:** Berger could have pressed for more detail, dropped the question, or assumed the link was sufficient.

| | action | would look like | gives up |
|---|---|---|---|
| **<-- taken** | `HOLD` | Ask Ernst to clarify the specific setting. | Opportunity to figure it out independently. |
|  | `DROP` | Move on without getting the specific answer. | Clarity on how to change the setting. |
|  | `CONCEDE` | Assume the link is enough and stop asking. | Specific knowledge of the setting location. |

**What they actually wrote:** Ah I think I know - in the `Update service`?

VERDICT: 
NOTE: 

---

## 21. `oppenheimer:16_la_science:J-Oppenheimer:28`  (dev, None, 16. The work, 1943)

```
[16] Nichols: I've informed General Groves you've been holding cross-divisional open discussions-
[17] Groves: Shut 'em down. Compartmentalization is the key to maintaining security-
[18] Condon: It's only the top men.
[19] Nichols: Who, presumably, communicate with subordinates.
[20] J-Oppenheimer: These men aren't stupid, they can be discrete.
[21] Groves: I don't like it.
[22] J-Oppenheimer: You don't like anything enough for that to be a fair test.
[23] Groves: Once a week. Top men only.
[24] J-Oppenheimer: I'd like to bring my brother here.
[25] Groves: No.
[26] J-Oppenheimer: I still haven't heard that my security clearance has been approved.
[27] Nichols: It hasn't.
```

**Person:** J-Oppenheimer  
**Situation:** His security clearance is still not approved, and Nichols advises him to wait before going to Chicago.  
**Why it was open:** Oppenheimer could go, wait, or demand action on his clearance.

| | action | would look like | gives up |
|---|---|---|---|
| **<-- taken** | `HOLD` | State his intention to go to Chicago despite the advice. | Appearing to follow protocol and authority. |
|  | `CONCEDE` | Agree to wait for his security clearance. | Maintaining project momentum and his authority. |
|  | `ESCALATE` | Demand an immediate resolution to his clearance. | Maintaining a calm and professional demeanor. |

**What they actually wrote:** We're going to Chicago tomorrow-

VERDICT: 
NOTE: 

---

## 22. `boeing:flight-controls:FlightControls-Engineer-01:35`  (test, 2016-06-16, Squawk for MCAS trim Event)

```
[23] SC-Engineer-01: Discussion of MCAS Characteristics: Last week a squawk was written concerning the inability to trim at 1.13Vsr due to MCAS activation. Previously we observed “ratcheting” of the stabilizer, heavier nose up column force during recovery, and delayed stabilizer return to trim.
[24] SC-Engineer-01: We would like to give you an update on MCAS characteristics (flight test data to illustrate the characteristics) and get your feedback. We have received Phase 1 CLAA approval and have begun an MCAS revision. We are scheduled to meet the next FCC box roll.
[25] SC-Engineer-01: Feel free to forward on the meeting notice. Thanks, [REDACTED] Aero S&C 737MAX
[26] FlightControls-Engineer-01: Meeting minutes from 6/22/2016 MCAS Review 1. Trim Capability – Squawked MCAS was not allowing pilot to trim at 1.13Vsr (would take out whatever trim was input). Resolution: move AOA trip higher to avoid low Mach 1.13Vs trims, easy fix / small work statement.
[27] FlightControls-Engineer-01: No real requirement violation, however it will reduce the work load when demonstrating cert conditions. Some discussion on fail high AOA or fail high Mach resulting in MCAS motion.
[28] FlightControls-Engineer-01: Conclusion: other systems will be reacting to the failure such as Mach trim or stick shaker; MCAS is small in comparison. No need to redesign to address this. All changes are minimal / low collateral damage, therefore no additional flight testing.
[29] SC-Engineer-01: [content redacted]
[30] SC-Engineer-02: [content redacted]
[31] FlightControls-Engineer-01: 📎 737 NG – PCIP [redacted] Consolidated Stabilizer Trim Architecture BCA 737NG, MAX, and Fleet Support: Flight Control Engineering, (2014-04-25) 737 NG – PCIP [REDACTED] Consolidated Stabilizer Trim Architecture BCA 737NG, MAX, and Fleet Support: Flight Control Engineering Meeting #5 Meeting Agenda: ○ Roll Call ○ VCCB Report Out (ETS: [REDACTED] and enter the code [REDACTED]) ○ ITRACS ○ Technical Reviews & Status ○ Issues and Concerns ○ Help Needed BOEING PROPRIETARY 4/25/2014 737 NG – PCIP [REDACTED] Consolidated Stabilizer Trim Architecture BCA 737NG, MAX, and Fleet Support: Flight Control E ...
[32] FlightControls-Engineer-01: 📎 737 NG – PCIP [redacted] Consolidated Stabilizer Trim Architecture BCA 737NG, MAX, and Fleet Support: Flight Control Engineering, (2014-04-25) (2/2) This document outlines engineering issues and concerns regarding the 737 stabilizer system, categorized by meeting number. It lists technical problems including incorrect electrical switch contact ratings, corrosion, excessive wire and ground return lengths, production break issues, transient suppression concerns, and specific wiring and bus configuration issues. It also notes uncertainty regarding the interaction between the PCIP project and th ...
[33] FlightControls-Engineer-01: 📎 737 MAX 8 MCAS Issues and Proposed Fix, (2015-07-06) BOEING 737 MAX 8 MCAS Issues and Porposed Fix [REDACTED] Primary Flight Control 07/06/15 Proprietary: The information contained herein is proprietary to The Boeing Company and shall not be reproduced or disclosed in whole or in part or used for any reason except when such user possesses direct, written authorization from The Boeing Company. Copyright © 2012-2013 Boeing. All rights reserved. BOEING PROPRIETARY 737-8 Airplane CDR - Presentation-Title | Section-Number - p.1 1 Agenda - MCAS/Speed Trim Interaction - Delta Stabilizer Estimation  ...
[34] FlightControls-Engineer-03: Test [REDACTED] Date 6/13/16 Setting up for cond [REDACTED] (Fup Sideslip at 1.13Vsr) - could not trim with stab due to MCAS input
```

**Person:** FlightControls-Engineer-01  
**Situation:** FlightControls-Engineer-03 reported inability to trim due to MCAS input during a test.  
**Why it was open:** They could confirm the squawk, escalate the issue, or ignore it and let the process unfold.

| | action | would look like | gives up |
|---|---|---|---|
| **<-- taken** | `HOLD` | Confirm that the issue will be officially squawked. | Investigating the issue further themselves. |
|  | `ESCALATE` | Call for an immediate meeting to address the test finding. | Following the standard squawk process. |
|  | `IGNORE` | Say nothing, assuming the squawk process will handle it. | Acknowledging the reported issue. |

**What they actually wrote:** [REDACTED] has agreed to officially squawk the inability to trim at 1.13

VERDICT: 
NOTE: 

---

## 23. `bloomfield:#ai-vision:Wolf:330`  (dev, 2024-03-20, purple_threshold)

```
[318] McLafferty: Hayden and I were chatting about the blurriness I feel like it's the same in every image across all Hortifrut scans? That you can't zoom in to see the very small features, do you think it's worse here than what you've seen? Did notice what looked like a bit of water at the top of the frame in some images.
[319] McLafferty: Hmm I'll have to take another look, now I'm wondering if it is more blurry, might help me to look at it with some color correction / brightening to see better.
[320] Ortega: I turned up the brightness for this point, I think it does help a bit. But for example what is circled in blue are missed detections, but the blurriness makes it hard to tell what stage of flower / berry is being missed. <https://dashboard.bloomfield.ai/?customerID=c62af268-6dba-4ee5-9482-7452f918b2f4&amp;ranchID=18cebeab-ecf3-4171-849c-b8bf48748d08&amp;year=2024&amp;blockID=36f55022-9491-4b3b-9037-958c801f0795&amp;scanID=f1381799-1b37-4597-8297-8f47f06f4bc1&amp;feature=blueberry&amp;mapDimension=Per+Foot&amp;chartDimension=Per+Rowside&amp;ranchPageView=blockDetail&amp;measure=DetectionFact.av ...
[321] Ortega: It's hard to tell if these detections are being missed bc 1. New variety 2. Blurriness 3. Combination of both?
[322] Lyubovsky: It does look very blurry. Is there a way to ask them to reduce exposure rate? On a side note, maybe they are increasing exposure rate to compensate for dark images on the dashboard/bloomgo?
[323] McLafferty: I wasn't thinking this was related to exposure.. we had asked them to turn it up one month ago, mostly because the plants that were farther away in scans were dark and missing detections.. This is an example from a Milagro scan from a couple weeks ago, at a different distance, do we think this looks better? Hortifrut / ESP 2-Q / Q4 / 2024-03-07
[324] McLafferty: This is an earlier scan that didn't go through batch processing because there is color correction on the dashboard, to compare. Hortifrut / ESP 2-Q / Q9 / 2023-12-05
[325] McLafferty: The raw image is darker..
[326] McLafferty: I noticed the water spot too. Thanks for looking at these I am not sure what to do about the blurry question.
[327] Ortega: Hortifrut / ESP 2-Q / Q4 / 2024-03-07 looks better, but a confounding factor may be that the blueberries are pretty mature here and there's fewer flowers than Hortifrut / ESP 2-Q / Q1 / 2024-03-19.
[328] Ortega: As for handling the blurriness, I think if we label any of these Milagro scans for missing closed flowers, we should prioritize annotating the least blurry scans.
[329] Ortega: Though long term, seems worth figuring out the question of why it's so blurry in the first place
```

**Person:** Wolf  
**Situation:** Ortega and Lyubovsky are discussing blurriness and potential causes related to exposure settings.  
**Why it was open:** Wolf could have offered an opinion, dropped it, or escalated to get more input, which they did.

| | action | would look like | gives up |
|---|---|---|---|
| **<-- taken** | `ESCALATE` | Pull in other team members to confirm the issue and solution. | Handling the issue directly or letting it drop. |
|  | `HOLD` | State own opinion on exposure without involving others. | Getting broader team consensus. |
|  | `DROP` | Let the blurriness discussion fade without a clear action. | Finding a resolution to the blurriness. |
|  | `TRADE` | Offer to investigate exposure settings in exchange for more data. | A direct answer to the exposure question. |

**What they actually wrote:** <@UBD0F3UH0> / <@UFLAM166A> / <@U0459CR7Y59> / <@U05R0BYS2CS> can one of you please confirm a) that too high of an exposure is the issue here b) what exposure we should explicitly tell them to set the cameras too

VERDICT: 
NOTE: 

---

## 24. `oppenheimer:18_security_interview:J-Oppenheimer:3`  (dev, None, 18. The security interview, 1943)

```
[0] Johnson: Dr.Oppenheimer, it's an honor. Please, take a seat-
[1] J-Oppenheimer: No need. I just wanted to check whether I should talk to Lomanitz while I'm here- given your concerns.
[2] Johnson: That's up to you, really, Professor. But I'd be cautious.
```

**Person:** J-Oppenheimer  
**Situation:** Johnson has advised caution regarding Lomanitz, and Oppenheimer has acknowledged it.  
**Why it was open:** Oppenheimer could have continued on Lomanitz, dropped it, or introduced a new topic.

| | action | would look like | gives up |
|---|---|---|---|
|  | `CONCEDE` | Drop the Lomanitz issue and wait for Johnson to ask about other topics. | Opportunity to proactively share information about Eltenton. |
|  | `HOLD` | Ask for more clarity on the Lomanitz caution. | Moving on from a potentially sensitive topic. |
|  | `DROP` | Move on to a different, less sensitive topic. | Sharing information about Eltenton. |
| **<-- taken** | `RE_RAISE` | Introduce a new, unrelated concern about Eltenton. | Further discussion on Lomanitz or waiting for Johnson's lead. |

**What they actually wrote:** Understood. Oh, and as far as the union goes, I wanted to give you a heads up on a man named Eltenton.

VERDICT: 
NOTE: 

---

## 25. `bloomfield:#ai-vision:Rovani:635`  (test, 2024-05-31, purple_threshold)

```
[623] Wolf: can you explain the third (xy) distribution?
[624] Ortega: Yes, the screenshot is a bit blurry, apologies. The two bar graphs are object pixel length estimation and object meter length estimation (which relies on pixel length estimation). The scatter plot is every single berry and the associated pixel length and object length.
[625] Ortega: So for example with yellow berries, the pixel / meter length is pretty consistent. There are fewer outliers in terms of size (both variables). Compared to the green scatter plot, the values are a little bit all over the place. This is likely due to the fact that the majority of detections used to make these plots were green. But inversely, the yellows being detected were pretty consistent in size (even if they are larger for some reason)
[626] Wolf: OK--I'm still a little confused, can I set up time for us to talk about this Monday <@U058N0UUBU7>?
[627] Ortega: Yes!
[628] Lyubovsky: Updated <https://bloomfieldrobotics.slack.com/archives/C019517GGUA/p1716414287313819|results> for error analysis with flowers: *Blueberries*: We see lots of issues with confusion between Purple, Pink, and Green due to the blue sky reflection issue (~80%). Most of these issues can be attributed to agv04, htf01 datasets (make up ~1500/2500 green or pink predictions for pink labels and agv04 makes up for 1300/2000 green or purple predictions for purple labels). The second most impactful issue is missing detections (~40%). These are issues where berries are in occluded/ darker regions/ have lower  ...
[629] Rovani: awesome! Let's plan a meeting to discuss sky reflection and how to confirm this
[630] Lyubovsky: Thanks. I'm not sure what our schedule will be this week. I'll ask Ricardo, and if there's a chance I'll add it to the calendar.
[631] Rovani: I created the meeting next week
[632] Rovani: I will centralize all the color investigation done so far
[633] Lyubovsky: Thanks. That's even better. The "Why are blueberries blue" confluence is pretty detailed
[634] Rovani: Maybe if you take phone pictures of the scene, we could confirm/ infirm sky reflection
```

**Person:** Rovani  
**Situation:** Rovani has suggested phone pictures to confirm sky reflection.  
**Why it was open:** Rovani could have waited for a response, dropped the point, or immediately added another potential cause.

| | action | would look like | gives up |
|---|---|---|---|
|  | `HOLD` | Wait for Lyubovsky's response to the phone picture idea. | Adding another potential cause immediately. |
|  | `DROP` | Let the phone picture idea stand alone without further input. | Suggesting an additional factor for consideration. |
| **<-- taken** | `RE_RAISE` | Bring up a previously discussed point about color correction. | Focusing solely on the new phone picture idea. |

**What they actually wrote:** it could be due to color correction too

VERDICT: 
NOTE: 

---

## 26. `bloomfield:#south_america-faragrotech:Mueller-Sim:143`  (test, 2024-06-21, purple_threshold)

```
[131] Wolf: can we get their comparison data
[132] Letelier: Hi all, people from Bomarea uploaded their information and they are asking me, just to understand, how long could take from In Progress to Completed. They are using the diaspora 024. Thanks
[133] Prendiville: <@U073RDABE6M> <@U0677JT1HSL> Could you find out from Hortifrut if they intend to scan recently pruned plants? These L blocks were scanned with intense coverage, but are freshly pruned and we have noticed false detections, particularly around the white cuts/pruning wounds. If this was intentional, our AI team is curious about the use case or what Hortifrut hoped to learn from the scans. From our side, the blueberry models have not been trained using dormant or recently pruned imagery. These images will be added to training and this should improve with the next version of the service, if they s ...
[134] Letelier: <@U04M1Q0KGDT> yes, they scan this blocks because Tim and I suggested them to scan this kind of plants to try to develop the model for counting of plants
[135] Letelier: Ok, understood. Thanks
[136] Rodriguez: Hola <@U049HH0LWE4> / <@U03JL7WMUCQ> could you please give us an update of the smasher situation and data upload? Do you think we will have it ready by tomorrow? Cerro Prieto also began to ask about when will they be able to see the info. We will tell them that we need to finish some configuration, but would be good to have more info. Thanks!
[137] Ernst: No worries. I am still working through some confusing AWS authorization issues. I have not yet started processing any scans yet.
[138] Prendiville: <@U073RDABE6M> <@U0677JT1HSL> Thanks for sending the updated report. I have adjusted our Hardware Status Tracker. We have questions regarding the areas in red (see image #1). A036 - Your recent document says it is at Hortifrut. Is that correct? Our records show that it was "grounded" at Agroberries-Bluegold. Was it moved for storage? A042 - Also shown as being at Hortifrut. However, this was last used to scan at Agroberries-Bluegold on 6/11 (see image #2). Has it been moved to another customer since then?
[139] Ernst: Thank you! Would you like me to keep forwarding messages like this to this channel?
[140] Prendiville: Hey Ricardo, do they want to delete the entirety of the disk?
[141] Letelier: <@U04M1Q0KGDT> yes, all the scans from today June 20 for customer Bluegold. They scanned a field that does not exist no the database
[142] Rodriguez: Perfecto! Thank you very much for the info, we will speek with the client as soon as possible.
```

**Person:** Mueller-Sim  
**Situation:** Rodriguez asks for an update on the smasher situation and data upload.  
**Why it was open:** Mueller-Sim could have escalated or dropped the issue, but chose to provide a clear directive.

| | action | would look like | gives up |
|---|---|---|---|
| **<-- taken** | `HOLD` | Explain the current focus and ask them to stop inserting disks. | Immediately solving all problems. |
|  | `ESCALATE` | Suggest a meeting to discuss the smasher issues. | Providing immediate instructions. |
|  | `DROP` | Acknowledge the issue but don't provide instructions. | Giving clear guidance to the team. |

**What they actually wrote:** <@U0677JT1HSL> <@U073RDABE6M> Yes, please ask them to stop inserting disks until Tuesday, when we'll have more information. We are focused on solving the Blue-Gold problem, which will solve the Cerro Prieto problem, but we need to do manual work to fix some of the issues and we don't have the bandwidth to do that until Blue-Gold is resolved

VERDICT: 
NOTE: 

---

## 27. `oppenheimer:28_postwar_la:J-Oppenheimer:9`  (test, None, 28. After, late 1945)

```
[0] Strauss: ...for not supporting their petition against bombing Japan.
[1] Morrison: This was taken 31 days after the bombing. Virtually everyone in the street for nearly a mile around was instantly and seriously burned by the heat of the bomb. The hot flash burned suddenly and strangely.
[2] Robert-Serber: The Japanese told us of people who wore striped clothing upon whom the skin was burned in stripes.
[3] Morrison: There were many who thought themselves lucky, who crawled out of the ruins of their homes only slightly injured. But they died anyway. They died days or weeks later from the radium- like rays emitted in great numbers at the moment of the explosion.
[4] Teller: Did you read this crap in the papers? A British physicist saying the atomic bombings weren't the last act of World War Two but the first act of this cold war with Russia.
[5] J-Oppenheimer: Which physicist?
[6] Teller: I think you knew him. Patrick Blackett?
[7] J-Oppenheimer: He may not be wrong. We bombed an enemy that was essentially defeated.
[8] Teller: Robert, you have all the influence now. Urge them to continue my research on the Super.
```

**Person:** J-Oppenheimer  
**Situation:** Teller asks J-Oppenheimer to urge continued research on the Super.  
**Why it was open:** J-Oppenheimer could have agreed, refused, or avoided the topic entirely.

| | action | would look like | gives up |
|---|---|---|---|
|  | `CONCEDE` | Agree to urge continued research on the Super. | His belief that it's not the right use of resources. |
| **<-- taken** | `HOLD` | Refuse to urge continued research on the Super. | Teller's goodwill and potential collaboration. |
|  | `DROP` | Change the subject without addressing the request. | Addressing Teller's direct request. |

**What they actually wrote:** I neither can nor will, Edward.

VERDICT: 
NOTE: 

---

## 28. `oppenheimer:26_hiroshima:J-Oppenheimer:10`  (test, None, 26. Hiroshima, August 1945)

```
[0] Katherine-Oppenheimer: Sorry. Yes, Charlotte, go ahead.
[1] Charlotte-Serber: Well, I don't know, he just said to tell you to bring in the sheets. Kitty? Kitty?
[2] J-Oppenheimer: They musn't drop it through cloud cover- If they detonate it too high in the air, the blast won't be as powerful-
[3] Officer: With respect, Dr.Oppenheimer. We'll take it from here.
[4] J-Oppenheimer: Did Truman brief Stalin at Potsdam?
[5] Groves: 'Brief' would be an overstatement. He referred to a powerful new weapon, Stalin said he hoped we'd make good use of it against Japan.
[6] J-Oppenheimer: That's it?
[7] Groves: Robert, we've given them an ace. It's for them to play the hand.
[8] J-Oppenheimer: You're aiming for the 6th?
[9] Groves: That's up to the C.O. in the Pacific.
```

**Person:** J-Oppenheimer  
**Situation:** Groves has just stated that the bombing date is up to the C.O. in the Pacific.  
**Why it was open:** He could have continued to press Groves or dropped the topic, but instead offered to go to Washington.

| | action | would look like | gives up |
|---|---|---|---|
|  | `HOLD` | Press Groves further on the timing or his influence. | Accepting Groves's statement as final. |
|  | `DROP` | Accept the answer and move on to another topic. | Further inquiry into the bombing schedule. |
| **<-- taken** | `TRADE` | Ask to go to Washington, implying a desire for continued involvement. | Directly challenging Groves's authority on the matter. |

**What they actually wrote:** Should I come with you to Washington?

VERDICT: 
NOTE: 

---

## 29. `bloomfield:#ai-vision:Wolf:358`  (dev, 2024-03-22, purple_threshold)

```
[346] Colbourn: Shoots, Clusters, and Early Clusters all have normal distributions on this scan, but buds and early shoots both have that strange very similar double bump distribution which makes me think something is going wrong.
[347] Wolf: look at brar
[348] Wolf: the ones from friday
[349] Wolf: that is the appropriate time to see early shoots
[350] Wolf: and i feel like its even too late there
[351] Colbourn: Interesting, ok that at least confirms that all the display logic is correct and it's a data issue bc looks pretty good on Brar. I'm going to go ahead with the prod push then in a few min. This makes me think we can likely programmatically detect bad distribution data in the future and flag it somehow so the user knows it might be out of the growing phase where it's applicable (or maybe even help us auto determine which models to run on a scan).
[352] Colbourn: Oh jeez, look at this triple bump on brar early cluster: <https://dev.bloomfield.ai/?customerID=18a3e659-7985-4c3a-80b4-f13810fc7231&amp;ranchID=50bcad4f-d071-41fc-a231-4ae4f6fd14c2&amp;year=2024&amp;blockID=afe4fda5-6749-4882-aedf-0aeb8a8d9fb2&amp;scanID=eacf74ea-f7ab-4b5b-bd36-d30ff614395e&amp;feature=early-cluster&amp;mapDimension=Per+Foot&amp;chartDimension=Distribution&amp;ranchPageView=blockDetail&amp;measure=DetectionFact.avgCountPerMeter&amp;pointID=df0d8aee-d5f3-30a3-90b2-f3d17901d196>
[353] Wolf: I think its row-side driven
[354] Wolf: could be a lot of things
[355] Wolf: would be awesome to be able to filter by each peak range
[356] Colbourn: So like pick an upper/lower threshold that we filter and show the rows that have data between that range?
[357] Wolf: the points that are in that range
```

**Person:** Wolf  
**Situation:** Colbourn asked for clarification on a filtering idea Wolf proposed, implying it could be a feature request.  
**Why it was open:** Wolf could have let Colbourn assume it was a request, or dropped it, but chose to clarify its status.

| | action | would look like | gives up |
|---|---|---|---|
| **<-- taken** | `HOLD` | Confirm it's a feature request, elaborate on details. | Flexibility to keep it an informal idea. |
|  | `DROP` | Let the idea go, don't clarify its status. | Opportunity to shape future development. |
|  | `CONCEDE` | Agree it's a feature request, let Colbourn run with it. | Control over the idea's scope. |

**What they actually wrote:** *this is an idea not an explicit feature request*

VERDICT: 
NOTE: 

---

## 30. `oppenheimer:18_security_interview:J-Oppenheimer:38`  (dev, None, 18. The security interview, 1943)

```
[26] Groves: You said that to Pash?
[27] J-Oppenheimer: I was trying to put it in the context of... Russia's not Germany.
[28] Groves: Boris Pash is the son of a Russian Orthodox bishop. Born here, but in 1918 he went back to Russia to fight the Bolsheviks. This is a man who's killed Communists with his own hands.
[29] Pash: I'm not the judge of who should or should not get information. My business is to stop it going through illegally. Could you be a little more specific?
[30] J-Oppenheimer: There's a man whose name was mentioned to me a couple of times- Eltenton. I think he's a chemist employed by Shell. He talked to a friend of his who's an acquaintance of someone on the project. To go beyond that would be to put names down of people who are not only innocent but were 100 percent cooperative.
[31] Groves: You thought Pash would be satisfied with that?
[32] J-Oppenheimer: I was trying to give them Eltenton without opening a can of worms. I told him a cock-and-bull story.
[33] Pash: These other people you mentioned, were they contacted by Eltenton direct?
[34] J-Oppenheimer: No.
[35] Pash: Well now, could we know through whom that contact was made?
[36] J-Oppenheimer: It would involve people who ought not be involved in this.
[37] Pash: Is this person a member of the project?
```

**Person:** J-Oppenheimer  
**Situation:** Pash has asked if the person involved is a member of the project.  
**Why it was open:** Oppenheimer could have given a simple yes/no, provided a nuanced answer, or refused to answer.

| | action | would look like | gives up |
|---|---|---|---|
|  | `CONCEDE` | Directly answer yes or no. | Providing additional context about the person's affiliation. |
| **<-- taken** | `HOLD` | Provide a nuanced answer about their affiliation (faculty, but not project). | A simple, direct answer to Pash's question. |
|  | `DROP` | Refuse to answer, citing privacy concerns. | Cooperating with Pash's line of questioning. |

**What they actually wrote:** A member of the faculty, but not on the project.

VERDICT: 
NOTE: 

---

## 31. `oppenheimer:19_tatlock_last:J-Oppenheimer:11`  (dev, None, 19. Last night with Jean, 1943)

```
[0] Tatlock: You left. Not a word. What did you think that would do to me?
[1] J-Oppenheimer: I wrote.
[2] Tatlock: Pages of nothing. Where'd you go?
[3] J-Oppenheimer: I can't tell you.
[4] Tatlock: Why not?
[5] J-Oppenheimer: Because you're a Communist.
[6] Tatlock: You drop in and out of my life and don't have to tell me why. That's power.
[7] J-Oppenheimer: Not that I enjoy. I'd rather be here for you as you need.
[8] Tatlock: But now you've got other priorities.
[9] J-Oppenheimer: I have a wife and child.
[10] Tatlock: That's not what either of us is talking about.
```

**Person:** J-Oppenheimer  
**Situation:** Tatlock dismisses J-Oppenheimer's mention of his wife and child as not what they're talking about.  
**Why it was open:** J-Oppenheimer could have defended his point about his family, or conceded, but chose to move on.

| | action | would look like | gives up |
|---|---|---|---|
|  | `HOLD` | Reiterate that his family are his priorities. | Addressing her deeper point about their relationship. |
|  | `CONCEDE` | Agree that his family isn't the core issue. | Using his family as a reason for his distance. |
| **<-- taken** | `DROP` | Let go of the 'wife and child' point and shift focus. | Defending his previous statement. |

**What they actually wrote:** Jean, you asked me to come. And I'm glad I did. But I can't come again.

VERDICT: 
NOTE: 

---

## 32. `boeing:customer-airlines:Technical-Pilot-01:48`  (test, 2017-06-05, Instant messages from Boeing employee to (former) 737 Chief Technical Pilot, June 5, 2017, BATES Number TBC-T&I 549015 –)

```
[36] Technical-Pilot-01: Copy me in on emails if you dont mind, so that i can keep up to speed with what is going on at home, in particular RTL and wind additive Flight was good, but weird business seat layout on [REDACTED]
[37] Mark-Forkner: do you know if MAX sim in MIA has the overrun and speedbrake warnings activated, or capable of being activated?
[38] Technical-Pilot-01: Not bad, but i would probably choose another airline over their 787 I don't know. But I will fire of an email right now to find out
[39] Mark-Forkner: I already sent one to [REDACTED]
[40] Technical-Pilot-01: Good
[41] Mark-Forkner: Now friggin Lion Air might need a sim to fly the MAX, and maybe because of their own stupidity. I'm scrambling trying to figure out how to unscrew this now! idiots
[42] Technical-Pilot-01: WHAT THE F%$&!!!! But their sister airline is already flying it!
[43] Mark-Forkner: I know I've asked for a webex so we can thru this with the DGCA not sure if this is Lion's fault or DGCA yet
[44] Technical-Pilot-01: Let me know if you need me to go down for a day while im there, not ideal but if we have to we have to
[45] Mark-Forkner: one of the DGCA guys is coming for the delivery so we can always get him there but supposedly they're making a training determination on Wed, that's why I'm trying to jump on this tonight with them
[46] Technical-Pilot-01: You definitely want to be in front of that one! Unbelievable, when will these curve balls stop coming...
[47] Mark-Forkner: its unreal man if we can make it thru summer we'll be ok, in theory
```

**Person:** Technical-Pilot-01  
**Situation:** Mark-Forkner expresses hope that if they make it through summer, they'll be okay.  
**Why it was open:** Technical-Pilot-01 could have agreed, agreed humorously, or changed the subject.

| | action | would look like | gives up |
|---|---|---|---|
| **<-- taken** | `HOLD` | Agree with the sentiment, perhaps with a humorous twist. | Offering a more serious or strategic response. |
|  | `CONCEDE` | Agree without adding a humorous comment. | Adding his own perspective or humor. |
|  | `DROP` | Change the subject to something else entirely. | Engaging with Mark-Forkner's long-term outlook. |

**What they actually wrote:** haha, I do recall saying and hearing the same thing at the end of last summer!!

VERDICT: 
NOTE: 

---

## 33. `bloomfield:#ai-vision:Lyubovsky:268`  (dev, 2024-03-11, purple_threshold)

```
[256] Lyubovsky: <@U044CULDG80> <@U024D0WMJAJ>, correct me if I'm wrong, but do we need to update ELT to filter on different flower classes?
[257] Rovani: yes
[258] McLafferty: If there are new names like blueberry_v6fit, please please add them to this spreadsheet for the front end team, so that we don't run into issues like we did this weekend.
[259] Chi: Thanks <@U053NF4DQV7>! Perhaps we could also add version/class/service names that are in development and just make a note that they're not live yet? That would help frontend development as well
[260] Lyubovsky: <@U06DH8RDJ2K> could you elaborate on that?
[261] Rovani: Maybe we could already add blueberry_v7?
[262] Lyubovsky: I'de want to resolve issues we'd been seeing with v6fit before starting to use v7, though it might make sense to add it
[263] Lyubovsky: I'de added it, assuming it will use the same classes as v6
[264] Rovani: I will add it to ELT
[265] Chi: Thanks Jean! <@U053NF4DQV7> Assuming this spreadsheet is just for documentation, if we could track planned names it would help with seeing what's coming for the dashboard, not having to use placeholders, that kind of thing
[266] Wolf: <@U053NF4DQV7> / <@U044CULDG80> we have a meeting later today with Agrovision--I don't know that it will come up, but can someone give me 2-4 bullet points on progress we've made in blueberry accuracy over the past 2 months and what are the next things we plan to do to improve color accuracy?
[267] Wolf: And we need more guidance from them on mapping true color to color stage for each variety and more feedback as well?
```

**Person:** Lyubovsky  
**Situation:** Wolf asks for guidance on mapping true color to color stage and more feedback from Agrovision.  
**Why it was open:** Lyubovsky could have agreed to the request, or asked for clarification, or simply acknowledged it.

| | action | would look like | gives up |
|---|---|---|---|
| **<-- taken** | `HOLD` | Ask for clarification on the request. | Immediately providing the requested information or committing to it. |
|  | `CONCEDE` | Agree to provide the guidance and feedback. | Understanding the specifics of the request first. |
|  | `DROP` | Acknowledge but not commit to the request. | Addressing Wolf's need for information for the meeting. |

**What they actually wrote:** What do you mean by this? > more guidance from them on mapping true color to color stage for each variety

VERDICT: 
NOTE: 

---

## 34. `bloomfield:#south_america-faragrotech:Letelier:134`  (test, 2024-06-18, purple_threshold)

```
[122] Colbourn: It's possibly bc they are "Viewer" which is only a theoretical role at the moment. Admin is the standard for everyone right now except Bloomfield which is Root. Admin essentially means Viewer right now so they can't screw anything up. Viewer I programmed as an option to prepare for us to start actually using roles soon. I'm going to swap them all to admin right now and we can see if that solves it.
[123] Colbourn: This email address isn't verified yet so that's probably the issue
[124] Mueller-Sim: Ok, Ricardo will reach out to them with those instructions. Thanks!
[125] Rodriguez: Sry my bad. Here goes the excel file. In the "Dashboard" spreadsheet are all of the calculation.
[126] Prendiville: The image is showing Agroberries. Is that your view or the customers? Which user?
[127] kaufmann: Yes, it is not appearing on the network (yet)
[128] Letelier: Yes, they can see now
[129] Letelier: Thanks all of you guys
[130] Letelier: <@U04M1Q0KGDT> this is a great question, I think they just were trying to set up the diaspora, but I can ask them tomorrow, because they are off work now.
[131] Wolf: can we get their comparison data
[132] Letelier: Hi all, people from Bomarea uploaded their information and they are asking me, just to understand, how long could take from In Progress to Completed. They are using the diaspora 024. Thanks
[133] Prendiville: <@U073RDABE6M> <@U0677JT1HSL> Could you find out from Hortifrut if they intend to scan recently pruned plants? These L blocks were scanned with intense coverage, but are freshly pruned and we have noticed false detections, particularly around the white cuts/pruning wounds. If this was intentional, our AI team is curious about the use case or what Hortifrut hoped to learn from the scans. From our side, the blueberry models have not been trained using dormant or recently pruned imagery. These images will be added to training and this should improve with the next version of the service, if they s ...
```

**Person:** Letelier  
**Situation:** Prendiville asks about scanning pruned plants, curious about the use case.  
**Why it was open:** Letelier could have deflected or escalated, but chose to directly answer the 'why'.

| | action | would look like | gives up |
|---|---|---|---|
| **<-- taken** | `HOLD` | Explain why they scanned pruned plants. | Avoiding potential scrutiny of the decision. |
|  | `ESCALATE` | Suggest a meeting to discuss the scanning strategy. | Providing an immediate answer. |
|  | `DROP` | Acknowledge the question but don't provide a reason. | Addressing Prendiville's curiosity. |

**What they actually wrote:** <@U04M1Q0KGDT> yes, they scan this blocks because Tim and I suggested them to scan this kind of plants to try to develop the model for counting of plants

VERDICT: 
NOTE: 

---

## 35. `bloomfield:#ai-vision:McLafferty:238`  (dev, 2024-02-27, purple_threshold)

```
[226] Wolf: can you send a calendar invite to me and Ryan?
[227] Lyubovsky: Sent for 3:00
[228] Prendiville: <@U022UFDHLHG> <@U058N0UUBU7> I had a question for <@U03JL7WMUCQ> today and he suggested I ask you. Does the AI team want to know and or record individual instances of poor/incorrect/questionable model performance? I think <@U0190EQ43E1> has a process for this as part of the QA procedure, but am uncertain if there is a way for other people to do it in the course of looking at scans for customers. An example is in the image below: <https://dashboard.bloomfield.ai/?customerID=c3c0cddb-10d8-4545-92a6-9910195f961d&amp;ranchID=7ad34a10-b692-4ba6-97b0-375b6a0a59ff&amp;year=2023&amp;blockID=6dc0d491- ...
[229] McLafferty: I have recently started a confluence page. There is a general page, and child pages for grape and blueberry. It would be awesome if you could add things here. And agree a way to do this on the dashboard is something we want to do. We've talked about flags and notes... <https://bloomfield.atlassian.net/wiki/spaces/AD/pages/12927041538/Scan+QA+2024+-+General>
[230] Colbourn: Ya totally agree, map markers/notes is the eventual way to do this. The short term method is just bookmarking the link for personal use or making a google sheet of links for sharing with ai team. Here's the jira ticket where we started thinking through this map marker flow: <https://bloomfield.atlassian.net/jira/software/c/projects/BLF/boards/68?search=markers&amp;selectedIssue=BLF-1454>
[231] McLafferty: I feel like it is going to be tough for us to not count these as 2, this is such a clear view of the stem! But absolutely we should re-annotate on things like this that you see.
[232] Berger: thank you Ryan for this note, I would highly appreciate if we record this feedback even for individual images, because they might give us the food for thought on further improvements. We definitely won't retrain with response to each image, but it's good to know where some potential problems might reside and we can further estimate how often they are, how critical, etc. With regards to this particular cluster, I'm pretty sure we have such examples labelled as two. And I'm even unsure how we can address this - since indeed it's difficult to tell why it's 1 and not 2 and not to confuse the label ...
[233] Prendiville: &gt; And I'm even unsure how we can address this - since indeed it's difficult to tell why it's 1 and not 2 and not to confuse the labellers in all other cases. (I have some thoughts on this, for reference, but may not be important at this time or any cause for action.) What's the Bloomfield definition of what a grape "cluster" is? It's an interesting question, that I thought was more straightforward and I've looked through some things to confirm my own understanding. The multiclass labeling guidelines kind of skirt around a clear definition. Using "a cluster of flowers" or "a cluster of berri ...
[234] Ortega: I like the idea of defining a cluster by it's shoulder, I could see it being a useful feature for table grapes if we would estimate width of the shoulder too. I do wonder if narrowing the definition would cause confusion around secondary clusters, etc. As you mentioned, I think the challenge is the shoulder is not always visible. But I think this is something interesting to keep in mind if / when we refine cluster strategy in the future.
[235] Berger: I think we should at least record the above observations somewhere (so they are not lost in the comments) and maybe revisit them before we start annotation of clusters this season? Would it make sense to include this pedancle into the cluster label then? Maybe though we shouldn't postpone this discussion too much since we are already labelling the datasets with early clusters, so it might be a chance to make a right move there. What do you think, is it worth doing? <@U058N0UUBU7> <@U0190EQ43E1> I believe we were mostly labelling and separating the clusters based on its internal stem (don't kno ...
[236] Prendiville: The internal stem is the rachis. It's the same axis as the peduncle but is surrounded by pedicles holding fruit and flowers. Whereas the peduncle is the bare part of that axis that attaches to the shoot. Probably too fine detail for labeling, but I don't know best practices. Starts to get into real botany, technical jargon. Whereas things like shoulder, wing, and to some degree cluster are more general (but also vaguer).
[237] McLafferty: I think it might make sense to include the pedancle in the label when it is seen, as this is how we are defining that this is one cluster and not two.. We should make sure the guidelines are clear for Flipside on shoulders. As we know the shoulders are sometimes labeled separately. But like we have said we are not going to be perfect on this of course because of what we can and cannot see in the images. We can't always tell if there is this connection.
```

**Person:** McLafferty  
**Situation:** The team is discussing how to define and label grape clusters, with various suggestions and complexities.  
**Why it was open:** McLafferty could have pushed for a change or dropped the discussion, but chose to maintain the status quo.

| | action | would look like | gives up |
|---|---|---|---|
| **<-- taken** | `HOLD` | Reiterate the need for clear guidelines on shoulders. | Resolution on the peduncle inclusion. |
|  | `CONCEDE` | Agree to include the peduncle in the cluster label. | Maintaining current labeling guidelines. |
|  | `DROP` | Move on without a definitive decision on guidelines. | Clarity for future labeling. |

**What they actually wrote:** Sounds like we will leave the guidelines as is. We've talked a lot about shoulders in the past. And I think the idea for a solution is 'cluster filtering' possibly?

VERDICT: 
NOTE: 

---

## 36. `oppenheimer:16_la_science:J-Oppenheimer:53`  (dev, None, 16. The work, 1943)

```
[41] J-Oppenheimer: Put Mrs.Hornig on the plutonium team.
[42] Groves: What the hell were you doing in Chicago?!
[43] Condon: Visiting the Met Lab-
[44] Groves: Why?!
[45] Condon: You can't talk to us like this. We have every right-
[46] Groves: You have just the rights I give you! No more, no less.
[47] Condon: This is ridiculous- we're adults, trying to run a project here. Tell him, Robert.
[48] J-Oppenheimer: Compartmentalization is the protocol we agreed to.
[49] Condon: You've got to be kidding me. Enough of this madhouse- nobody can work under these conditions. You know what, Generalissimo? I quit. Thanks for nothing.
[50] Groves: Better off without him.
[51] J-Oppenheimer: Aren't you more worried about his discretion out there?
[52] Groves: We'll have him killed. Kidding. He hates me, not America.
```

**Person:** J-Oppenheimer  
**Situation:** Groves jokes about killing Condon and dismisses Oppenheimer's concern about discretion.  
**Why it was open:** Oppenheimer could have dropped the issue, hinted at Groves' motives, or directly confronted him.

| | action | would look like | gives up |
|---|---|---|---|
| **<-- taken** | `HOLD` | Hint at Groves' manipulative hiring practices. | Keeping his suspicions about Groves to himself. |
|  | `DROP` | Let the topic of Condon's discretion go. | Pressing Groves on his methods. |
|  | `ESCALATE` | Directly accuse Groves of manipulation. | Subtlety in his challenge to Groves. |

**What they actually wrote:** Not everyone has levers like mine to pull.

VERDICT: 
NOTE: 

---

## 37. `bloomfield:#ai-vision:Colbourn:678`  (test, 2024-06-28, purple_threshold)

```
[666] Prendiville: <@U0190EQ43E1> there wasn't much of an indication that nursery customers used the Dashboard (or were expected to) last year. Novavine has expressed an interest in using our tools (mainly the point to point measurement tool) on Dashboard to do their own data extraction. But when the rubber hits the road I'm not sure that will be the case. Double A...I don't know, mainly I'm hoping the points end up in the right blocks. <@U02CBGVPZ2L> The aa_gng_by_sku file is what we delivered last year. The sheet you shared is something we made while trying to align our results, but I wouldn't want to give peo ...
[667] McLafferty: Thank you! This is very helpful, so the heights request was just for Novavine? I think it was Novavine last year that wanted vine grading? The heights will be the mean per point, not individual detections.
[668] McLafferty: This was the vine grading deck we gave to Novavine last year. I am wondering if the mean per point heights are going to be useful for them at all, if this is what they are looking for? "There are a couple of factors. First if the overall length was over 3" to 4" I considered it a number 1. If it was 2" or less it was a number 2. Also internode growth. IF the plant had stretched internodes I would say it was a 1. If they were compressed I call it a 2" <https://docs.google.com/presentation/d/14xPuI-Z3MM_yzRGe2NDTiVdUi5-e0Rq5zd2MCIGowXQ/edit?usp=sharing>
[669] Colbourn: Ya I definitely wonder how useful the heights will be too bc right now seems like we can only do these options: 1. Give mean per point height (probably a little useful on the heatmap to get overall trends). 2. Aggregating the mean of a set of points for each variety which will give inaccurate double counting issues or aggregate of aggregate issues (again maybe somewhat useful still for overall trends). 3. Potentially give a list of the individual heights for each detection in a point so they can at least click on image and see the heights of each vine in a specific image. 4. Potentially try an ...
[670] Prendiville: I have a hard time imagining that mean height per point will be a useful metric for them. Is the mean based only on the growth detection or also the no growth? My understanding is that if there were 500 growth detections, and the cutoff for height was 2" they would want to know there were for example 200 2"+ and 300 sub-2" detections.
[671] Colbourn: Ya kinda seems like until we can deduplicate vines that height won't be too helpful for them. And currently the height per point would show either growth or no_growth depending which you have selected in the dropdown so we'd probably need to combine those two somehow.
[672] Prendiville: "no_growth" should be excluded from any average height calculation. They've by definition already been defined as heightless.
[673] Colbourn: Is it possible to use similar logic to how we do estimated counting to do estimated average heights that mathematically handle the overlapping image detections?
[674] Wolf: I'm not concerned about de-duplicating vines, if we assume a constant duplication rate. We're looking for really three numbers. -Total growth estimates count -Total no growth estimated count - -Percent of total raw growth count above/below a certain height threshold That last one we can use raw count heights and draw a line through the distribution whenever the height threshold is
[675] Rovani: Here is a summary of grape_late_season_v2 improvements: - new model was trained and tested to fix cluster mask and overlapping issue - running the test last week showed a significant performance decrease (f1-score went from .78 to .69) - the reason is most likely a side-effect of switching from bitmap to vector datasets - we are investigating a solution to fix this
[676] McLafferty: Will we call the new version v3 when it's deployed?
[677] Chi: <@U022UFDHLHG> About the grapes point color - the dashboard does increase the saturation by default, but it can be turned off inside of the color distribution chart. There is a slight flow bug where the chart doesn't automatically switch to color distribution if you directly navigate to a scan from a link, but hopefully this matches a bit better! The opacity is also slightly lowered for the selected row in the point map vs. the color indicator on the photo.
```

**Person:** Colbourn  
**Situation:** Chi describes a slight flow bug where the color distribution chart doesn't automatically switch.  
**Why it was open:** Colbourn could have acknowledged the bug without action, or explicitly requested a ticket.

| | action | would look like | gives up |
|---|---|---|---|
| **<-- taken** | `HOLD` | Ask Chi to add the flow bug to a ticket for eventual fixing. | Assuming the bug will be addressed without explicit action. |
|  | `CONCEDE` | Acknowledge the bug and trust it will be handled. | Ensuring the bug is formally tracked. |
|  | `IGNORE` | Say nothing and let the conversation move on. | Ensuring the bug is formally tracked. |

**What they actually wrote:** <@U06DH8RDJ2K> Can you add that flow bug to a ticket so we can eventually fix it?

VERDICT: 
NOTE: 

---

## 38. `bloomfield:#south_america-faragrotech:Letelier:502`  (dev, 2024-11-15, blueberry_size)

```
[490] Colbourn: No problem, let me know if you need any help figuring it out!
[491] Letelier: <@U02CBGVPZ2L> thanks!!! For sure I am going to need help to understand this.
[492] Colbourn: I saw Adrian gave you a tutorial earlier but feel free to reach out to either of us if you end up needing more help with it!
[493] Letelier: <@U0459CR7Y59> <@U04PB81LQ3D> hi team. The people from Purafruit updated the camera 072 last Oct 25th but they still have this message
[494] Theofiledes: on boot up , the camera wants to get GPS time, so once the GPS is connected and it sees the sky it should clear the red popup . the Camera Icon will turn green let me know if this resolved
[495] Wolf: <@U0459CR7Y59> any insight here?
[496] Kelley: Could you also have them check that the cable is installed correctly and the wire isn't damaged.
[497] Theofiledes: I miss read the message it was "old" if they want to try it now they may need to update that camera . I will check status . one sec
[498] Theofiledes: this camera is in need of the SSL update so before we trouble shot this can they connect the camera to their smasher setup for an update.
[499] Letelier: <@U0459CR7Y59> thanks, If I understand well the updated of the camera was not correct, so we should update again and then try again with the GPS, right?
[500] Theofiledes: Yes please , because now without updating they wouldn't be able to connect the tablet to the camera. So if we start there with an update then try again , if the error stays then we can troubleshoot why.
[501] Theofiledes: If you give me a heads up when they connect it to the smasher I can watch it from my end and maybe try to grab log data for our team :relaxed:
```

**Person:** Letelier  
**Situation:** Theofiledes explained the camera needs an SSL update and offered to watch from their end.  
**Why it was open:** Letelier could agree to the update, re-emphasize the smasher issue, or simply acknowledge without committing.

| | action | would look like | gives up |
|---|---|---|---|
| **<-- taken** | `HOLD` | Acknowledge and state intent to proceed with the update. | Pushing for alternative solutions. |
|  | `RE_RAISE` | Re-raise the issue of not having a smasher. | Moving forward with the suggested solution. |
|  | `DROP` | Acknowledge without committing to the update. | Clarity on the next steps for the camera. |

**What they actually wrote:** <@U0459CR7Y59> thanks. In this case makes sense to me this kind of issue, because they do not have smasher and they trie twice to upload the camera with the tablet. I will ask them to do it again and I will let you know

VERDICT: 
NOTE: 

---

## 39. `boeing:flight-controls:FlightTest-Engineer-01:16`  (dev, 2014-05-01, 737 MAX Flight Controls/Pilots Meeting,)

```
[4] FlightControls-Engineer-02: Should it be annunciated with the existing SPEED TRIM FAIL light ◦ The speed trim system is not all that reliable - [REDACTED] MCAS does not operate within the 1.3g flight envelope ◦ So the probability of being in the flight regime and having a failure is [REDACTED]
[5] FlightControls-Engineer-02: Currently, we are stuck with the overhead light to annunciate the failures, as the flight control computers are not connected to the maintenance status message system
[6] FlightControls-Engineer-02: The failures that would cause MCAS to fail are almost all the same as the ones that would cause Speed Trim to fail. At the moment, autoflight plans to use the same light for failures of the additional signal that MCAS uses
[7] FlightControls-Engineer-02: The condition statement could be changed to note that MCAS is failed, but it might not even be necessary to let crews know that MCAS is on the airplane. It is just one of those automatic protection functions.
[8] FlightControls-Engineer-02: Since speed trim runs the trim wheel anyway, crews likely wouldn't distinguish if the wheel was moving for speed trim or MCAS.
[9] FlightControls-Engineer-01: As long as you need. Just let me know how much time you need so I can schedule it with them. Do you think you'll be ready to give this Friday the 16th?
[10] FlightControls-Engineer-01: I've recruited a few others from our group to help out. Is it alright if I let you know this Friday, after we have a better idea of how quickly it's coming together?
[11] FlightControls-Engineer-01: You bet, sounds good. Thanks for all the help on this.
[12] FlightControls-Engineer-01: [REDACTED]
[13] FlightControls-Engineer-01: [REDACTED], Today we collectively decided it would be better strategically to present the AEG with the systems commonalities and differences at a bit of a higher level before we deep dive into the handling qualities issue.
[14] FlightControls-Engineer-01: So you have a bit more time to tweak your presentation. I’d like to shoot to have it available for Thursday May 22nd. Thanks for all the hard work on this, we’re looking forward to reviewing what you’ve come up with. Have a good wkd all. [REDACTED]
[15] FlightControls-Engineer-01: [REDACTED], Next Thursday will be fine. We’ll plan on reviewing our presentation with you later this week. I received the approach attitude information you sent – we’ll be sure to include that. Thanks, [REDACTED]
```

**Person:** FlightTest-Engineer-01  
**Situation:** FlightControls-Engineer-01 has confirmed next Thursday for a presentation review.  
**Why it was open:** They could propose a time, agree without a time, or suggest a more formal scheduling approach.

| | action | would look like | gives up |
|---|---|---|---|
| **<-- taken** | `HOLD` | Propose a specific time for the review and check availability. | Letting FlightControls-Engineer-01 set the time. |
|  | `CONCEDE` | Agree to next Thursday without suggesting a specific time. | Taking initiative in scheduling. |
|  | `ESCALATE` | Suggest a broader meeting to coordinate all schedules. | Keeping the scheduling informal. |

**What they actually wrote:** Thanks [REDACTED] How does 1pm a week from Thurs look for your team? [REDACTED] Does that look ok for you as well?

VERDICT: 
NOTE: 

---

## 40. `bloomfield:#labeling:Rovani:32`  (dev, 2024-01-24, purple_threshold)

```
[20] Prendiville: Yeah, I'm talking to Brad soon. If I have the opportunity I might asked about this block and if they observed mildew. I'm interested to know if he tracks field observations on any platforms when he finds them.
[21] McLafferty: We have had west coast customers want sunburn damage assessments in the past, and we have done some labeling for this. Might be something to consider when we are able!
[22] Wolf: we have sunburn labelling??
[23] McLafferty: We do! I will look at how much, it might not be very much. I just remember we tried to start working on this for A to Z looks like in 2021, but not sure how far we got. This is an example of the attributes we used for sunburn-damage; mild, medium and severe.
[24] McLafferty: <https://segments.ai/Bloomfield.ai/ATZ-0120503M-20210721/>
[25] Lyubovsky: I was seeing a case where it looked like flowers were going directly from petalfal to being Purple (1-Green, 2-Purple) blueberries, and wanted to check if my understanding/labeling was correct: cc: <@U03JL7WMUCQ> <@U04M1Q0KGDT> <@UBD0F3UH0> <@U058N0UUBU7> <https://segments.ai/Bloomfield.ai/AUTO-HTF-ESP-2-G-G31-20230926-HIGH-RES/samples/a010935a-aa2f-4397-aa06-ac5bd0e3946c/ground-truth|Link> to image:
[26] Wolf: I'll let others weigh in here, but I wouldn't consider any of those to be 2-Purple.
[27] Lyubovsky: Also, confirming these four flowers would be all closed?
[28] McLafferty: I agree with Hayden I don't see any purple on those berries.
[29] Lyubovsky: I'm seeing a small purple discoloration on the edges of the berries, which would fit under my understanding of 2-Purple (ie. any purple). Is there a better way to define it so that this wouldn't fall under 2-Purple?
[30] McLafferty: Ooh I see what you are saying, I did not notice this at first.
[31] Rovani: I think we need the customer to give feedback on the fact that 1-Green 2-Purple is counted as purple
```

**Person:** Rovani  
**Situation:** Rovani has suggested getting customer feedback on the '1-Green 2-Purple' definition, and Wolf has stated that customer answers are inconsistent.  
**Why it was open:** Rovani could have accepted Wolf's point about inconsistent customer feedback or continued to advocate for seeking it.

| | action | would look like | gives up |
|---|---|---|---|
|  | `CONCEDE` | Agree with Wolf that customer feedback is difficult to standardize. | The idea of using customer feedback to resolve the issue. |
| **<-- taken** | `HOLD` | Reiterate the need for customer feedback despite inconsistencies. | Accepting Wolf's assessment of customer feedback challenges. |
|  | `DROP` | Acknowledge Wolf's point and move on to a different aspect of the problem. | Pushing for a resolution via customer input. |

**What they actually wrote:** I am not sure all of them are 2-Purple, especially this one:

VERDICT: 
NOTE: 

---
