# Baseline results: untuned player tracking

Run on: 2026-10-06
Detector: RF-DETR Medium, pretrained weights, "person" class only, threshold 0.5
Tracker: ByteTrack, default settings (frame rate set per clip)
Speed on this machine: about 13.6 to 17.7 frames per second, which is about 3.4 to 4.4 minutes per minute of 60 fps footage (3,600 frames).

How the speed was measured: the held-out run processed 23,666 frames in 28 min 55 s (13.6 frames/s, overlay videos written) and the training run about 50,200 frames in 47 min 23 s (17.7 frames/s, no overlays). The two runs went at the same time on the same machine and shared the GPU, so the true speed of a single run is higher than either figure. A separate 60-frame smoke run took 7.81 s including model load and overlay writing, which gives a floor of 7.7 frames/s. All three figures are well above the 1 frame/s stop line.

## Scores

| Set | Clips | IDF1 | Precision | Recall |
|---|---|---|---|---|
| Held out | 5 | 70.6% | 87.2% | 80.6% |
| Training | 11 scored (6 more could not be scored, see below) | 70.5% | 90.5% | 82.9% |

Spec bar for IDF1: 85%. Published reference: about 93%.

The held-out result is 14.4 points below the bar. One of the five held-out clips meets the bar (CAR_VS_BOS_002). The training figure covers only 11 of the 17 training clips.

## Per clip (held out)

| Clip | IDF1 | Precision | Recall | Ground-truth boxes | False boxes | Missed boxes | Meets bar |
|---|---|---|---|---|---|---|---|
| **Overall** | 70.6% | 87.2% | 80.6% | 202273 | 23950 | 39269 | no |
| CAR_VS_BOS_001 | 59.4% | 71.9% | 66.4% | 53436 | 13874 | 17933 | no |
| CAR_VS_BOS_002 | 86.0% | 97.5% | 90.8% | 22814 | 536 | 2103 | yes |
| CAR_VS_NYR_001 | 71.4% | 98.9% | 88.9% | 45854 | 434 | 5071 | no |
| CAR_VS_NYR_002 | 81.4% | 97.4% | 91.1% | 37919 | 939 | 3360 | no |
| CAR_VS_NYR_003 | 65.7% | 79.4% | 74.4% | 42250 | 8167 | 10802 | no |

## Clips that failed

Held-out run: none. All five clips ran and scored.

Training run: all 17 clips ran (`ok`), but six could not be scored. Each has the message `ground truth: a track id appears more than once in the same frame`:

- CHI_VS_TOR_002
- CNTRL_VS_PAC_001
- CNTRL_VS_PAC_005
- CNTRL_VS_PAC_006
- STL_VS_SJ_001
- STL_VS_SJ_002

This is not a pipeline failure. In these six clips the dataset's own ground truth puts the same track id on two different boxes in the same frame (documented in `docs/data-verification.md`, VIP-HTD section: for example frame 1279, id 27 in `CNTRL_VS_PAC_001`, one box at x=1242 and one at x=96, far apart). The scorer rejects that by design, and the loader has not been changed because the policy is the owner's to choose (question 3 below). None of the six is a held-out clip. The training IDF1 above therefore excludes them.

## What the overlays show

Method and limits: I cannot play video, so every observation below comes from sampled still frames, not from watching the clips. Frame indices are 0-based. The stills are in `data/tmp/overlay-check/` (git-ignored). Anything not listed as seen was not looked at.

- Smoke clip (SMOKE_001, first 60 frames of CGY_VS_DAL_001): I looked at frames 0, 20, 21 and 59 (frames 1, 2, 22 and 40 were also written but not opened). Frame 0 has no boxes at all (the tracker has not yet started a track). In frames 20 and 21 boxes sit on the skaters and goalie, and each keeps its colour and number from one frame to the next, for example the player in the white jersey near the bottom centre keeps the same black box. In frame 59 the player with the same colour box is still the same skater, and the number 2 label is still on the same skater who was down on the ice in frames 20 and 21. A few overlapping players in a cluster near the left of the net are boxed on top of each other. By eye the boxes sit on players and keep their numbers over the frames seen.
- Best clip (CAR_VS_BOS_002, IDF1 86.0%): sampled stills at frames 793, 1323, 1325, 1850 and 2379 (the clip has 2,644 frames; the full set of 13 sampled frames is listed in the task report). Boxes sit on skaters in all five stills, and on a referee in frames 1850 (number 45), 1323 (number 33) and 2379 (number 61). Number labels stayed with the same skater across frames 1323 and 1325 (for example numbers 21, 27, 34 and 15 stay on the same skaters). Two players in a boards battle (frame 793, near the bottom) and a pair against the boards (frames 1323 and 1325, number 32) are boxed as a single box, so the detector merges players that overlap. In frame 793 two referees are visible, one at the top and one at the bottom right, and neither has a box. At the same frame the three skaters in the open ice are boxed. In frame 1323 the Carolina bench is visible along the boards and none of the bench players has a box. I did not draw ground truth on this clip, so I cannot say whether the unboxed referees are boxed in the ground truth.
- Worst clip (CAR_VS_BOS_001, IDF1 59.4%): sampled stills at frames 651, 2606, 3258 and 5213, plus ground truth drawn against the tracker's boxes for frames 1303, 3258 and 5213 (green is ground truth, red is the tracker's box). The clip has 6,517 frames. What I saw: (a) In frames 1303, 3258 and 5213, for most players the tracker's box and the ground-truth box are on the same player but offset from each other, the ground-truth box being larger and shifted by roughly 10 to 30 pixels. The scorer needs an overlap of at least 0.5 to count a match, so offsets like these can turn a correct detection into one false box and one missed box. A numeric check supports this (see Reading of the numbers). I do not know whether the cause is the tracker's box, the annotation, or camera motion; the stills do not say. (b) In frame 651 a box (number 27) sits on a person in the stands at the bottom right, and in frame 1303 an unmatched box sits at the very bottom edge on someone at the boards; in frame 2606 boxes sit on players or staff along the boards at the top, near the bench, and at the right edge, and I cannot tell from the stills whether these people are on the ice or on the bench. I did not draw ground truth on frames 651 and 2606, so I do not know whether they count as false. (c) Referees are boxed (frames 651, 2606, 5213). (d) A player partly hidden behind another at the boards (frame 651, numbers 20 and 21) gets two overlapping boxes, and in the same frame the goalie in the net at the right edge has no box, and in frame 5213 a skater at the bottom edge is cut by the frame and boxed only partly. (e) In frame 3258 there are 8 ground-truth boxes and 7 tracker boxes; at the left edge there is a partly visible player (it looks like a goalie) whose ground-truth box is larger than the tracker's box.
  I did not look for numbers swapping when players cross. The only consecutive pair I opened was frames 1323 and 1325 of CAR_VS_BOS_002, where no number moved to a different skater; frames 3259 to 3261 of CAR_VS_BOS_001 were written but not opened.

## Reading of the numbers

- Is the gap to the bar mostly detection (low precision or recall) or mostly identity (precision and recall high but IDF1 low)? Both, and it differs by clip. Held out, precision is 87.2% and recall 80.6%. If every identity were right, IDF1 could not exceed roughly the harmonic mean of those two, about 83.8%, which is itself below 85%. So at these detection numbers the bar cannot be reached even with perfect identity; detection alone costs about 1 point against the bar and identity costs a further 13 points (83.8% to 70.6%). Per clip: CAR_VS_NYR_001 has precision 98.9% and recall 88.9% (about 93.6% if identity were perfect) but IDF1 71.4%, so its loss is mostly identity, about 22 points. CAR_VS_BOS_002 and CAR_VS_NYR_002 are also mostly identity (about 8 and 13 points). CAR_VS_BOS_001 (precision 71.9%, recall 66.4%, about 69.0% ceiling) and CAR_VS_NYR_003 (79.4%, 74.4%, about 76.8%) lose most of their points in detection, with 13,874 and 8,167 false boxes. A caveat for CAR_VS_BOS_001 (figures from `data/tmp/iou_check.py`, run on every 25th frame of each clip): on those frames, 72% of the tracker's boxes overlap a ground-truth box at 0.5 or more, 22% overlap one at between 0.1 and 0.5, and 6% overlap none at 0.1 or more; for CAR_VS_BOS_002 it is 98%, 1% and 1%. So much of the low precision and recall in BOS_001 (and 17% of CAR_VS_NYR_003's boxes, 1,595 boxes checked, from the same script) is boxes that are close to a ground-truth box but not close enough, rather than boxes on nothing.
- Are false boxes mostly referees, bench players or crowd? I cannot say from the data. Referees are boxed in the ground truth with the same class as players (`docs/data-verification.md`), so a tracker box on a referee is a correct match, not a false box, and I saw referees boxed by the tracker in frames of both clips. In the stills I did look at, I saw a box on a person in the stands (CAR_VS_BOS_001 frame 651) and unmatched boxes at the bottom edge and along the boards (frames 1303 and 2606), but I did not tally false boxes by type, and the stills are too few to say which kind dominates. The numeric check above points to misaligned boxes rather than crowd boxes as a large part of the false boxes in BOS_001. A proper answer needs a count of unmatched boxes drawn on every false box, which was out of scope here.
- Does the 30 fps versus 60 fps split matter? It cannot be answered on the held-out clips, because all five are 60 fps. On the 11 scored training clips (the training clips are tuning-side data, so this is only an observation), the six 30 fps clips average 76.9% IDF1 (range 60.2% to 93.1%) and the five 60 fps clips average 68.0% (range 60.0% to 71.5%). That points to 30 fps scoring higher, but the sample is small, the clips come from different games and cameras, and CHI_VS_TOR_003 alone (93.1%) lifts the 30 fps average; without it the 30 fps average is 73.7%. I would not conclude it matters until clips from the same game at both rates are compared. One consequence for the owner: the held-out set contains no 30 fps clips, so held-out scores say nothing about 30 fps footage.

## Questions for the owner (decision point 1)

1. Keep the IDF1 bar at 85%, or change it? Measured on the held-out set: 70.6% overall, a gap of 14.4 points; only one clip of five (CAR_VS_BOS_002, 86.0%) meets it, and one of the other four is above 80% (CAR_VS_NYR_002 at 81.4%). Recommendation: keep 85% for now. It is reached by one clip with an untuned detector and tracker, and the published reference is about 93%, so it looks reachable on clean clips; the open issue is the 0.5 overlap rule against the ground-truth boxes on clips like CAR_VS_BOS_001, which should be understood before the bar is judged too strict or too loose.
2. Should Plan 2 start with detector fine-tuning? The evidence points two ways. Detection limits CAR_VS_BOS_001 and CAR_VS_NYR_003 (precision and recall below 80%), and recall is about 89% to 91% even on the cleanest clips, which suggests a hockey-specific detector would help. But on three of five held-out clips (CAR_VS_BOS_002, CAR_VS_NYR_001, CAR_VS_NYR_002) precision is at or above 97% and identity loss (8 to 22 points) is the bigger cost, and fine-tuning the detector does not by itself fix that. Recommendation: do not start Plan 2 with fine-tuning. First spend a short step on the BOS_001 box alignment (is the offset in the annotation, the tracker, or the scorer's overlap rule) and on a tracker change aimed at identity (the tracker's settings, then a different tracker), measured on the training clips only; then decide on fine-tuning from what is left.
3. How should ground-truth frames where one track id has two boxes be treated? Six training clips (CHI_VS_TOR_002, CNTRL_VS_PAC_001, CNTRL_VS_PAC_005, CNTRL_VS_PAC_006, STL_VS_SJ_001, STL_VS_SJ_002) cannot be scored until this is decided. Options: drop the repeated boxes in those frames, keep one box per id per frame, or exclude the six clips. Recommendation: exclude those six clips from scoring for now and keep the loader strict, because choosing which of two far-apart boxes is the right one would be a guess; if more training clips are wanted later, drop only the repeated rows and record how many were dropped per clip.
