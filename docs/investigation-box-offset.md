# Investigation 1: the box offset on CAR_VS_BOS_001

Checked on: 2026-10-07. Runs: `data/runs/plan2a-heldout`, `data/runs/plan2a-train` (RF-DETR Medium "person" at 0.5, ByteTrack defaults; settings in each `run.json`). Tables: `data/runs/diagnose/{heldout,train}/summary.md`. Per-clip numbers come from `data/runs/diagnose/{heldout,train}/<clip>.json` unless a line says "computed from"; those lines come from small read-only helpers in `data/tmp/` (`offset_sim.py`, `shift_sim.py`, `drift.py`) that read the run tables and the ground truth and change nothing.

Held-out vs training: CAR_VS_BOS_001 is a held-out clip. It is described here because the owner asked. Every recommendation below rests on the training clips only.

## The question

Plan 1's baseline found that on CAR_VS_BOS_001, 22% of tracked boxes overlap a ground-truth box only at IoU 0.1 to 0.5. Is that the annotation (how ground-truth boxes are drawn or timed), the tracker (boxes lagging the players), or the 0.5 overlap rule (punishing small boxes that are only slightly off)?

## Summary table

Held-out (`data/runs/diagnose/heldout/summary.md`):

| Clip | fps | boxes | <0.1 | 0.1-0.3 | 0.3-0.5 | >=0.5 | med dx | med dy | med d_bottom | med w ratio | med h ratio | tracks==dets | IDF1@0.3 | @0.4 | @0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CAR_VS_BOS_001 | 60 | 49377 | 5.5% | 6.3% | 16.3% | 72.0% | -0.2 | 1.2 | 3.4 | 0.90 | 0.92 | 100% | 73.5% | 67.9% | 59.4% |
| CAR_VS_BOS_002 | 60 | 21247 | 1.6% | 0.1% | 0.8% | 97.5% | -0.1 | 1.4 | 3.4 | 0.87 | 0.92 | 100% | 86.7% | 86.6% | 86.0% |
| CAR_VS_NYR_001 | 60 | 41217 | 0.6% | 0.0% | 0.4% | 99.0% | 0.9 | 0.1 | 2.7 | 0.93 | 0.96 | 100% | 72.0% | 71.8% | 71.4% |
| CAR_VS_NYR_002 | 60 | 35498 | 2.3% | 0.0% | 0.4% | 97.4% | 0.8 | 0.2 | 3.1 | 0.93 | 0.94 | 100% | 81.9% | 81.7% | 81.4% |
| CAR_VS_NYR_003 | 60 | 39615 | 3.2% | 5.3% | 12.1% | 79.4% | 1.3 | 0.6 | 3.4 | 0.91 | 0.93 | 100% | 75.5% | 71.6% | 65.7% |

Training (`data/runs/diagnose/train/summary.md`):

| Clip | fps | boxes | <0.1 | 0.1-0.3 | 0.3-0.5 | >=0.5 | med dx | med dy | med d_bottom | med w ratio | med h ratio | tracks==dets | IDF1@0.3 | @0.4 | @0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CGY_VS_DAL_001 | 60 | 30101 | 2.4% | 0.2% | 0.9% | 96.6% | 0.9 | 0.1 | 2.6 | 0.93 | 0.96 | 100% | 69.0% | 68.6% | 67.9% |
| CGY_VS_DAL_002 | 60 | 32897 | 1.2% | 0.1% | 0.8% | 97.8% | 1.1 | -0.0 | 3.3 | 0.91 | 0.95 | 100% | 72.3% | 72.0% | 71.4% |
| CGY_VS_DAL_003 | 60 | 23902 | 1.0% | 1.5% | 8.5% | 89.0% | 1.7 | 0.4 | 3.3 | 0.92 | 0.94 | 100% | 75.0% | 73.1% | 69.0% |
| CHI_VS_TOR_003 | 30 | 7257 | 0.4% | 0.3% | 1.6% | 97.6% | 0.3 | 1.2 | 3.6 | 0.88 | 0.93 | 100% | 94.7% | 94.4% | 93.1% |
| CHI_VS_TOR_004 | 30 | 12622 | 1.5% | 2.0% | 9.4% | 87.1% | 1.0 | 1.3 | 2.9 | 0.90 | 0.94 | 100% | 66.3% | 64.0% | 60.2% |
| CNTRL_VS_PAC_002 | 30 | 11652 | 1.0% | 0.0% | 0.6% | 98.4% | 0.1 | 1.8 | -0.2 | 0.96 | 0.98 | 100% | 80.6% | 80.5% | 80.0% |
| PIT_VS_SJ_001 | 30 | 9146 | 0.4% | 3.6% | 13.4% | 82.5% | 0.8 | 1.8 | 2.6 | 0.90 | 0.93 | 100% | 82.8% | 78.3% | 71.2% |
| PIT_VS_SJ_002 | 30 | 19342 | 0.6% | 0.1% | 2.5% | 96.8% | 1.2 | 1.6 | 0.2 | 0.95 | 0.97 | 100% | 78.4% | 78.0% | 76.5% |
| PIT_VS_SJ_003 | 30 | 25731 | 0.9% | 0.2% | 1.8% | 97.0% | 1.1 | 0.8 | 1.8 | 0.93 | 0.96 | 100% | 82.0% | 81.6% | 80.5% |
| PIT_VS_WAS_001 | 60 | 22809 | 0.6% | 1.8% | 9.8% | 87.8% | 0.9 | 1.3 | 3.6 | 0.89 | 0.91 | 100% | 79.7% | 77.0% | 71.5% |
| STL_VS_SJ_003 | 60 | 43520 | 3.0% | 7.1% | 14.1% | 75.8% | 0.5 | 1.0 | 3.6 | 0.90 | 0.92 | 100% | 71.5% | 66.6% | 60.0% |

Not diagnosed (the tool scores through `score_tracking`, which rejects them): six training clips whose ground truth repeats a track id inside one frame, namely CHI_VS_TOR_002, CNTRL_VS_PAC_001, CNTRL_VS_PAC_005, CNTRL_VS_PAC_006, STL_VS_SJ_001 and STL_VS_SJ_002. They stay excluded from scoring by design.

Column meanings (from `src/tracker/analysis/box_offset.py`): offsets are the median over predicted boxes whose best IoU is at least 0.3; `dx` and `dy` are predicted centre minus ground-truth centre, `d_bottom` is ground-truth bottom edge minus predicted bottom edge (positive: the predicted box ends higher on screen than the annotated box); the ratios are predicted size divided by ground-truth size. Matching is many-to-one (each tracked box picks its own best ground-truth box), so on an IoU-0 row the `gt_*` columns are arbitrary.

## Hypothesis: tracker lag

Verdict: not the cause, on the held-out clip and on the training clips.

- `tracks==dets` is 100% on every one of the 16 diagnosed clips (both `summary.md` files). In the JSON, `tracks_vs_detections` has `share_identical` 1.0 and `median_iou` 1.0 for CAR_VS_BOS_001 (`data/runs/diagnose/heldout/CAR_VS_BOS_001.json`) and for all 11 training clips.
- So ByteTrack's output boxes are the detector's own boxes, unchanged. The tracker does not smooth or predict positions into its output, so it cannot add lag on top of the detector.
- The tracker does drop some detections (it only emits confirmed tracks): CAR_VS_BOS_001 has 49,377 tracked boxes from 52,181 detections (`CAR_VS_BOS_001.json`, `tracks.n_pred` and `detections.n_pred`). The tracked share at IoU 0.5 or more (72.0%) is slightly higher than the detection share (69.4%, same file), so dropping boxes does not explain the poor overlap; it removes some of the worst ones.

## Hypothesis: annotation style

Verdict: a fixed box-drawing difference exists but is small and equal on good and bad clips; the poor clips are explained by a timing offset between ground truth and picture that grows along the clip. That second part is from analysis I added beyond the brief's table columns (see "Time offset" below). I call the timing offset an annotation or frame-alignment effect; the data cannot say which of the two it is (see "What the evidence cannot decide").

Box shape, from the summary tables and `<clip>.json` (`tracks.matched_medians`):

| Clip (split) | IDF1@0.5 | med dx | med dy | med d_bottom | med w ratio | med h ratio |
|---|---|---|---|---|---|---|
| CAR_VS_BOS_001 (held-out) | 59.4% | -0.2 | 1.2 | 3.4 | 0.90 | 0.92 |
| STL_VS_SJ_003 (train, worst) | 60.0% | 0.5 | 1.0 | 3.6 | 0.90 | 0.92 |
| CHI_VS_TOR_004 (train, second worst) | 60.2% | 1.0 | 1.3 | 2.9 | 0.90 | 0.94 |
| CHI_VS_TOR_003 (train, best) | 93.1% | 0.3 | 1.2 | 3.6 | 0.88 | 0.93 |

- The detector's boxes are about 7-12% narrower and 6-8% shorter than the annotated ones, and end about 3 px higher, on all four clips. The best clip, CHI_VS_TOR_003, has the same shape difference (0.88 / 0.93 / 3.6 px) as the worst. Centre offsets (`dx`, `dy`) are 1 px or less in x and about 1 px in y everywhere. A shape difference that is the same on a 93% clip and a 60% clip cannot be what separates them.
- It is a general difference in how the detector and the annotators draw a box, not an annotation quirk of CAR_VS_BOS_001: the other held-out clips and the other training clips show the same pattern (width ratio 0.87 to 0.96, height ratio 0.91 to 0.98, `d_bottom` mostly 2.6 to 3.6 px; CNTRL_VS_PAC_002, PIT_VS_SJ_002 and PIT_VS_SJ_003 are exceptions at about -0.2 to 1.8 px).
- How much it costs: computed from `data/runs/plan2a-*/<clip>/tracks.parquet` and the ground truth (`data/tmp/offset_sim.py`), I widened every tracked box by 1/0.913 and heightened it by 1/0.941 (the median of the 11 training clips' per-clip median ratios), keeping the bottom edge fixed. The share of tracked boxes at IoU 0.5 or more rose from 72.0% to 74.9% on CAR_VS_BOS_001 (held-out), 75.8% to 78.0% on STL_VS_SJ_003, 87.1% to 88.9% on CHI_VS_TOR_004, and 97.6% to 98.3% on CHI_VS_TOR_003. A gain of 1 to 3 points does not account for the 22% to 24% of boxes below 0.5 on the poor clips.

Time offset (computed from the tracks tables and the ground truth, `data/tmp/shift_sim.py` and `data/tmp/drift.py`): I compared each tracked box at frame f with the ground truth labelled f-k, for several k, and measured the share at IoU 0.5 or more. k=0 is the real labelling.

| Clip (split) | share >=0.5 at k=0 | best single k tried | share at that k |
|---|---|---|---|
| CAR_VS_BOS_001 (held-out) | 72.0% | +4 | 87.6% |
| STL_VS_SJ_003 (train) | 75.8% | +4 | 92.3% |
| CHI_VS_TOR_004 (train) | 87.1% | +1 | 94.8% |
| CHI_VS_TOR_003 (train) | 97.6% | 0 | 97.6% |

(k tried: -8, -4, -2, -1, 0, 1, 2, 4, 8.) The ground truth matches the picture best when it is read 1 to 4 frames earlier, and the better the clip's score, the smaller that shift. The best shift is not constant along a clip; splitting each clip into five equal frame ranges (`data/tmp/drift.py`, k tried -2 to 9):

| Frame range, best k (share >=0.5 at best k, at k=0) | CAR_VS_BOS_001 (held-out) | STL_VS_SJ_003 (train) | CHI_VS_TOR_004 (train) | CHI_VS_TOR_003 (train) |
|---|---|---|---|---|
| 1st fifth | +1 (92.8%, 89.7%) | +1 (98.4%, 96.0%) | +1 (94.1%, 89.2%) | 0 (95.5%, 95.5%) |
| 2nd fifth | +2 (94.0%, 80.9%) | +2 (98.3%, 90.9%) | +1 (97.6%, 93.8%) | 0 (99.2%, 99.2%) |
| 3rd fifth | +4 (96.3%, 69.2%) | +4 (96.8%, 74.7%) | +1 (96.8%, 89.2%) | 0 (96.9%, 96.9%) |
| 4th fifth | +5 (96.4%, 66.9%) | +5 (97.8%, 52.0%) | +2 (97.7%, 86.7%) | 0 (98.2%, 98.2%) |
| 5th fifth | +6 (95.3%, 47.1%) | +6 (99.0%, 63.1%) | +2 (94.2%, 78.6%) | 0 (97.6%, 97.6%) |

On CAR_VS_BOS_001 and STL_VS_SJ_003 the offset grows by roughly one frame per 1,000 frames, from 1 frame at the start to 6 at the end. With the best shift per fifth, 93% to 99% of boxes reach IoU 0.5 in every fifth. CHI_VS_TOR_004 drifts more slowly (1 to 2 frames over 1,618 frames), and CHI_VS_TOR_003 does not drift. This growing offset is the main cause of the low-IoU boxes in the poor clips. Only these four clips were tested; I did not run the shift test on the other 12 clips.

## Hypothesis: the 0.5 overlap rule

Verdict: not the cause, but it amplifies the offset. A box that is a few frames off still overlaps its player, and 0.5 is where that stops counting; the box-shape difference does not matter much to the rule.

- IDF1 against the IoU threshold (`summary.md` files): CAR_VS_BOS_001 (held-out) 59.4% at 0.5, 67.9% at 0.4, 73.5% at 0.3, a gain of 14.1 points. Training: STL_VS_SJ_003 60.0 / 66.6 / 71.5 (+11.5), CHI_VS_TOR_004 60.2 / 64.0 / 66.3 (+6.1), CHI_VS_TOR_003 93.1 / 94.4 / 94.7 (+1.6). Poor clips gain several points when the rule is relaxed, the good clip barely moves. This is what the brief's test looks for.
- IoU by box size (`iou_by_area_tercile`, median IoU for the clip's smallest, middle and largest third of tracked boxes by area, `tracks` block of each JSON):

| Clip (split) | small | medium | large | large minus small |
|---|---|---|---|---|
| CAR_VS_BOS_001 (held-out) | 0.587 | 0.660 | 0.688 | 0.101 |
| STL_VS_SJ_003 (train) | 0.607 | 0.679 | 0.717 | 0.110 |
| CHI_VS_TOR_004 (train) | 0.640 | 0.717 | 0.752 | 0.112 |
| CHI_VS_TOR_003 (train) | 0.718 | 0.780 | 0.821 | 0.103 |

Small boxes score about 0.10 lower than large ones on all four clips, good and bad. The size gap does not separate the poor clips from the good one: the whole curve sits lower on the poor clips. So the 0.5 rule is not especially punishing small boxes; a clip-wide shift is lowering every size, and the rule counts the shifted boxes as misses. The terciles are cut per clip (tercile edges from that clip's own box areas), so "small" is relative to the clip.
- The spec's measure stays at 0.5. The finding about the rule is only that the poor clips' scores are sensitive to it.

## What the frames show

Colours: green = ground truth, red = tracked box with its track id, thin blue = raw detection. Frame indices are 0-based, from the file names in `data/runs/diagnose/<split>/frames/<clip>/`. Blue boxes sit on the red ones in every frame I opened except where noted, which agrees with `tracks==dets` of 100%; where a blue box shows alone, the tracker did not emit that detection.

CAR_VS_BOS_001 (held-out), frames 0, 724, 2172, 2896, 3620, 5068, 6516:
- Frame 0: no red at all, green and blue coincide (the tracker has not confirmed a track yet). Frame 724: red sits inside or on the green boxes for the players and referee (ids 23, 24, 15, 21), within a few px; red is a little narrower than green. Goalie (id 28) and the player in front of the net: red sits left of and inside green.
- Frame 2172: red is left of green on the referee (id 55), the player at lower right (id 54) and the group near the net (id 60); on the left (id 58) and bottom (ids 59, 52) red and green nearly agree.
- Frames 2896 and 3620: red hugs the players; green is shifted sideways by roughly 15 to 30 px. At 2896 green is right of red on the goalie (id 80), the player (id 79) and the pair (id 78). At 3620 green is left of red on ids 87, 83, 86, 95 and 85, and right of red on id 102 and the referee at top; a referee at the top (id 109) has green left of red with no overlap.
- Frame 5068: green is right of red on both referees (ids 140, 138) by about 25 px, and left of red on id 136; id 122 has red inside green. Frame 6516: the shift is smaller and mixed; ids 168, 175, 163, 166 are well matched, ids 177 and 173 (pile-ups near the boards) are offset.
- The direction of the offset changes with the player and frame (green sometimes left, sometimes right of red), consistent with a timing lag against moving players rather than a fixed pixel offset. Red-with-no-green and green-with-no-red are few; overall the problem is displacement, not missed players.

CAR_VS_BOS_002 (held-out), frames 294, 1175, 1762, 2643:
- Frames 294 and 1175: red and green agree closely on nearly all players (e.g. ids 0, 3, 5, 15, 21, 27). Frame 1762: green is a little left of red on ids 34 and 51 and the player at left; a far-boards referee at upper left has a green box with no red box. Frame 2643: green sits a little left of red on ids 63, 66, 65; the group at the boards (ids 61, 64) is mixed. This clip scores 97.5% at 0.5; the shift test was not run on it.

STL_VS_SJ_003 (train), frames 670, 2678, 4687, 6026:
- Frame 670: red and green agree (ids 10, 2, 17, 11, 4, 0, 21); small shifts in the pile (ids 15, 20).
- Frame 2678: red inside or on green for most players; ids 66 (left edge) and 56 have green slightly left; a spectator at the bottom (id 54) has red but no green (a false box from the detector on a crowd member).
- Frame 4687: green is shifted left of red by about 40 to 50 px on the skaters (ids 129, 127, 125, 126) and a green box at the right edge has no red (no blue either). The goalie (id 123) and the player at left edge (id 124) match.
- Frame 6026: green left of red on ids 141, 146, 148 and the referee; goalie (id 145) and ids 147, 132 are close. The offset at the end of the clip is the largest I saw on this clip, consistent with the drift table.

CHI_VS_TOR_004 (train), frames 180, 719, 1258, 1617:
- Frame 180: red is right of green by 10 to 25 px for most players (ids 13, 3, 6, 0, 11, 12); a spectator at lower left (id 14) has a red box with no green box (false positive on a person in the stands). A green box at lower right has only a blue detection (no red).
- Frame 719: red is right of green on ids 11 (left), 41; mixed in the central pile (ids 38, 39, 42). Frame 1258: red and green agree on id 68 and ids 47, 44; the pile at the net (ids 46, 60, 67) is offset by 10 to 30 px.
- Frame 1617: red is right of green by 20 to 40 px on ids 60, 66, 67 and 58, left on 86; goalie (id 28) is shifted. This is the end of the clip and has the largest offset of the four frames, consistent with the drift table.

CHI_VS_TOR_003 (train, best clip), frames 118, 474, 829, 1066:
- All four: red and green agree to within a few px on every player and referee (frame 118 ids 8, 15, 16, 17, 18; frame 474 the four skaters and the goalie; frame 829 ids 47, 48, 42, 41, 50, 38, 37, 49, 52, 53; frame 1066 ids 41, 55, 56, 57, 58). Red is a little narrower than green. Red-with-no-green and green-with-no-red: none seen. Frame 829: a goalie at the left edge (id 54) is cut by the frame.

Frames not opened: (CAR_VS_BOS_001 frames 1448, 4344 and 5792; CAR_VS_BOS_002 frames 0, 587, 881, 1468, 2056, 2349; STL_VS_SJ_003 frames 0, 1339, 2009, 3348, 4017, 5356; CHI_VS_TOR_004 frames 359, 539, 898, 1078, 1437; CHI_VS_TOR_003 frames 0, 237, 355, 592, 711, 948).

## What the evidence cannot decide

- Whether the growing time offset is a property of the annotation (ground-truth labels time-stamped against a slightly different clock) or of how the frames were extracted. In `data/raw/VIP-HTD`, CAR_VS_BOS_001's `img1` folder and its ground truth both have 6,517 frames (`seqinfo.ini`: `seqLength=6517`, `frameRate=60`), while the source video `clips/CAR_VS_BOS/001.mp4` reports 6,551 frames at 59.93 fps (read with OpenCV). CHI_VS_TOR_003 also has a longer mp4 (1,113 frames vs 1,067) yet shows no offset, so the frame-count difference alone does not predict the drift. I did not look further.
- Whether the drift is linear. Five frame ranges per clip show a steady rise on the two long 60 fps clips, but five points do not prove a straight line, and I did not test the other clips, so I do not know how common the effect is. The summary's per-clip share of boxes under 0.5 is larger on some other clips (for example CAR_VS_NYR_003 at 20.6%, PIT_VS_SJ_001 at 17.5%, PIT_VS_WAS_001 at 12.2%, `summary.md` files); whether those clips have the same drift is untested.

## Conclusion

- On CAR_VS_BOS_001 (held-out): the tracker is not the cause (tracked boxes are the detector's boxes, 100% identical). A fixed box-shape difference exists (detector boxes about 10% narrower and 8% shorter, ending about 3.4 px higher than annotated) but is the same as on the best clips and worth about 3 points of the 22% of boxes below IoU 0.5. The main cause is a timing offset between ground truth and picture that grows from about 1 frame at the start to about 6 frames at the end of the clip (about 17 to 100 ms at 60 fps). Aligning for it lifts the share of boxes at IoU 0.5 or more from 72.0% to 87.6% with one global shift, and to 93% to 96% within each fifth of the clip with a per-fifth shift. The 0.5 rule is not the cause but magnifies it: IDF1 rises from 59.4% to 73.5% when the threshold is relaxed to 0.3.
- The same pattern on training clips: yes, in proportion to how poorly they score. STL_VS_SJ_003 (60.0%) shows the same drift, 1 to 6 frames, and the same IDF1 sensitivity to the threshold (+11.5 points at 0.3). CHI_VS_TOR_004 (60.2%) shows a smaller drift (1 to 2 frames). CHI_VS_TOR_003 (93.1%) shows none. The box-shape difference is present on all of them equally.

## Recommendation

Justified from training clips only (STL_VS_SJ_003, CHI_VS_TOR_004, CHI_VS_TOR_003 and the training summary table):

1. Do not tune the detector or tracker to fix this, and do not treat the box-shape difference as a defect worth a detector change. On the training clips the shape correction was worth only 1 to 2 points at IoU 0.5 (`offset_sim.py` figures above), and the tracker passes the detector's boxes through untouched. If a detector change is ever considered for box shape, it belongs to decision point 1b; this investigation gives no reason to bring it forward.
2. Treat low scores at 0.5 on the long 60 fps clips as partly a labelling-time effect, not as tracker quality. On STL_VS_SJ_003 a global shift of 4 frames moves the share at IoU 0.5 from 75.8% to 92.3%, which a tracker change cannot do. Any tracker-settings comparison on training clips should therefore be read on the change between settings on the same clip, not as an absolute distance from the ceiling.
3. Report IDF1 at 0.3 and 0.4 alongside the spec's 0.5 in the tracker sweep summaries, as an addition and not a replacement; the spec's measure stays at 0.5. On training clips the gap between 0.5 and 0.3 is a cheap marker of this timing offset: +11.5 points on STL_VS_SJ_003, +6.1 on CHI_VS_TOR_004, +1.6 on CHI_VS_TOR_003 (`train/summary.md`).
4. Before the final held-out measurement, run the shift test (`data/tmp/shift_sim.py` and `drift.py`) on the remaining training clips to learn how common the drift is and whether it correlates with fps or clip length, and find out whether it comes from the annotation or from frame extraction. Do not rewrite ground truth or re-align frames until that is known; that is an owner decision.
