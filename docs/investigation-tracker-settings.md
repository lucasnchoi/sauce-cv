# Investigation 2: ByteTrack settings

Checked on: 2026-10-07. Source run: `data/runs/plan2a-train` (detections cached; detector unchanged: RF-DETR Medium "person" at 0.5). Tables: `data/runs/grid/one-factor/grid.md`, `data/runs/grid/cross/grid.md`. Per-clip numbers are in the `per_clip` field of the matching `grid.json`; lines marked "computed from" come from a small read-only helper, `data/tmp/perclip7.py`.

## Method

Each grid point re-tracks the cached detections of the 17 training clips with one set of ByteTrack settings and scores the result against the ground truth; 11 clips are scorable, and the 6 with duplicate ground-truth ids (CHI_VS_TOR_002, CNTRL_VS_PAC_001, CNTRL_VS_PAC_005, CNTRL_VS_PAC_006, STL_VS_SJ_001, STL_VS_SJ_002) fail scoring at every point, as expected. No held-out clip was read or scored in this investigation.

The baseline is the library default (`data/runs/plan2a-train/scorecard.md`, Run block): `lost_track_buffer` 30, `track_activation_threshold` 0.7, `minimum_consecutive_frames` 2, `minimum_iou_threshold` 0.1, `high_conf_det_threshold` 0.6. Its pooled training IDF1 is 70.5%. Values tried: `track_activation_threshold` {0.5, 0.6}, `minimum_iou_threshold` {0.2, 0.3}, `lost_track_buffer` {15, 60, 90}, `minimum_consecutive_frames` {3, 4}, `high_conf_det_threshold` {0.5, 0.7}. The plan also listed `minimum_consecutive_frames` = 1; it was dropped because in `trackers` 2.6.1 a value of 1 behaves exactly like 2 (a track's spawn frame is always unconfirmed), so it would only repeat the baseline. The 12 one-factor points took 4 min 15 s (`data/runs/grid-one-factor.log`).

## One factor at a time

From `data/runs/grid/one-factor/grid.md` (11 clips scored and 6 failed on every row; the 6 failures are the duplicate-id clips above):

| Setting | IDF1 | Precision | Recall | Clips scored | Failed |
|---|---|---|---|---|---|
| high_conf_det_threshold=0.7 | 71.1% | 90.5% | 82.9% | 11 | 6 |
| high_conf_det_threshold=0.5 | 70.9% | 90.5% | 83.0% | 11 | 6 |
| track_activation_threshold=0.5 | 70.6% | 89.2% | 83.8% | 11 | 6 |
| track_activation_threshold=0.6 | 70.6% | 89.2% | 83.8% | 11 | 6 |
| baseline | 70.5% | 90.5% | 82.9% | 11 | 6 |
| lost_track_buffer=60 | 70.5% | 90.5% | 82.9% | 11 | 6 |
| lost_track_buffer=90 | 70.5% | 90.5% | 82.9% | 11 | 6 |
| minimum_consecutive_frames=3 | 70.5% | 90.6% | 82.6% | 11 | 6 |
| minimum_consecutive_frames=4 | 70.4% | 90.7% | 82.3% | 11 | 6 |
| lost_track_buffer=15 | 70.3% | 90.5% | 82.9% | 11 | 6 |
| minimum_iou_threshold=0.2 | 69.7% | 90.5% | 82.8% | 11 | 6 |
| minimum_iou_threshold=0.3 | 68.5% | 90.5% | 82.6% | 11 | 6 |

The whole table spans 68.5% to 71.1%, a 2.6-point range, and only 0.6 points sit above the baseline.

## Cross

From `data/runs/grid/cross/grid.md` (definition: `data/runs/grid/cross.json`):

| Setting | IDF1 | Precision | Recall | Clips scored | Failed |
|---|---|---|---|---|---|
| track_activation_threshold=0.5,high_conf_det_threshold=0.7 | 71.1% | 90.5% | 82.9% | 11 | 6 |
| track_activation_threshold=0.6,high_conf_det_threshold=0.7 | 71.1% | 90.5% | 82.9% | 11 | 6 |
| high_conf_det_threshold=0.7 | 71.1% | 90.5% | 82.9% | 11 | 6 |
| track_activation_threshold=0.5 | 70.6% | 89.2% | 83.8% | 11 | 6 |
| track_activation_threshold=0.6,high_conf_det_threshold=0.5 | 70.6% | 89.2% | 83.9% | 11 | 6 |
| baseline | 70.5% | 90.5% | 82.9% | 11 | 6 |
| track_activation_threshold=0.5,high_conf_det_threshold=0.5 | 69.0% | 88.0% | 84.2% | 11 | 6 |

How it was chosen: only two factors beat the baseline in the one-factor table, `high_conf_det_threshold` (0.7: +0.6, 0.5: +0.4) and `track_activation_threshold` (0.5 and 0.6: +0.1 each). `lost_track_buffer` 60 and 90 only tied the baseline, so they were not crossed. The cross is all four combinations of those two factors' values, plus the two best singles for reference (6 points).

## What the settings did

Reading the one-factor table (values to one decimal from `grid.md`; where I give a third decimal it is computed from `data/runs/grid/one-factor/grid.json`). The reasons below are plausible mechanisms, not something these outputs prove.

- `high_conf_det_threshold` (0.6 default). 0.7 gave +0.6 IDF1 (71.1%) with precision and recall unchanged at one decimal (90.5% and 82.9%). 0.5 gave +0.4 (70.9%), also with unchanged precision and recall (83.0% recall). So the gain is not from more or fewer boxes: the same boxes carry better identities. This factor sets which detections go to ByteTrack's first association stage and which to the second, lower-confidence stage; moving the line changes which detections can match and which can start tracks, so it reassigns identities rather than adding or removing boxes. The effect is small and moves both ways: 0.5 and 0.7 both beat 0.6, so it looks like noise around a flat optimum more than a direction.
- `track_activation_threshold` (0.7 default). 0.5 and 0.6 give the same result to every digit stored (70.606% IDF1, 89.198% precision, 83.838% recall): +0.1 IDF1, precision down 1.3 points, recall up 0.9. Lowering the threshold lets detections scoring below 0.7 start tracks (the detector threshold is 0.5), so recall rises and precision falls, as expected. 0.5 and 0.6 being identical fits with detections below `high_conf_det_threshold` (0.6) not being able to start a track in this configuration, but I did not verify that against the library code.
- `lost_track_buffer` (30 default). 60 and 90 are identical to each other and to the baseline at one decimal (70.5%, 90.5%, 82.9%); the third decimal differs by 0.001 IDF1, and only 3 of 11 clips changed at all (computed from `grid.json`). 15 loses 0.2 (70.3%) with precision and recall still at 90.5% and 82.9%, so only identity bookkeeping changed, in 8 of 11 clips. Lengthening the buffer has no measurable effect, which says that re-linking a track after a gap longer than 30 frames is almost never what the matcher gets right or wrong here; shortening it breaks a few links.
- `minimum_consecutive_frames` (2 default). 3 and 4 move IDF1 by -0.1 and -0.2, precision up (90.6%, 90.7%) and recall down (82.6%, 82.3%). Requiring more frames to confirm a track drops short, doubtful tracks (fewer false boxes) and also the first frames of real tracks (more missed boxes); the two roughly cancel, with a small net loss.
- `minimum_iou_threshold` (0.1 default). The only clearly harmful factor: 0.2 gives -0.8 (69.7%) and 0.3 gives -2.0 (68.5%), with precision and recall almost unchanged (90.5% to 90.5%, recall 82.9% to 82.6%). A stricter overlap gate refuses matches between a track and the next frame's detection when the box moved or changed size more than the gate allows; the track then breaks and restarts under a new id, which costs identity but not box counts. This fits ByteTrack having no motion cue strong enough to compensate: fast skaters in 60 fps and 30 fps footage need a loose gate. The cross-term evidence points the same way (below).

Settings with no effect on precision and recall at one decimal: `lost_track_buffer` 15, 60 and 90, `high_conf_det_threshold` 0.5 and 0.7 (the 0.5 recall differs by 0.1), and `minimum_iou_threshold` 0.2 and 0.3 (precision 90.5%; recall 82.8% and 82.6%). For these, IDF1 moves while the number of boxes right and wrong does not: the settings only rearrange which id sits on which box. That is informative, since it means identity switches and fragmentation, not box counts, are where these settings act, and they change IDF1 by at most 0.6 points up or 2.0 points down.

The cross shows the gains do not add up. `track_activation_threshold` 0.5 or 0.6 combined with `high_conf_det_threshold` 0.7 gives exactly the 0.7 result (71.1%, 90.5%, 82.9% in both rows), so the activation threshold no longer matters once the high-confidence line is at 0.7 (a track can only start from a detection above both). Combining both low values (0.5 and 0.5) is the worst point in the cross: 69.0% IDF1, 88.0% precision, 84.2% recall, 1.5 points under the baseline, since it lets the most low-score boxes start tracks.

## Chosen default

Baseline stays: nothing beat it by 1.0 point. The best points (`high_conf_det_threshold=0.7`, and its two crosses with `track_activation_threshold` 0.5 and 0.6) reach 71.1% pooled training IDF1 against the baseline's 70.5%, a gain of 0.6 points (computed from `grid.json`: 71.085% against 70.542%, 0.54 points). That is below the plan's 1.0-point margin. So there is no candidate: the held-out clips were not measured, and the `ByteTrackSettings` defaults and their test are unchanged.

## Per-clip view

Computed from `data/runs/grid/one-factor/grid.json` and `data/runs/grid/cross/grid.json` (`per_clip`; the baseline per-clip values match `data/runs/plan2a-train/scorecard.md`). Frame rates: 30 fps are CHI_VS_TOR_003, CHI_VS_TOR_004, CNTRL_VS_PAC_002 and PIT_VS_SJ_001/002/003; 60 fps are CGY_VS_DAL_001/002/003, PIT_VS_WAS_001 and STL_VS_SJ_003 (`docs/data-verification.md` and `clip.json`). Changes in IDF1 points against the baseline:

| Clip | fps | baseline IDF1 | high_conf=0.7 | activation=0.5 | min_iou=0.3 |
|---|---|---|---|---|---|
| CGY_VS_DAL_001 | 60 | 67.9% | +0.2 | -0.3 | -2.3 |
| CGY_VS_DAL_002 | 60 | 71.4% | +0.9 | +0.5 | -2.2 |
| CGY_VS_DAL_003 | 60 | 69.0% | -1.1 | -1.7 | -3.4 |
| CHI_VS_TOR_003 | 30 | 93.1% | -2.1 | -0.2 | -2.6 |
| CHI_VS_TOR_004 | 30 | 60.2% | +1.4 | -0.1 | -1.8 |
| CNTRL_VS_PAC_002 | 30 | 80.0% | +2.7 | -0.7 | -3.0 |
| PIT_VS_SJ_001 | 30 | 71.2% | +0.7 | +0.4 | 0.0 |
| PIT_VS_SJ_002 | 30 | 76.5% | +3.6 | -0.6 | -1.0 |
| PIT_VS_SJ_003 | 30 | 80.5% | +0.3 | -0.4 | -3.7 |
| PIT_VS_WAS_001 | 60 | 71.5% | -1.9 | +0.8 | -1.2 |
| STL_VS_SJ_003 | 60 | 60.0% | +1.0 | +1.3 | -1.2 |

- The best single point, `high_conf_det_threshold=0.7`, moves clips by -2.1 to +3.6 points; the pooled +0.6 is the net of 8 clips up and 3 down (CGY_VS_DAL_003 -1.1, CHI_VS_TOR_003 -2.1, PIT_VS_WAS_001 -1.9). The biggest gains are PIT_VS_SJ_002 (+3.6) and CNTRL_VS_PAC_002 (+2.7); the biggest loss is CHI_VS_TOR_003 (-2.1), the best-scoring clip (93.1%, the only clip over the 85% bar in the scorecard).
- 30 fps against 60 fps: for `high_conf_det_threshold=0.7`, the 30 fps clips average +1.1 points and the 60 fps clips -0.2; they do not move alike, but with 6 and 5 clips and per-clip swings of several points, I would not call this a real frame-rate effect. For `track_activation_threshold=0.5` the averages are -0.3 (30 fps) and +0.1 (60 fps), both near zero. For `minimum_iou_threshold=0.3`, they move alike: -2.0 (30 fps) and -2.1 (60 fps), and 10 of 11 clips fall (PIT_VS_SJ_001 is flat), so that loss is not a frame-rate effect.
- `minimum_iou_threshold=0.2` is uneven rather than uniformly bad: CGY_VS_DAL_003 -8.0 but PIT_VS_WAS_001 +2.5 (computed from `grid.json`), so the one pooled number hides large per-clip effects of both signs. The cross point 0.5 and 0.5 shows the same pattern, with CGY_VS_DAL_003 -5.0 and PIT_VS_SJ_003 -3.8.
- The weakest clips under the baseline, STL_VS_SJ_003 (60.0%) and CHI_VS_TOR_004 (60.2%), gain 1.0 and 1.4 points from `high_conf_det_threshold=0.7`, not enough to change their standing. These two clips also have the lowest IDF1 scores in `docs/investigation-box-offset.md`'s IoU-0.5 column; there, IDF1 at IoU 0.3 is 71.5% and 66.3%, so a large part of their gap comes from overlap rather than identity, and no ByteTrack setting touches that (the tracker outputs the detector's own boxes, per that note).

## Recommendation for decision point 1b

The tracker is not the main limit, on these outputs; detection and overlap are the better place to look, with one caveat that keeps this short of conclusive.

- Twelve ByteTrack settings across five factors, and a six-point cross, spread pooled IDF1 over 68.5% to 71.1%. The best is 0.6 points above the baseline; the remaining distance to the tracking bar of 85.0% (`scorecard.md`) is about 14 points. Tuning ByteTrack's own knobs moves a fraction of that, and the bar is not reachable by them.
- The baseline's pooled precision is 90.5% and recall 82.9% (`scorecard.md`): 22,702 false boxes and 44,491 missed boxes out of 260,768 ground-truth boxes. The knobs that change the box counts only trade them: `track_activation_threshold` 0.5 gave recall +0.9 for precision -1.3 and no IDF1 gain. The tracker emits the detector's own boxes unchanged (Investigation 1, `docs/investigation-box-offset.md`, `tracks==dets` 100% on all 11 clips), and the setting that lowers the confidence bar for starting tracks added under one point of recall at a larger cost in precision. I did not measure how many of the missed boxes the detector never produced, so I cannot say that split from these outputs; the grid only shows that the tracker's settings recover little of it.
- Where identity is lost: the settings that act purely on identity (`lost_track_buffer`, `minimum_iou_threshold`, `high_conf_det_threshold`; precision and recall near-constant) change IDF1 by -2.0 to +0.6. Longer memory (buffer 60 and 90) did nothing, so lost identity is not mainly about re-finding the same player after a gap of 30 to 90 frames; a stricter overlap gate made things worse, so ByteTrack's single IoU-based association is not too loose. ByteTrack has no appearance cue, so it cannot re-link a player after an occlusion by what he looks like; this grid cannot test that, because no setting adds an appearance cue. What this grid does show is that more or less of the same motion-only memory does not help.
- Caveat on the conclusion: this grid varies only the settings of one motion-only tracker. It cannot say whether a tracker with appearance features (or a different association rule) would recover identity that ByteTrack loses, and the per-clip spread (for example CGY_VS_DAL_003 -8.0 to PIT_VS_WAS_001 +2.5 under one setting) shows that clip-level identity behaviour is sensitive to something other than the box counts. The evidence therefore favours "detection first": improving detection recall, the largest component (44,491 missed boxes), would also feed any later tracker; trying another tracker is worth it only as a second step, and the property of ByteTrack to test is its lack of appearance matching, not its settings.
