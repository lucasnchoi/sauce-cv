# Rink mapping results (Plan 2b-ii)

Measured on: 2026-10-08. Run: `data/runs/plan2b-heldout` (`run.json` records the weights path, thresholds and settings). Scorecard: `data/runs/plan2b-heldout/scorecard.md`. The five held-out clips (`splits/heldout.json`) were never trained on: `train-rink` drops their labels before training, and they were not used to choose weights or settings.

## Labels

From `labels/rink_keypoints.parquet` and `labels/rink_frames.json` (commit d28703c):

| Subset | Frames | Clips | Visible points |
|---|---|---|---|
| Training | 408 | 17 | 2,812 |
| Evaluation (held-out clips) | 50 | 5 | 346 |

Median 7 visible points per frame (mean 6.9; 3,158 in all). 81 training frames were labelled cold (the Plan 2b-i stop, commit 0419762); the other 377 (327 training + 50 evaluation) were pre-labels from the round-1 heatmap model (`data/runs/prelabel-r1h`, weights `exp-pw10`) corrected by the owner with `label-rink --prelabels`. Before the commit every frame was checked for 33 rows, split membership, no two visible points within 6 px, and a leave-one-out homography on every frame with 5 or more visible points; the owner fixed the 13 duplicate-dot frames and 8 wrong-identity frames that turned up.

## Model

The model is `RinkHeatmapNet` (ResNet-50 trunk, three deconvolutions, 33 heatmaps at stride 4, input 576×1024, BCE with `pos_weight` on the positive pixels; see `docs/superpowers` Plan 2b-ii Amendment, Task 11). All runs: batch 4, AdamW lr 1e-3, seed 20261007, validation share 0.2 from the same clips, device mps. Each run's `run.json` beside the weights holds the settings, package versions and the per-epoch history; these figures come from those files.

| Round | Model | Train / val frames | Epochs | Best epoch | PCK@10 px | Mean error px | Minutes | Weights |
|---|---|---|---|---|---|---|---|---|
| 1 (R-CNN) | torchvision Keypoint R-CNN | 65 / 16 | 30 | 20 | 76.9% | 86.3 | 15.3 | `data/models/rink-keypoints/r1/best.pt` |
| 1 (heatmap, first) | `RinkHeatmapNet`, no `pos_weight` | 65 / 16 | 60 | 53 | 78.8% | 54.1 | 11.5 | `data/models/rink-keypoints/r1h/best.pt` |
| 1 (heatmap, accepted) | `RinkHeatmapNet`, `pos_weight` 10 | 65 / 16 | 100 | 71 | 88.5% | 24.0 | 19.0 | `data/models/rink-keypoints/exp-pw10/best.pt` |
| 1 (heatmap, comparison) | `RinkHeatmapNet`, `pos_weight` 2 | 65 / 16 | 100 | 95 | 85.6% | 28.0 | 19.4 | `data/models/rink-keypoints/exp-pw2/best.pt` |
| 2 | `RinkHeatmapNet`, `pos_weight` 10 | 326 / 82 | 100 | 58 | 97.5% | 8.8 | 95.3 | `data/models/rink-keypoints/r2/best.pt` |

PCK and mean error are on the run's own validation frames at the best epoch (visible labelled points only). Round 1's R-CNN found visible points but gave every point, visible or not, confidence 1.0, which is why the heatmap model replaced it (Amendment, Task 11). Round 2 at its best epoch also found 96.6% of the visible points at the 0.5 confidence floor with 0.20 false points per frame (`r2/run.json`, epoch 58). Each epoch took 0.95 minutes, under the 10-minute stop line, so no GPU rental was needed.

`rink-check` on round 2's 82 validation frames (`--frames-from data/models/rink-keypoints/r2/run.json`, threshold 0.3, held-out clips refused), both weights:

```
exp-pw10: {"pck": 0.8987, "mean_px": 8.24, "n_points": 553, "n_predicted": 516, "n_frames": 82, "n_valid": 65, "valid_share": 0.7927, "median_m": 0.0791}
r2:       {"pck": 0.9602, "mean_px": 2.93, "n_points": 553, "n_predicted": 538, "n_frames": 82, "n_valid": 73, "valid_share": 0.8902, "median_m": 0.1168}
```

Chosen weights: `data/models/rink-keypoints/r2/best.pt`. PCK is the plan's criterion and r2 leads it by 6 points, with more frames reaching a valid transform; its median distance is 4 cm higher, which is within the spread of 73 versus 65 frames, and `exp-pw10` had trained on some of these frames, so the comparison favoured it if anything. The slow test (`tests/test_rink_model_slow.py`) runs on these weights.

## Held-out scorecard

From `data/runs/plan2b-heldout/scorecard.md`.

Rink transform:

| Clip | Median | Valid share | Points | Play frames | Labelled | Scored | No transform | Non-play | Meets bar |
|---|---|---|---|---|---|---|---|---|---|
| **Overall** | 0.15 m | 98.3% | 346 | 21780 | 50 | 45 | 1 | 4 | yes |
| CAR_VS_BOS_001 | 0.16 m | 99.5% | 85 | 5997 | 10 | 10 | 0 | 0 | yes |
| CAR_VS_BOS_002 | 0.15 m | 98.1% | 70 | 2131 | 10 | 9 | 0 | 1 | yes |
| CAR_VS_NYR_001 | 0.14 m | 95.9% | 54 | 4992 | 10 | 8 | 0 | 2 | yes |
| CAR_VS_NYR_002 | 0.12 m | 99.6% | 76 | 4013 | 10 | 10 | 0 | 0 | yes |
| CAR_VS_NYR_003 | 0.16 m | 98.5% | 61 | 4647 | 10 | 8 | 1 | 1 | yes |

Tracking (IDF1 at IoU 0.5; Plan 2a's figure for the same clip in the last column, from `docs/baseline-results.md`):

| Clip | IDF1 | IDF1@0.3 | Precision | Recall | Ground-truth boxes | False boxes | Missed boxes | Plan 2a IDF1 |
|---|---|---|---|---|---|---|---|---|
| **Overall** | 62.9% | 68.1% | 87.6% | 74.5% | 202273 | 21295 | 51611 | 70.6% |
| CAR_VS_BOS_001 | 57.5% | 69.9% | 73.3% | 62.6% | 53436 | 12181 | 19995 | 59.4% |
| CAR_VS_BOS_002 | 73.6% | 74.3% | 98.1% | 73.6% | 22814 | 319 | 6012 | 86.0% |
| CAR_VS_NYR_001 | 56.7% | 57.2% | 99.0% | 82.5% | 45854 | 393 | 8038 | 71.4% |
| CAR_VS_NYR_002 | 75.6% | 76.2% | 97.4% | 86.0% | 37919 | 865 | 5297 | 81.4% |
| CAR_VS_NYR_003 | 59.3% | 67.0% | 79.9% | 71.0% | 42250 | 7537 | 12269 | 65.7% |

Bars: median ≤ 1.0 m — **met (0.15 m)**; valid transform on ≥ 90% of play frames — **met (98.3%)**. Every clip meets both bars on its own. The tracking bar (IDF1 85%) is not met (62.9%); see Reading.

Overlay check: two stills from `CAR_VS_BOS_001/overlay.mp4` at 10 s and 60 s (`data/tmp/overlay-check-10.jpg`, `-60.jpg`). In both, the rink view puts each player's dot in the zone the camera shows (between the centre line and the near blue line at 10 s, around the near blue line at 60 s), coloured by track id.

## Reading

- **Per clip.** No clip misses a rink bar; the worst valid share is CAR_VS_NYR_001 at 95.9% (207 invalid play frames of 4,992; `rink_transforms.parquet` statuses: 4,624 fitted, 161 interpolated, 207 invalid). Across the five clips 20,770 play frames were fitted, 646 interpolated and 364 invalid.
- **Labelled evaluation frames without a transform: 1** (CAR_VS_NYR_003 frame 742: 5 points found at the 0.5 floor, RANSAC left 0 inliers, status invalid). **Labelled frames the filter called non-play: 4** (CAR_VS_BOS_002 frame 1386; CAR_VS_NYR_001 frames 354 and 1276; CAR_VS_NYR_003 frame 1944). All four show live play from a view with only 4 or 5 confident keypoints, under the filter's floor of 6 (`run.json`: `min_keypoints` 6, `min_conf` 0.5). A filter matter, not a model one: the points it found in those frames sit on the right rink features (`data/tmp/dropped-*.jpg`).
- **Footage filter.** Share of frames called play, from each clip's `frames.parquet`: CAR_VS_BOS_001 92.0%, CAR_VS_BOS_002 80.6%, CAR_VS_NYR_001 93.4%, CAR_VS_NYR_002 93.7%, CAR_VS_NYR_003 95.2%; 1,886 of 23,666 frames (8.0%) were called non-play. Every held-out clip is broadcast play, and stills from the middle of the three longest non-play segments (`data/tmp/nonplay-*.jpg`: CAR_VS_BOS_002 frames 1148–1486, CAR_VS_BOS_001 5742–6107, CAR_VS_NYR_001 319–411) all show play: tight views along the benches or deep in an end zone where only 3 to 5 keypoints clear the floor. So the whole 8.0% is filter error, concentrated in CAR_VS_BOS_002 (19.4%). The ground truth has 15,354 boxes on those frames (7.6% of 202,273).
- **Tracking IDF1 versus Plan 2a's 70.6%.** The drop to 62.9% comes from the filter in two ways. (1) Dropped frames: the pipeline writes no tracks on non-play frames, so their 15,354 ground-truth boxes are all missed; this is the recall drop from 80.6% to 74.5%. (2) Segment restarts: stage 4 starts a fresh tracker per play segment (spec §7), so every false non-play gap splits the tracks either side of it into new ids. Scoring only the play frames (ground truth and tracks both restricted to them, `score_tracking` from `tracker.evaluate.tracking`) gives precision 87.6% and recall 80.6%, the same as Plan 2a, but IDF1 65.6% (per clip: 59.9%, 82.5%, 58.8%, 77.8%, 60.8%), and the clips with the most segments lose the most identity: CAR_VS_NYR_001 has 8 segments and 176 track ids against Plan 2a's 125, CAR_VS_NYR_003 7 segments and 189 against 143, while CAR_VS_BOS_001 with 4 segments is at 59.9% against 59.4%. Detection and the tracker are unchanged from Plan 2a (same detector, same ByteTrack settings, `run.json`).
- **Duplicate faceoff dots.** The owner's observation while correcting round-1 pre-labels was that one visible dot often carried several dot ids. On round 2's held-out output (`keypoints.parquet`, confidence ≥ 0.5) two faceoff-dot ids within 20 px of each other occur in 3 of 23,666 frames (CAR_VS_NYR_002, two frames, ids 11 and 13; CAR_VS_NYR_001, one frame), so round 2 has removed it. What remains within 8 px is the centre dot (id 10) with the centre-circle top (id 27) in 69, 54 and 86 frames of the three NYR clips, and the far goal-line boards point (id 0) with the far centre-line boards point (id 8) in 6 frames of CAR_VS_BOS_001; both are identity confusions between two different features, rare enough that RANSAC absorbs them (every clip meets the bars). No suppression was added.
- **Speed.** The run processed 23,666 frames in 48 min 14 s (`run.json` `started_at` 22:23:21 UTC to the last overlay at 23:11:35 UTC): 8.2 frames/s with overlays, against Plan 1's 13.6 frames/s on the same clips (`docs/baseline-results.md`, measured while sharing the GPU with a second run). From the output files' timestamps, stage 1 (keypoints on every frame) took 28 min 53 s of the 48 (13.7 frames/s on its own, including the model load), detection and tracking 16 min 40 s, and the overlays 2 min 40 s. The stage-1 cost is the 19-minute difference from Plan 1, close to doubling the run time.

## Decision

The rink positions are ready: both bars are met on every held-out clip with a median of 0.15 m, the model's visibility signal works (0.20 false points per frame on validation, and off-screen points stay under the floor), and the pre-label loop is in place for more labels later. The tables say the bottleneck is now the footage filter, not the model or the transform: it calls 8.0% of live play non-play because the floor of 6 confident keypoints is above what tight bench-side and end-zone views offer, and that costs 6 recall points and 5 IDF1 points of identity through the per-segment tracker restarts, on top of the identity losses Plan 2a already had. Recommendation: proceed to Plan 3 (teams and identity) on these positions, and take the filter's floor (`min_keypoints` 6 → 4 or 5, or a transform-validity test instead of a count) into Plan 3's first task or a short Plan 2c, since Plan 3's identity work will be scored on the same IDF1 and should not start 5 points behind Plan 2a for a filter setting. More labels or a larger model are not indicated by these tables.
