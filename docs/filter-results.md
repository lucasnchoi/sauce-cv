# Footage filter results (Plan 2c)

Measured on: 2026-10-08. Runs: `data/runs/plan2c-train` (the 17 training clips, the cache), `data/runs/plan2c-train-{transform,keypoints,passthrough,bridge_gap_s-1.0,bridge_gap_s-2.0,min_segment_s-0.5}` (refiltered from it), `data/runs/plan2c-heldout` (the measurement). Each `run.json` records the filter mode and its numbers, the tracker settings and the weights path (`data/models/rink-keypoints/r2/best.pt`). Spec: `docs/superpowers/specs/2026-10-08-hockey-vision-tracker-plan-2c-footage-filter-design.md` (outside the repository).

## What changed

A frame is now play when its own rink transform fits (stage 3 runs on every frame first and writes `rink_fits.parquet`; the filter reads it), instead of when six keypoints clear a confidence floor; the old rule survives as `--footage-filter keypoints` and Plan 1's pass-through as `passthrough`. The person detector runs on every frame, so a run's cache covers the clip and `tracker refilter` can re-run the filter, the transform smoothing, tracking and positions under other settings in seconds. Tracking uses one ByteTrack per clip fed the play frames only, so a gap no longer renumbers the players. Settings were chosen on the training clips; the held-out clips were run once, at the end.

## Training clips: modes compared

From `data/runs/plan2c-train-*/scorecard.md` (tracking over the 11 training clips whose ground truth the scorer accepts; the 6 with duplicate ids are listed as failed, as in `docs/baseline-results.md`) and each clip's `frames.parquet` (play share, pooled over 50,224 frames):

| Mode | Play share (pooled) | IDF1 | Precision | Recall | Rink median | Valid share |
|---|---|---|---|---|---|---|
| transform, Plan 2's bridge 0.5 s, min segment 1.0 s | 95.0% | 65.1% | 90.5% | 79.1% | 0.09 m | 100.0% |
| keypoints (Plan 2 rule) | 85.5% | 59.8% | 90.6% | 73.0% | 0.10 m | 99.1% |
| passthrough (Plan 1) | 100% | 70.5% | 90.5% | 82.9% | 0.09 m | 95.3% |
| transform, bridge 1.0 s | 98.4% | 69.3% | 90.5% | 82.0% | 0.09 m | 96.8% |
| **transform, bridge 2.0 s (chosen)** | **99.4%** | **70.4%** | 90.5% | 82.6% | 0.09 m | 95.9% |
| transform, min segment 0.5 s | 95.2% | 65.3% | 90.5% | 79.2% | 0.09 m | 100.0% |

Precision is the same in every row: the filter and the per-clip tracker change which frames are tracked, not how well. Pass-through at 70.5% equals Plan 2a's pooled training IDF1 (`docs/investigation-tracker-settings.md`), which confirms that detecting every frame and running one tracker per clip reproduces Plan 2a's tracking exactly when nothing is filtered.

Play share per clip (`frames.parquet`):

| Clip | fps | Frames | keypoints | transform | transform + bridge 2.0 s |
|---|---|---|---|---|---|
| CGY_VS_DAL_001 | 60 | 3969 | 84.0% | 92.5% | 100.0% |
| CGY_VS_DAL_002 | 60 | 4169 | 95.5% | 99.1% | 100.0% |
| CGY_VS_DAL_003 | 60 | 3326 | 90.4% | 94.3% | 99.6% |
| CHI_VS_TOR_002 | 30 | 1585 | 88.2% | 96.8% | 99.6% |
| CHI_VS_TOR_003 | 30 | 1067 | 68.5% | 90.3% | 100.0% |
| CHI_VS_TOR_004 | 30 | 1618 | 75.0% | 92.8% | 99.3% |
| CNTRL_VS_PAC_001 | 30 | 1671 | 96.1% | 97.1% | 100.0% |
| CNTRL_VS_PAC_002 | 30 | 1965 | 95.0% | 96.2% | 100.0% |
| CNTRL_VS_PAC_005 | 30 | 2066 | 84.8% | 94.5% | 100.0% |
| CNTRL_VS_PAC_006 | 30 | 2646 | 89.6% | 94.1% | 97.1% |
| PIT_VS_SJ_001 | 30 | 1391 | 90.2% | 94.0% | 94.0% |
| PIT_VS_SJ_002 | 30 | 2762 | 89.0% | 93.9% | 99.7% |
| PIT_VS_SJ_003 | 30 | 3525 | 81.6% | 94.6% | 100.0% |
| PIT_VS_WAS_001 | 60 | 2981 | 94.9% | 98.8% | 100.0% |
| STL_VS_SJ_001 | 60 | 5647 | 70.2% | 93.9% | 97.9% |
| STL_VS_SJ_002 | 60 | 3809 | 86.7% | 95.2% | 100.0% |
| STL_VS_SJ_003 | 60 | 6027 | 82.7% | 95.1% | 100.0% |

Per-clip IDF1, chosen setting against pass-through (`scorecard.md` of each run): identical on 8 of the 11 scored clips; CHI_VS_TOR_004 59.9% vs 60.2%, PIT_VS_SJ_001 68.0% vs 71.2% (its remaining non-play stretch is the last 2.8 s of the clip), PIT_VS_SJ_002 76.4% vs 76.5%.

**Why frames still fail the transform rule, and the setting chosen.** Every stretch the `transform` rule called non-play that was checked by eye is live play (stills `data/tmp/p2c-nonplay-CGY1-*.jpg`, `data/tmp/p2c-rest-*.jpg`): tight end-zone views where the model finds four or five confident points. On CGY_VS_DAL_001, 223 of its 298 non-play frames had four or more confident points, yet the fit returned no inliers, and restricting the fit to points at confidence 0.5 or more changes nothing. Frame 458 shows why: its five points (goal-line boards, the end dot and three circle points) disagree with each other by 7 to 30 px in a leave-one-out least-squares fit, more than the 5 px RANSAC threshold, so no four of them agree. The heatmap's stride-4 grid is about 5 px at 1280 width on these zoomed views, and three of the five points lie on one circle diameter, so a five-point frame has no slack. This is not a filter defect; it is the transform's threshold meeting the model's localisation on few-point views, and it is the candidate for a later plan (a looser RANSAC threshold for few-point frames, or sub-cell peak refinement in the model). Within this plan the lever is the bridge: the dropped stretches are 1 to 3 s long, so `bridge_gap_s` 2.0 s recovers nearly all of them (play share 99.4%, lowest clip 94.0%, where the gap is the clip's final 2.8 s, which bridging never fills by design) and brings IDF1 to 70.4% against pass-through's 70.5%. The cost of the larger bridge is that a real cut shorter than 2 s would be called play on future footage; a replay or bench shot is normally longer. After the final review the code default moved to 2.0 s (the spec's procedure moves the value when the default clearly loses); the held-out run below was made with `--set footage.bridge_gap_s=2.0` before that change and `run.json` records 2.0 either way.

Speed of the training run (`data/runs/plan2c-train.log`, 21:48:28 to 23:29:19 local): 50,224 frames in 100 min 51 s, 8.3 frames/s without overlays, with the detector now on every frame.

## Held-out scorecard

One run, `data/runs/plan2c-heldout` (`run.json`: mode `transform`, `bridge_gap_s` 2.0, `min_segment_s` 1.0, weights `data/models/rink-keypoints/r2/best.pt`, tracker settings baseline), five `ok`. From its `scorecard.md`; the last two columns are the same clips in Plan 2a (`docs/baseline-results.md`) and Plan 2b-ii (`docs/rink-results.md`).

Tracking (IDF1 at IoU 0.5):

| Clip | IDF1 | IDF1@0.3 | Precision | Recall | Ground-truth boxes | False boxes | Missed boxes | Plan 2a | Plan 2b-ii |
|---|---|---|---|---|---|---|---|---|---|
| **Overall** | 69.8% | 76.0% | 87.1% | 79.8% | 202273 | 23942 | 40838 | 70.6% | 62.9% |
| CAR_VS_BOS_001 | 59.4% | 73.5% | 71.9% | 66.4% | 53436 | 13874 | 17933 | 59.4% | 57.5% |
| CAR_VS_BOS_002 | 86.0% | 86.7% | 97.5% | 90.8% | 22814 | 536 | 2103 | 86.0% | 73.6% |
| CAR_VS_NYR_001 | 68.1% | 68.7% | 98.9% | 85.5% | 45854 | 426 | 6640 | 71.4% | 56.7% |
| CAR_VS_NYR_002 | 81.4% | 81.9% | 97.4% | 91.1% | 37919 | 939 | 3360 | 81.4% | 75.6% |
| CAR_VS_NYR_003 | 65.7% | 75.5% | 79.4% | 74.4% | 42250 | 8167 | 10802 | 65.7% | 59.3% |

Rink transform:

| Clip | Median | Valid share | Points | Play frames | Labelled | Scored | No transform | Non-play | Meets bar |
|---|---|---|---|---|---|---|---|---|---|
| **Overall** | 0.14 m | 97.8% | 360 | 23479 | 50 | 48 | 2 | 0 | yes |
| CAR_VS_BOS_001 | 0.16 m | 98.8% | 85 | 6517 | 10 | 10 | 0 | 0 | yes |
| CAR_VS_BOS_002 | 0.15 m | 98.5% | 75 | 2644 | 10 | 10 | 0 | 0 | yes |
| CAR_VS_NYR_001 | 0.13 m | 96.7% | 63 | 5156 | 10 | 10 | 0 | 0 | yes |
| CAR_VS_NYR_002 | 0.12 m | 99.1% | 76 | 4283 | 10 | 10 | 0 | 0 | yes |
| CAR_VS_NYR_003 | 0.16 m | 96.0% | 61 | 4879 | 10 | 8 | 2 | 0 | yes |

Bars: IDF1 ≥ 70.6% — **not met (69.8%)**, 0.8 points short; rink median ≤ 1.0 m — met (0.14 m); valid share ≥ 90% — met (97.8%), every clip on its own. Play share per held-out clip (`frames.parquet`): CAR_VS_BOS_001 100%, CAR_VS_BOS_002 100%, CAR_VS_NYR_001 96.5%, CAR_VS_NYR_002 100%, CAR_VS_NYR_003 100%; 23,479 of 23,666 frames (99.2%) are play. Labelled evaluation frames called non-play: 0 (was 4 in Plan 2b-ii); labelled frames without a transform: 2, both in CAR_VS_NYR_003 (one of them frame 742, the five-point frame Plan 2b-ii also could not fit).

Overlay check: `CAR_VS_BOS_002/overlay.mp4` at frame 1317 (22 s, a frame Plan 2b-ii's filter dropped; `data/tmp/p2c-overlay-BOS2-1317.jpg`) shows boxes on the skaters and the rink view's dots along the far boards and in the neutral zone where the camera shows them.

## Reading

- **Where the points came back.** Four of the five clips score exactly Plan 2a's IDF1, precision and recall, with the same number of track ids (`tracks.parquet`: 178, 68, 112 and 143 ids against Plan 2a's 178, 68, 112, 143; Plan 2b-ii had 184, 70, 119, 189). On those clips every frame is play and the one tracker per clip reproduces Plan 2a's tracking to the box. Recall is back from 74.5% to 79.8% (Plan 2a 80.6%) and identity from the per-segment restarts is gone.
- **The 0.8 points still missing** are all in CAR_VS_NYR_001: 68.1% against Plan 2a's 71.4%, 130 track ids against 125, play share 96.5%. Its `frames.parquet` has two non-play stretches, frames 0–56 (0.9 s, a leading stretch, which bridging never fills by design) and 4299–4428 (2.2 s, just over the 2.0 s bridge). Both are live play from the same end-zone view (`data/tmp/p2c-heldout-gap-30.jpg`, `-4363.jpg`): five to seven confident points per frame (goal-line boards, the end dot and the circle points), every `rink_fits` row `invalid`. It is the few-point failure described above, not a filter rule failure, and across the skipped 2.2 s the camera moves, so boxes after the gap no longer overlap their tracks from before it and take new ids, which is where the identity goes.
- **The filter now drops 0.8% of these clips** (187 of 23,666 frames), against 8.0% in Plan 2b-ii, and none of the 50 labelled evaluation frames. Everything it still drops here is play; its ability to reject replays and bench shots is untested on this dataset, which has none, and is the reason the filter is kept.
- **Rink transform.** Median 0.14 m against 0.15 m in Plan 2b-ii; valid share 97.8% against 98.3%, the small drop being bridged frames that have no fit of their own and sit more than 0.5 s (`interpolate_gap_s`) from one, so they stay `invalid` and their positions `missing`. The two labelled frames without a transform are five-point frames the fit cannot settle.
- **Speed.** `run.json` `started_at` 03:34:02 UTC to the last overlay at 04:23:54 UTC: 23,666 frames in 49 min 52 s, 7.9 frames/s with overlays (Plan 2b-ii 8.2 with the detector on 92% of the frames; Plan 1 13.6 with no rink stages). The training run without overlays ran at 8.3 frames/s.

## Decision

The filter no longer costs the tracking its ground: IDF1 on the held-out clips is 69.8% against Plan 2a's 70.6%, four clips are identical to Plan 2a, both rink bars are met on every clip, and the filter keeps 99.2% of the frames while remaining in the pipeline for real broadcast footage. The 0.8-point shortfall against the bar set in the spec is one 2.2 s stretch of live play in CAR_VS_NYR_001 whose five-point frames the transform cannot fit at the 5 px RANSAC threshold; it is not something the filter's two settings can fix without bridging beyond 2 s, and it was not tuned away on the held-out clips.

The bottleneck the tables now show is the few-point fit: zoomed end-zone views where the model finds four to six points whose positions disagree by 7 to 30 px. Recommendation: proceed to Plan 3 (teams and identity) on these positions and tracks, and give Plan 3's first task, or a short follow-up, the few-point fit (a looser RANSAC threshold when fewer than six points are found, or sub-cell peak refinement in the keypoint model) with `refilter` as the test bench; the identity work itself is where the remaining 15 points to the 85% bar sit (`docs/baseline-results.md`: with perfect identity these clips would score about 84%).
