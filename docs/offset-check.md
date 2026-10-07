# Ground-truth offset: the NTSC check

Checked on: 2026-10-07. Inputs: `data/runs/diagnose/heldout/ntsc.md`, `data/runs/diagnose/train/ntsc.md` (from `tracker diagnose-boxes --shift` on `data/runs/plan2a-heldout` and `data/runs/plan2a-train`), `data/raw/VIP-HTD/mot-challenge-format/*/*/seqinfo.ini`.

## The hypothesis
Decision point 1b hypothesised that the ground-truth/image offset is mechanical: 60 fps indexing against a 59.94 fps clock, which gives a drift of one frame per 1001 frames. The test: for each fifth of a clip, the best shift k must lie within 0.75 of mid-frame / 1001 and must not decrease along the clip.

## Result
Held-out clips (`data/runs/diagnose/heldout/ntsc.md`):

Run: `data/runs/plan2a-heldout`. Predicted k at a range's mid-frame is mid / 1001; a clip fits when every |best k - predicted| <= 0.75 and best k never decreases. 'no drift' means best k is 0 in every fifth.

| Clip | frames | fps | best k per fifth | predicted k per fifth | max abs error | verdict |
|---|---|---|---|---|---|---|
| CAR_VS_BOS_001 | 6517 | 60 | +1, +2, +4, +5, +6 | 0.7, 2.0, 3.3, 4.6, 5.9 | 0.75 | fits |
| CAR_VS_BOS_002 | 2644 | 60 | +0, +0, +0, +0, +0 | 0.3, 0.8, 1.3, 1.8, 2.4 | 2.38 | no drift |
| CAR_VS_NYR_001 | 5343 | 60 | +0, +0, +0, +0, +0 | 0.5, 1.6, 2.7, 3.7, 4.8 | 4.80 | no drift |
| CAR_VS_NYR_002 | 4283 | 60 | -1, +0, +0, +0, +0 | 0.4, 1.3, 2.1, 3.0, 3.9 | 3.85 | drifts, does not fit |
| CAR_VS_NYR_003 | 4879 | 60 | +1, +2, +3, +4, +5 | 0.5, 1.5, 2.4, 3.4, 4.4 | 0.61 | fits |

Train clips (`data/runs/diagnose/train/ntsc.md`; the 6 clips under "Not diagnosed" in `data/runs/diagnose/train/summary.md` are not scorable and have no row):

Run: `data/runs/plan2a-train`. Predicted k at a range's mid-frame is mid / 1001; a clip fits when every |best k - predicted| <= 0.75 and best k never decreases. 'no drift' means best k is 0 in every fifth.

| Clip | frames | fps | best k per fifth | predicted k per fifth | max abs error | verdict |
|---|---|---|---|---|---|---|
| CGY_VS_DAL_001 | 3969 | 60 | +0, +0, -1, +0, +0 | 0.4, 1.2, 2.0, 2.8, 3.6 | 3.57 | drifts, does not fit |
| CGY_VS_DAL_002 | 4169 | 60 | +0, +0, +0, +0, +0 | 0.4, 1.2, 2.1, 2.9, 3.7 | 3.75 | no drift |
| CGY_VS_DAL_003 | 3326 | 60 | +2, +2, +2, +4, +5 | 0.3, 1.0, 1.7, 2.3, 3.0 | 2.01 | drifts, does not fit |
| CHI_VS_TOR_003 | 1067 | 30 | +0, +0, +0, +0, +0 | 0.1, 0.3, 0.5, 0.7, 1.0 | 0.96 | no drift |
| CHI_VS_TOR_004 | 1618 | 30 | +1, +1, +1, +2, +2 | 0.2, 0.5, 0.8, 1.1, 1.5 | 0.87 | drifts, does not fit |
| CNTRL_VS_PAC_002 | 1965 | 30 | +0, +0, +0, +0, +0 | 0.2, 0.6, 1.0, 1.4, 1.8 | 1.77 | no drift |
| PIT_VS_SJ_001 | 1391 | 30 | +1, +1, +1, +2, +2 | 0.1, 0.4, 0.7, 1.0, 1.3 | 1.03 | drifts, does not fit |
| PIT_VS_SJ_002 | 2762 | 30 | +0, +0, +0, +0, +0 | 0.3, 0.8, 1.4, 1.9, 2.5 | 2.48 | no drift |
| PIT_VS_SJ_003 | 3525 | 30 | +0, +0, +0, +0, +0 | 0.4, 1.1, 1.8, 2.5, 3.2 | 3.17 | no drift |
| PIT_VS_WAS_001 | 2981 | 60 | +1, +1, +1, +3, +3 | 0.3, 0.9, 1.5, 2.1, 2.7 | 0.92 | drifts, does not fit |
| STL_VS_SJ_003 | 6027 | 60 | +1, +2, +4, +5, +6 | 0.6, 1.8, 3.0, 4.2, 5.4 | 0.99 | drifts, does not fit |

## Reading
- Clips that fit: CAR_VS_BOS_001 (max |error| 0.75) and CAR_VS_NYR_003 (0.61), both held-out, both 60 fps (`heldout/ntsc.md`). CAR_VS_BOS_001 sits exactly at the 0.75 limit. No train clip fits.
- Clips with no drift: CAR_VS_BOS_002, CAR_VS_NYR_001 (`heldout/ntsc.md`); CGY_VS_DAL_002, CHI_VS_TOR_003, CNTRL_VS_PAC_002, PIT_VS_SJ_002, PIT_VS_SJ_003 (`train/ntsc.md`). All 7 have frames >= 1001 (1067 to 5343), so the hypothesis predicts at least one frame of drift in each, and they show none. These contradict a dataset-wide mechanism.
- Clips that drift but do not fit (7). Counting the 2 that fit, 9 clips carry a label other than `no drift` in these tables, but 2 of the 9 (CAR_VS_NYR_002 and CGY_VS_DAL_001) get it only from a single fifth at k = -1 with every other fifth 0; the other 7 grow from +1 or +2 to +2 to +6. docs/decision-point-1b.md counted 7 of 16 clips with a growing offset: those are the 7 positive-growth clips here (CAR_VS_BOS_001, CAR_VS_NYR_003, STL_VS_SJ_003, CGY_VS_DAL_003, PIT_VS_WAS_001, CHI_VS_TOR_004, PIT_VS_SJ_001). The label here counts any non-zero fifth, which is why it gives 9.
  - CAR_VS_NYR_002: best k -1, 0, 0, 0, 0 (max |error| 3.85).
  - CGY_VS_DAL_001: 0, 0, -1, 0, 0, so k decreases (3.57).
  - CGY_VS_DAL_003: starts at +2 and holds +2, +2, +2, then +4, +5 (2.01).
  - CHI_VS_TOR_004: +1, +1, +1, +2, +2 against predicted 0.2, 0.5, 0.8, 1.1, 1.5: k is at or above the prediction in every fifth, starting offset (+1 against 0.2) and jumping ahead in the fourth fifth (0.87).
  - PIT_VS_SJ_001: +1, +1, +1, +2, +2 (1.03).
  - PIT_VS_WAS_001: +1, +1, +1, +3, +3 (0.92).
  - STL_VS_SJ_003: +1, +2, +4, +5, +6, the same k sequence as CAR_VS_BOS_001 but the opposite verdict (max |error| 0.99 against 0.75): it has 6027 frames against 6517, so its predictions are lower (5.4 against 5.9 in the last fifth). This shows how sensitive the 0.75 limit is.
- Relation to frame rate, split folder or game: both frame rates occur among drifting clips (60 fps: 7 of 9, namely CAR_VS_BOS_001, CAR_VS_NYR_002, CAR_VS_NYR_003, CGY_VS_DAL_001, CGY_VS_DAL_003, PIT_VS_WAS_001, STL_VS_SJ_003; 30 fps: CHI_VS_TOR_004, PIT_VS_SJ_001) and among no-drift clips (60 fps: CAR_VS_BOS_002, CAR_VS_NYR_001, CGY_VS_DAL_002; 30 fps: CHI_VS_TOR_003, CNTRL_VS_PAC_002, PIT_VS_SJ_002, PIT_VS_SJ_003). Drifting clips sit in all three seqinfo split folders (test, train, validation). Within a game the clips differ (CAR_VS_BOS_001 drifts, CAR_VS_BOS_002 does not; CGY_VS_DAL_001 and _003 drift, _002 does not; PIT_VS_SJ_001 drifts, _002 and _003 do not). No relation is visible.

## Verdict
(b). The mechanical 1000/1001 drift explains 2 of the 9 drifting clips (CAR_VS_BOS_001, CAR_VS_NYR_003) and 7 clips long enough to show it do not drift at all, so it is not a dataset-wide mechanism. The other 7 drifting clips either start offset, decrease, or grow at a different rate. The cause looks per-clip (extraction), not a fixed clock ratio.

## What Plan 2b-ii should do
Do not build the global loader correction `frame_gt = frame - round(frame / 1001)`. A per-clip offset table would be needed to correct the 7 clips with a growing offset (the 9 labelled drifting include 2 that are a single fifth at k = -1, which a table of growing offsets would not cover), and the owner decides whether to build one or to exclude those clips from scoring. Ground truth is not hand-edited either way. CAR_VS_BOS_001 and CAR_VS_NYR_003 stay in the held-out set whatever is decided.
