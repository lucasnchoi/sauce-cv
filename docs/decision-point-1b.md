# Decision point 1b: what limits tracking, and what to do about it

Date: 2026-10-07

Sources: `docs/baseline-results.md`, `docs/investigation-box-offset.md`, `docs/investigation-tracker-settings.md`. Held-out numbers (five 60 fps clips, never tuned on) and training numbers (11 scorable clips, tuning side) are kept apart below and labelled each time.

## Where the held-out IDF1 of 70.6% goes

On the held-out set, precision is 87.2% and recall 80.6% (`baseline-results.md`), which caps IDF1 at about 83.8% even with perfect identities. That is itself under the 85% bar, so detection alone costs about 1 point against the bar and identity a further 13 (`baseline-results.md`, "Reading of the numbers"). It differs by clip: CAR_VS_BOS_001 and CAR_VS_NYR_003 lose mostly in detection, the other three mostly in identity. Investigation 1 adds that on CAR_VS_BOS_001 much of the apparent detection loss is boxes that are close to a ground-truth box but not close enough: IDF1 there rises from 59.4% at IoU 0.5 to 73.5% at IoU 0.3 (`investigation-box-offset.md`). The note's explanation is a timing offset between ground truth and picture that grows from about 1 frame at the start to about 6 at the end of the clip. It tested this on four clips only (CAR_VS_BOS_001 held-out; STL_VS_SJ_003, CHI_VS_TOR_004, CHI_VS_TOR_003 training) and cannot say whether the cause is the annotation or the frame extraction. Part of the gap to the bar is therefore a measurement problem and not tracking, but the notes do not size that part for the whole held-out set.

## Evidence in one table

| Question | Answer | Source |
|---|---|---|
| Are ByteTrack's boxes the detector's boxes? | Yes. `tracks==dets` is 100% on all 16 diagnosed clips (5 held-out, 11 training), so the tracker adds no lag or smoothing to the boxes. | investigation-box-offset.md |
| Is CAR_VS_BOS_001's offset also seen on training clips? | Yes, in proportion to how poorly they score: STL_VS_SJ_003 (60.0%) drifts 1 to 6 frames, CHI_VS_TOR_004 (60.2%) 1 to 2 frames, CHI_VS_TOR_003 (93.1%) none. Only these three training clips were tested; the other 8 scorable ones were not. | investigation-box-offset.md |
| Does IDF1 at 0.4 differ much from 0.5 on the poor clips? | Yes. Held-out CAR_VS_BOS_001 59.4% to 67.9% (at 0.3: 73.5%). Training STL_VS_SJ_003 60.0% to 66.6% (at 0.3: 71.5%), CHI_VS_TOR_004 60.2% to 64.0% (at 0.3: 66.3%); the good clip CHI_VS_TOR_003 barely moves (93.1% to 94.4%). | investigation-box-offset.md |
| Best ByteTrack setting on training clips | `high_conf_det_threshold=0.7`: IDF1 71.1% vs 70.5% baseline (+0.6; below the plan's 1.0-point margin, so the default is unchanged). 12 one-factor points and a 6-point cross span 68.5% to 71.1%. | investigation-tracker-settings.md |
| Its single held-out measurement | None taken. No setting beat the baseline by the margin, so no held-out clip was scored. Held-out stays at the baseline's 70.6%. | investigation-tracker-settings.md, baseline-results.md |

## Options

All three stay inside the decision-point-1 rule: no Ultralytics, no new external data.

1. **Fine-tune the detector on HockeyAI + VIP-HTD boxes** (new small plan).
   - Addresses: missed boxes, the largest component on the training clips (44,491 missed of 260,768 ground-truth boxes, `investigation-tracker-settings.md`), though some of the missed boxes on the poor clips are displaced boxes that fall just under IoU 0.5, not undetected players (`investigation-box-offset.md`, `baseline-results.md`); recall is about 89% to 91% even on the cleanest held-out clips (`baseline-results.md`). Any later tracker would also use better boxes.
   - Does not address: identity loss, which is the larger cost on three of five held-out clips (8 to 22 points; `baseline-results.md`). Investigation 1 found no reason to change the detector for box shape (worth only 1 to 2 points at IoU 0.5 on training clips).
   - Cost and risk: HockeyAI is external data, which decision point 1 ruled out, so this option as worded needs the owner to reverse that; VIP-HTD training clips alone would stay internal. The notes did not measure how many missed boxes the detector never produced. My own inference, not a finding in the notes: fine-tuning on and scoring against labels with the time offset could mix the lag into what the detector learns; Investigation 1 recommends not rewriting ground truth until the offset's source is known.
2. **Try a different tracker** (for example an appearance-aware one, if a permissively licensed implementation exists).
   - Addresses: identity loss, ByteTrack having no appearance cue. Investigation 2 says its grid cannot test this, since no setting adds one. Longer memory (`lost_track_buffer` 60 and 90) did nothing, so more of the same motion-only memory does not help.
   - Does not address: missed or false boxes (the tracker passes the detector's boxes through unchanged) or the offset.
   - Cost and risk: I did not check whether a suitable permissively licensed implementation exists, and an appearance model may need weights from outside the project. Investigation 2 itself ranks this as a second step after detection.
3. **Neither now; proceed with rink mapping (Plan 2b).**
   - What is lost by waiting: the held-out IDF1 stays at 70.6% (14.4 points under the bar), and rink mapping would inherit the baseline's boxes (held-out: 23,950 false and 39,269 missed of 202,273 ground-truth boxes, `baseline-results.md`) and its identity errors. The 85% bar stays unmet and unjudged.
   - What is kept: nothing in the notes shows a cheap setting or tracker change that closes the gap, and the offset problem can be settled in parallel.

## Recommendation

Proceed with rink mapping (option 3), because on the training clips the best ByteTrack setting gained only 0.6 IDF1 points against a gap to the bar of about 14 points. On the same training clips, relaxing the overlap rule from IoU 0.5 to 0.3 alone lifts STL_VS_SJ_003 by 11.5 points (60.0% to 71.5%), so a tracker change or detector fine-tune cannot be judged against these scores until the ground-truth offset (see below) is understood. (The held-out CAR_VS_BOS_001 shows the same pattern, 59.4% to 73.5%, but it is description only and not the reason.)

## Open items carried forward

- **Ground-truth time offset (Investigation 1; none of the three options addresses it).** On CAR_VS_BOS_001 (held-out), STL_VS_SJ_003 and CHI_VS_TOR_004, ground truth matches the picture best when read 1 to 6 frames earlier, growing along the clip. Options: re-align the affected clips, or exclude them from scoring; the owner's call. First step per the note: run the shift test on the remaining training clips to learn how common it is (CAR_VS_NYR_003 at 20.6% of boxes under IoU 0.5, PIT_VS_SJ_001 at 17.5% and PIT_VS_WAS_001 at 12.2% are untested candidates) and whether it comes from the annotation or the frame extraction. Not known: CAR_VS_BOS_001's source video has 6,551 frames against 6,517 in the dataset, but CHI_VS_TOR_003 also has a longer video and shows no drift.
- **Extra scorecard columns.** Report IDF1 at 0.3 and 0.4 alongside the spec's 0.5 (addition, not replacement), since the gap between them marks the offset (`investigation-box-offset.md`, recommendation 3).
- **Six training clips with duplicate ground-truth ids** (CHI_VS_TOR_002, CNTRL_VS_PAC_001, CNTRL_VS_PAC_005, CNTRL_VS_PAC_006, STL_VS_SJ_001, STL_VS_SJ_002) stay excluded and the loader strict; the policy is the owner's (`baseline-results.md`, question 3).
- **No 30 fps held-out clips.** All five held-out clips are 60 fps, so held-out scores say nothing about 30 fps footage (`baseline-results.md`).
- **False-box types are untallied.** No count of false boxes by referee, bench or crowd exists (`baseline-results.md`).
- **Plan 2b carry-overs** from the Plan 2a design (duplicate clip-id guard, zero-frame video error, interrupted-run test, polars errors in `score_run`, scorecard per stage) were deliberately left to Plan 2b.
