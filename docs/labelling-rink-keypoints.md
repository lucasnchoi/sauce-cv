# Labelling rink keypoints

What you label, how, and when to stop.

## What
33 rink keypoints per frame (the list is `tracker.rink.KEYPOINTS`; the legend window shows them). Click where the point is on the ice surface: the centre of a line, the centre of a dot, the extreme of a circle, the base of a goal post. If a point is hidden by a player, estimate it only if the surrounding lines make its position certain; otherwise mark it not visible. If it is off-screen, mark it not visible. A frame with nothing visible is fine: mark every point not visible and move on — it is recorded.

Left is the camera's left. "far" is the far side of the rink from the camera, "near" the side nearest the camera. A circle's "top" is its far-side extreme.

## Frames
`labels/rink_frames.json`: 408 training frames (24 per training clip) and 50 evaluation frames (10 per held-out clip), chosen once with seed 20261007. Never regenerate it. Training frames train the model; evaluation frames only score the rink transform.

## Run
```
uv run tracker label-rink --dataset data/raw/VIP-HTD --frames labels/rink_frames.json --labels labels/rink_keypoints.parquet
```
Add `--subset evaluation` for the held-out frames (do those after the 80-frame stop, below). Add `--start N` to resume at frame position N, or press `g` to jump to the next unlabelled frame. Add `--prelabels data/runs/<run>` once a model has written `keypoints.parquet` per clip (Plan 2b-ii): its points appear as suggestions you correct.

Keys: left click place; right click mark not visible; `]` / `[` next / previous keypoint; `n` / `p` next / previous frame; `g` next unlabelled frame; `u` undo; `c` clear the current point; `s` save; `q` save and quit. Saving validates the file (visible points need pixels inside the image; 33 rows per frame) and prints what it saved; if it refuses, it says which point.

## The 80-frame stop
Label about 80 training frames (press `q`), then stop and say so. Plan 2b-ii trains the first model on them, pre-labels the rest, and you correct instead of placing. Judge the pace at the stop: 80 frames at 10–15 visible points each is roughly an hour.

## Commit
`labels/rink_keypoints.parquet` is committed after each session (`git add labels/rink_keypoints.parquet`). The frames are referenced, never copied.
