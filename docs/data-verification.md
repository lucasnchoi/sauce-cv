# Dataset verification

Checked on: 2026-10-06

Disk check (Task 2 step 1), run in the project root:

```
1:data/
/dev/disk3s5   460Gi   368Gi    64Gi    86%    3.3M  675M    0%   /System/Volumes/Data
```

How to read this document: "Observed" means read from the downloaded files (or from files fetched by link, as stated). "Stated" means the dataset's own README or card says so and it was not checked against the files. Downloaded files were treated as untrusted data: nothing from them was executed, nothing was unpickled, no model weights were loaded.

## VIP-HTD
Source: https://github.com/harshap-ai/VIP-HTD (shallow clone, about 27 GB on disk: 13 GB per annotation format plus 811 MB of clips). Every number below was computed from the files by scripts under `data/tmp/` (Step 3 output is in the task report). The two annotation formats were both checked.

- Licence file says: "GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007" (`LICENSE`, the standard GPL-3.0 text). The README (stated) says the dataset inherits the GPL-3.0 licence of the MHPTD dataset it was derived from.
- Annotated sequences (mot-challenge-format): 22 in total: 14 train, 7 test, 1 validation (`PIT_VS_WAS_001`). All 22 have images and annotations. The brief expected 13 train; there are 14.
- Annotated sequences (personnel-level-format): 22 in total: 14 train, 7 test, 1 validation, the same sequence names and the same split as the other format. Image counts, frame rates and ground-truth row counts per sequence are identical in the two formats (observed); the ids in the `gt.txt` files differ (for example `CAR_VS_BOS_001` has 159 ids in one format and 103 in the other, which fits the README: personnel ids survive a player leaving and re-entering view). The two formats each hold their own full copy of the frames (`img1`, 13 GB each); a `cmp` of `CAR_VS_BOS_002/img1/000001.jpg` in the two formats showed identical bytes; no other frame was compared.
- Clips in `clips/`: 22 `.mp4` files in 8 game folders (stated by the README: "All 22 clips").
- Frame rates seen (from `seqinfo.ini`): 30 and 60. 60 fps: 12 sequences (CAR_VS_BOS x2, CAR_VS_NYR x3, CGY_VS_DAL x3, STL_VS_SJ x3, PIT_VS_WAS x1). 30 fps: 10 sequences (CHI_VS_TOR x3, CNTRL_VS_PAC x4, PIT_VS_SJ x3). The rate is constant within a game.
- Frame sizes seen: 1280x720 only, in all 44 sequence folders.
- Sequences where image count differs from seqLength: none (all 22 match in both formats).
- Sequences where the seqinfo name differs from the folder name: 16 of 22, the same in both formats. Folder name -> seqinfo `name`: CAR_VS_BOS_001 -> CAR_VS_BOS_2019_001, CAR_VS_BOS_002 -> CAR_VS_BOS_2019_002; CHI_VS_TOR_002, _003, _004 -> CHI_VS_TOR_2016_002, _003, _004; CNTRL_VS_PAC_001, _002, _005, _006 -> allstar_2019_001, _002, _005, _006; PIT_VS_SJ_001, _002, _003 -> PIT_VS_SJ_2019_001, _002, _003; STL_VS_SJ_001, _002, _003 -> STL_VS_SAJ_2019_001, _002, _003; PIT_VS_WAS_001 -> PIT_VS_WAS_2016_001. The six that match: CAR_VS_NYR_001, _002, _003 and CGY_VS_DAL_001, _002, _003. Also the `clips/` folder for the all-star game is `CNTRL_VS_PAC_2019`, not `CNTRL_VS_PAC`. Folder names are used as clip ids.
- Annotation columns confirmed as frame,id,x,y,w,h,conf,class,vis: yes as to shape. All 585,763 rows in all 44 `gt.txt` files (22 per format; the two formats have the same row count) have exactly 9 comma-separated fields; no header row, no blank lines, no non-numeric first field. The column names are the MOTChallenge convention; the dataset does not state them in a file. All of frame, id, x, y, w, h are integers in every row (no decimals). Frames are 1-based (frame 1 to seqLength in every file, image files `000001.jpg` upward).
- Distinct values in the class column across all files: one value, `1` (585,763 rows in each format; the same in every sequence). The `conf` column is also only `1` (585,763 rows per format) and the `vis` column only `1` (585,763 rows per format). So the file carries no confidence, class or visibility information.
- Other checks on the ground truth (both formats):
  - Zero or negative width or height: none. Boxes partly or wholly outside the 1280x720 frame, or with a negative x or y: none.
  - Duplicate (frame, id) pairs within one file: yes. mot-challenge-format: 758 repeated pairs (766 surplus rows) in 6 sequences (CNTRL_VS_PAC_001 97, STL_VS_SJ_001 56, CHI_VS_TOR_002 66, CNTRL_VS_PAC_005 93, CNTRL_VS_PAC_006 242, STL_VS_SJ_002 204). personnel-level-format: 678 repeated pairs (686 surplus rows) in the same 6 sequences (CNTRL_VS_PAC_005 has 13 there). Most pairs occur twice, in CHI_VS_TOR_002 up to three times. None of the duplicate rows are identical to each other: they are different boxes under the same id in the same frame (example: frame 1279, id 27 in `CNTRL_VS_PAC_001`, one box at x=1242 and one at x=96, far apart). Any code that assumes one box per id per frame will misbehave on these 6 sequences.
- Are referees and goalies annotated, and are they distinguishable from players? Annotated: yes. Distinguishable by the files: no. Looked at `mot-challenge-format/train/CAR_VS_BOS_002` (first train folder in sorted order), frames 1 and 300, with the ground-truth boxes drawn and labelled with id and class (images `data/tmp/gt-check/CAR_VS_BOS_002_000001.jpg` and `..._000300.jpg`). In both frames the goaltender in the left net has a box (id 22) and the striped referee at the faceoff has a box (his id label is partly hidden under neighbouring boxes), both labelled class 1 like the skaters. The class column holds only 1, so one class covers players, goalies and referees together. Telling them apart would need the pipeline's own model, for example a referee and goaltender detector (HockeyAI has those classes).
- Games (sequence name without its trailing number), clips and sequences by split: CAR_VS_BOS 2 clips (001 test, 002 train); CAR_VS_NYR 3 (001 test, 002 and 003 train); CGY_VS_DAL 3 (001 and 002 train, 003 test); CHI_VS_TOR 3 (002 and 003 train, 004 test); CNTRL_VS_PAC 4 (001 test, 002, 005 and 006 train; the clips folder is named `CNTRL_VS_PAC_2019`); PIT_VS_SJ 3 (001 test, 002 and 003 train); PIT_VS_WAS 1 (001 validation); STL_VS_SJ 3 (001 test, 002 and 003 train). That is 8 games and 22 clips. Whether two clips of one game are close in time is not stated. The clip numbers skip some values (CHI_VS_TOR has no 001, CNTRL_VS_PAC no 003 or 004), which suggests clips were dropped from the original dataset; not stated.

## HockeyRink
Source: `SimulaMet-HOST/HockeyRink` on Hugging Face. This is a model repository, not a dataset. Its README says the model was trained on a separate "HockeyRink dataset" (https://huggingface.co/datasets/SimulaMet-HOST/HockeyRink); that dataset was not downloaded or checked.

- Licence: MIT. Stated only: the README front matter reads `license: mit`. There is no LICENSE file in the repository. Observed: the repository has exactly three files.
- Files (observed):
  - `HockeyRink.pt`, 93,309,985 bytes (about 89 MiB)
  - `README.md`, 2,983 bytes
  - `.gitattributes`, 1,519 bytes
  - The `.pt` file is a PyTorch zip archive. Listing its entries (`python -m zipfile -l`, nothing more) shows a top-level folder `best/` holding `data.pkl` (263,069 bytes), `byteorder`, and about 640 numbered tensor files under `best/data/`. `data.pkl` is a pickle; it was not opened or loaded, so the model's class names, keypoint names and training settings stored inside it are unchecked.
- What library the weights need to run: Ultralytics (YOLOv8 pose). Stated: the README says "This repository contains a YOLOv8-based model" and "Architecture: YOLOv8-Large pose estimation". Not confirmed from inside the weights (the pickle was not read). Ultralytics is AGPL-3.0: PyPI lists `license: AGPL-3.0` and the classifier "GNU Affero General Public License v3 or later (AGPLv3+)" for version 8.4.174 (https://pypi.org/pypi/ultralytics/json, fetched 2026-10-06).
- Number and names of keypoints: 56 keypoints. Stated only: the README says "Accurate detection of 56 keypoint landmarks" and "Output: 56 keypoint coordinates with confidence scores". Not checked: the repository has no config or data YAML, so the keypoint names and order are not defined in any file that was downloaded. They are most likely stored inside the pickled model and in the separate HockeyRink dataset repository. A later plan must get the list from that dataset repository (not downloaded) or from the Hugging Face Space, because the names are needed to build the rink transform.
- Footage it was trained on: stated, "Trained on diverse SHL (Swedish Hockey League) game footage". Other README claims, not checked: mAP@0.5 97.48%, mAP@0.5:0.95 76.45%, 13.64 FPS on a Tesla T4, 6.4 FPS on an M3 MacBook Pro.

## HockeyAI
Source: `SimulaMet-HOST/HockeyAI`, a Hugging Face dataset repository.

- Licence: MIT. Stated only: `license: mit` in the README front matter and `"licenses": ["mit"]` in `dataset-metadata.json`. There is no LICENSE file.
- Files (observed): `HockeyAI_Dataset.zip` (403,951,504 bytes), `data/train-00000-of-00001.parquet` (363,401,335 bytes), `README.md`, `dataset-info.json`, `dataset-metadata.json`, `.gitattributes`.
- Image count and resolution:
  - Observed: the zip holds 2,101 JPEG frames in `SHL/frames/` and 2,101 text files in `SHL/annotations/`, with matching names (no frame without a label file, no label file without a frame). All 2,101 frames are 1920x1080 (checked by decoding each image's header).
  - Observed: the parquet file holds 1,890 rows and a single column `image` (struct of bytes and path). It has no labels. The README front matter says `num_examples: 1890`, while the README body says "about 2,100 frames". So the parquet is a smaller, label-less subset of the zip (not checked which frames); the zip is the only source of annotations.
  - The README's stated folder layout (`HockeyAI/frames`, `HockeyAI/annotations`) does not match the zip, which uses `SHL/frames` and `SHL/annotations`. `dataset-info.json` lists splits `train`, `valid`, `test`, but the zip contains no split folders or split lists. No train/validation/test split exists in the files.
- Label format and class list:
  - Observed: YOLO text format, one box per line, `<class_id> <x_center> <y_center> <width> <height>`, five fields on every line (0 malformed lines), no empty label files.
  - Stated: `dataset-info.json` lists the classes in this order: 0 centerIce, 1 faceoff, 2 goal, 3 goaltender, 4 player, 5 puck, 6 referee. Observed: class ids in the label files run 0 to 6 only. The id-to-name mapping is taken from that list in `dataset-info.json`; no classes file exists inside the zip. Sizes look consistent with it (class 5 boxes are tiny, class 6 boxes are tall and thin).
  - Observed box counts by class id: 0 centerIce 591; 1 faceoff 3,460; 2 goal 1,219; 3 goaltender 1,281; 4 player 13,660; 5 puck 2,103; 6 referee 1,948.
  - Observed images containing each class: centerIce 584; faceoff 1,889; goal 1,198; goaltender 1,266; player 2,095; puck 2,101; referee 1,475.
- Number of puck boxes: 2,103 (class id 5) across 2,101 images. Every image has at least one puck box; two images have two. Mean puck box size is about 21 x 17 pixels at 1920x1080.
- Footage: stated, broadcast Swedish Hockey League games. The folder is named `SHL`. Camera behaviour and the number of games are not stated.

## Big Data Cup 2026
Source: https://github.com/bigdatacup/Big-Data-Cup-2026 (cloned, shallow, 3 files, about 660 KB).

- Licence or terms of use: no LICENSE file. The repository's `legal.md` is titled "Stathletes Public Data User Agreement". The relevant sentence (stated, quoted exactly): "Any research that is produced from this dataset is free to be shared publicly, provided that said research is not used for profit." It also says "Stathletes retains the right to publish any work submitted for the Hackathon." and "By using the Service the User will be bound by all terms making up this Agreement." This is a non-commercial restriction, not an open-source licence. The agreement text does not say anything about redistributing the data itself or about code that merely writes the same column layout.
- Files:
  - In the repository (observed): `README.md` (10,552 bytes), `legal.md` (858 bytes), `rink_coords.png` (275,314 bytes). The tracking and event data are NOT in the repository.
  - The README points to https://github.com/bigdatacup/Big-Data-Cup-2026/releases/tag/Data for the data. Observed through the public GitHub API (no login needed): that release, published 2026-01-05, has 54 assets. Per game it has an Events csv, a Shifts csv and Tracking csv files per period (`Tracking_P1`, `_P2`, `_P3`, and `_POT` for the three games that went to overtime), plus one `camera_orientations.csv`. There are 10 games, dated 2025-10-11 to 2025-11-28, with anonymised team names (Team A to Team L). Each tracking file is about 1 to 26 MB.
  - Sample fetched for this check, from the release assets (public, no login, form or terms acceptance), saved in `data/raw/Big-Data-Cup-2026-sample/` (about 26.6 MB in total; the files keep the literal `%40` in their names because `curl -O` saved the URL's name):
    - https://github.com/bigdatacup/Big-Data-Cup-2026/releases/download/Data/2025-10-11.Team.A.%40.Team.D.Tracking_P1.csv (26,455,697 bytes)
    - https://github.com/bigdatacup/Big-Data-Cup-2026/releases/download/Data/2025-10-11.Team.A.%40.Team.D.Events.csv (172,284 bytes)
    - https://github.com/bigdatacup/Big-Data-Cup-2026/releases/download/Data/camera_orientations.csv (371 bytes)
  - Everything below marked "observed" comes from those two files (game 2025-10-11, Team A at Team D; tracking is period 1 only: 331,871 rows, 34,413 frames) and from `rink_coords.png`. Other games and periods were not opened.
- Tracking file columns (observed, matches the README's list): `Image Id`, `Period`, `Game Clock`, `Player or Puck`, `Team`, `Player Id`, `Player Jersey Number`, `Rink Location X (Feet)`, `Rink Location Y (Feet)`, `Rink Location Z (Feet)`, `Goal Score`. Further observations:
  - `Image Id` is `<game name>_<frame number>`; the frame numbers in the sample run from 065468 to 132578 (not from 0).
  - `Player or Puck` has only the values `Player` (297,458 rows) and `Puck` (34,413 rows: exactly one per frame). There are no referee rows (the README does not mention referees either).
  - `Team` is `Home` or `Away` for players, blank for the puck.
  - `Player Jersey Number` is blank for 36,171 rows and holds the text `Go` on 24,694 rows, which looks like a goalie marker (observed on rows with x about 80 and -85; the README does not explain `Go`).
  - `Player Id` has 1,058 distinct values in this one period although only about 20 skaters play, so it is not one id per real player; the README says only that every row of one id has the same jersey number. It is blank for puck rows.
  - `Rink Location Z (Feet)` is 1 for players and 0.02 for the puck (observed, matches the README). `Goal Score` is `G` on a single puck row in this period.
  - 1,957 rows have no X or Y position (1,635 puck rows, 322 player rows).
  - Players per frame: 1 to 12 rows (most frames 9 to 11), so not every skater is tracked in every frame.
  - `Game Clock` is written as `20:00`, `2:08`, `1:00` (minutes not zero padded).
- Event types present (observed in the one Events file, 1,878 rows): Play 672, Puck Recovery 511, Incomplete Play 174, Zone Entry 156, Dump In/Out 131, Shot 113, Faceoff Win 59, Takeaway 50, Penalty Taken 6, Goal 6. These match the README's event list. Columns: `Date, Home_Team, Away_Team, Period, Clock, Home_Team_Skaters, Away_Team_Skaters, Home_Team_Goals, Away_Team_Goals, Team, Player_Id, Event, X_Coordinate, Y_Coordinate, Detail_1, Detail_2, Detail_3, Detail_4, Player_Id_2, X_Coordinate_2, Y_Coordinate_2`. In the Events file the `Player_Id` column holds jersey-style numbers (for example 88, 51, 63), unlike the tracking file's `Player Id`. The Shifts file was not opened.
- League and number of games: the league is not stated in the README or `legal.md`. The number of games is not stated either; the release has files for 10 games (observed from the asset list and `camera_orientations.csv`). The word "Hackathon" appears in `legal.md`.
- Frame rate of the tracking data: about 30 frames per second. Observed, not stated: 34,381 of 34,412 consecutive frame-number steps are exactly 1, the median count of frames per distinct `Game Clock` second is 30, and the other 31 steps are gaps (for example 108, 173, 157, 666 frames, presumably stoppages). The README gives no frame rate.
- Coordinate system:
  - Units: feet (README "Rink Location X (Feet)"; observed x range -100.0 to 100.0, y range -41.856 to 42.487).
  - Origin: centre ice (README image: "X = 0, Y = 0 at center ice"; observed: the opening faceoff event sits at x 0.0, y 0.5, and the mean x of all players is near 0).
  - Axes: the README image shows x increasing to the right and y increasing upward, with "Camera Position" drawn below the rink, so positive y is the side far from the camera, the same as our spec. That is stated, from the picture.
  - Orientation: the README says coordinates are "from the perspective of the camera" and `camera_orientations.csv` gives, per game, `GoalieTeamOnRightSideOfRink1stPeriod` (Home or Away). Observed for this game, which the file lists as Away: in period 1 Away goalies are at mean x +80.3 and Home goalies at -85.4, consistent with the file. Whether the sides swap in periods 2 and 3 was not checked (only period 1 was fetched); the README does not say.
  - The event coordinates are on the same -100..100 by -42.5..42.5 scale and are whole or half feet (observed).
- Columns in its tracking layout that our tracking table (spec section 3) could not fill:
  - `Period` and `Game Clock`: our table has `frame` and `time_s` only, with no game clock or period.
  - `Goal Score`: our tracker has no goal event.
  - `Rink Location Z (Feet)`: our table has no height. It could only be written as the constants 1 (players) and 0.02 (puck), which would be invented values, never the true height at a goal.
  - `Team` (`Home` or `Away`): our `team_a` and `team_b` are arbitrary labels per clip; filling Home/Away needs clip metadata or a mapping file.
  - `Image Id` (`<game>_<absolute frame number>`) and numeric `Player Id`: ours are a clip identifier plus a 0-based frame, and a string `player_key`. They can be written in this form but the values do not mean the same thing.
  - `Player Jersey Number`: fillable when known, blank when not; the `Go` goalie marker can be written from our `goalie` entity.
  - Orientation: BDC coordinates follow `camera_orientations.csv` per game; the spec calls the per-period flip out of scope.
  - Rows the other way: our `referee` entity has nowhere to go (BDC has no referee rows, so they must be dropped on export), and our `observed`/`interpolated`/`missing` status maps only to present or blank X/Y.

## Licence combination
- Project licence: GPL-3.0.
- Ultralytics is AGPL-3.0 (PyPI, checked 2026-10-06). Does HockeyRink need it? Yes, to run `HockeyRink.pt` as provided. Stated by the model README (YOLOv8-Large pose); not confirmed from inside the weights, because the pickle was not read.
- What section 13 of the GPL-3.0 text in the project's LICENSE file says about combining with AGPL-3.0 code: it gives permission to link or combine a GPL-3.0 covered work with an AGPL-3.0 work into one combined work and convey it; the GPL terms keep applying to the GPL part, but the AGPL's section 13 (offer source to users who interact with it over a network) applies to the combination as a whole.
- Decision needed from the owner: use Ultralytics for rink keypoints (allowed by GPL section 13, with the AGPL network-source obligation then covering the combined tracker if the site is served over a network), or retrain the keypoint model in a permissively licensed framework.

## Differences from the spec's data table
- HockeyRink: the spec's row is right on content and licence (SHL footage, MIT stated). Missing from the spec: MIT is only a README tag with no LICENSE file; the repository is the model only, and the training dataset is a separate repository; the keypoint count (56) is stated but their names are not in the downloaded files.
- HockeyAI: the spec's "about 2,100 frames" is right for the zip (2,101 frames and labels). Incomplete: the parquet has only 1,890 label-less images; the labels exist only in the zip; the README's folder layout and the listed train/valid/test splits do not exist in the files. The spec's "rink marks" are the classes centerIce, faceoff and goal. MIT is stated only, with no LICENSE file.
- Big Data Cup 2026: the spec says "Player and puck tracking with labelled events; no video" and "Not checked". Now checked. The licence is not an open one: the Stathletes Public Data User Agreement allows public sharing of research only if it "is not used for profit". The spec's data table also lacks that the tracking and event files are not in the repository but are release assets (54 files, 10 games). The tracking data has no referees, and it is at about 30 fps. The spec's "Licences" paragraph flags the jersey data as non-commercial but not this dataset, which has the same limitation.
- VIP-HTD: the spec's "22 NHL broadcast clips with frames, player boxes and track identities" and GPL-3.0 are correct (22 clips, 22 sequences with frames and boxes, LICENSE is GPL-3.0). Incomplete or wrong in the spec: (1) there are two full annotation sets, 13 GB each, with different track ids; (2) the ground truth has a single class: goalies and referees are boxed but not distinguishable from players, and `conf` and `vis` are always 1; (3) split is 14 train, 7 test, 1 validation; (4) frame rates are mixed, 30 fps (10 sequences) and 60 fps (12); (5) 6 sequences have repeated (frame, id) pairs with different boxes; (6) 16 of 22 `seqinfo.ini` names differ from the folder names; (7) frames are 1-based; (8) the clips cover 8 games, not named by the spec. The spec's NHL clips are 1280x720 only.
- York jersey data: not examined (requested by email, owner action).

## Owner actions
- Request the York jersey-number data as described in https://github.com/mkoshkina/jersey-number-pipeline. Status: not yet (this is the owner's action; nothing was sent).
- Decide the Ultralytics question above.
- Decide whether the Big Data Cup non-commercial agreement is acceptable for this project and the site.
