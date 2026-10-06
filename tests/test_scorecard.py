import json

import pytest

from tracker.evaluate.scorecard import BARS, build_scorecard, write_scorecard
from tracker.evaluate.tracking import TrackingCounts

GOOD = TrackingCounts(idtp=90, n_gt=100, n_pred=100, tp=95)     # idf1 0.90
POOR = TrackingCounts(idtp=100, n_gt=300, n_pred=300, tp=240)   # idf1 0.333


def test_the_bar_matches_the_spec():
    assert BARS == {"tracking_idf1": 0.85}


def test_overall_is_pooled_from_counts_not_averaged():
    card = build_scorecard({"A_001": GOOD, "B_001": POOR}, failed={}, run={"detector": "x"})
    overall = card["tracking"]["overall"]
    assert overall["idf1"] == pytest.approx(2 * 190 / 800)      # 0.475, not (0.90 + 0.333) / 2
    assert overall["n_gt"] == 400 and overall["fp"] == 65 and overall["fn"] == 65
    assert overall["meets_bar"] is False


def test_each_clip_is_reported_with_its_own_verdict():
    card = build_scorecard({"A_001": GOOD, "B_001": POOR}, failed={}, run={})
    per_clip = card["tracking"]["per_clip"]
    assert per_clip["A_001"]["idf1"] == pytest.approx(0.90)
    assert per_clip["A_001"]["meets_bar"] is True
    assert per_clip["B_001"]["meets_bar"] is False


def test_failed_clips_are_listed_and_do_not_count():
    card = build_scorecard({"A_001": GOOD}, failed={"B_001": "stage 'detect and track' failed"}, run={})
    assert card["failed_clips"] == {"B_001": "stage 'detect and track' failed"}
    assert card["tracking"]["overall"]["n_gt"] == 100


def test_nothing_scored_gives_no_overall():
    card = build_scorecard({}, failed={"A_001": "boom"}, run={})
    assert card["tracking"]["overall"] is None
    assert card["tracking"]["per_clip"] == {}


def test_run_details_are_carried_through():
    card = build_scorecard({"A_001": GOOD}, failed={}, run={"detector": "d", "packages": {"polars": "1"}})
    assert card["run"] == {"detector": "d", "packages": {"polars": "1"}}


def test_files_are_written_and_readable(tmp_path):
    card = build_scorecard(
        {"A_001": GOOD, "B_001": POOR}, failed={"C_001": "could not be read"}, run={"detector": "d"}
    )
    write_scorecard(card, tmp_path / "run")
    assert json.loads((tmp_path / "run" / "scorecard.json").read_text()) == card
    text = (tmp_path / "run" / "scorecard.md").read_text()
    assert "A_001" in text and "B_001" in text
    assert "90.0%" in text                      # A_001 idf1
    assert "47.5%" in text                      # overall idf1
    assert "85.0%" in text                      # the bar
    assert "C_001" in text and "could not be read" in text


def test_markdown_says_so_when_nothing_was_scored(tmp_path):
    write_scorecard(build_scorecard({}, failed={"A_001": "boom"}, run={}), tmp_path)
    assert "No clip could be scored" in (tmp_path / "scorecard.md").read_text()
