import pytest

from tracker.datasets.split import choose_heldout, game_of, load_split, save_split

REAL_NAMES = [
    "CAR_VS_BOS_001", "CAR_VS_BOS_002",
    "CAR_VS_NYR_001", "CAR_VS_NYR_002", "CAR_VS_NYR_003",
    "CGY_VS_DAL_001", "CGY_VS_DAL_002", "CGY_VS_DAL_003",
    "CHI_VS_TOR_002", "CHI_VS_TOR_003", "CHI_VS_TOR_004",
    "CNTRL_VS_PAC_001", "CNTRL_VS_PAC_002", "CNTRL_VS_PAC_005", "CNTRL_VS_PAC_006",
    "PIT_VS_SJ_001", "PIT_VS_SJ_002", "PIT_VS_SJ_003",
    "PIT_VS_WAS_001",
    "STL_VS_SJ_001", "STL_VS_SJ_002",
]


def test_game_is_the_name_without_its_trailing_number():
    assert game_of("CAR_VS_BOS_001") == "CAR_VS_BOS"
    assert game_of("CAR_VS_NYR__001") == "CAR_VS_NYR"
    assert game_of("CNTRL_VS_PAC_2019_006") == "CNTRL_VS_PAC_2019"


def test_game_of_a_name_without_a_number_is_the_name():
    assert game_of("WARMUP") == "WARMUP"


def test_exactly_five_clips_are_held_out_from_the_real_names():
    assert len(choose_heldout(REAL_NAMES)) == 5


def test_no_game_is_on_both_sides():
    held = set(choose_heldout(REAL_NAMES))
    train = set(REAL_NAMES) - held
    assert {game_of(c) for c in held}.isdisjoint({game_of(c) for c in train})


def test_same_seed_gives_same_split_whatever_the_input_order():
    assert choose_heldout(REAL_NAMES, seed=7) == choose_heldout(list(reversed(REAL_NAMES)), seed=7)


def test_different_seeds_can_give_different_splits():
    splits = {tuple(choose_heldout(REAL_NAMES, seed=s)) for s in range(20)}
    assert len(splits) > 1


def test_overshoots_when_whole_games_cannot_make_the_exact_number():
    names = [f"G{g}_{i:03d}" for g in range(4) for i in range(1, 4)]   # 4 games of 3
    held = choose_heldout(names, n_heldout=5)
    assert len(held) == 6
    assert len({game_of(c) for c in held}) == 2


def test_too_few_clips_is_an_error():
    with pytest.raises(ValueError, match="more than 5"):
        choose_heldout(REAL_NAMES[:5])


def test_a_single_game_cannot_be_split():
    names = [f"ONLY_GAME_{i:03d}" for i in range(1, 11)]
    with pytest.raises(ValueError, match="no clips for training"):
        choose_heldout(names)


def test_split_round_trips_through_a_file(tmp_path):
    held = choose_heldout(REAL_NAMES, seed=3)
    path = tmp_path / "splits" / "heldout.json"
    save_split(path, REAL_NAMES, held, seed=3)
    loaded = load_split(path)
    assert loaded["seed"] == 3
    assert loaded["heldout"] == held
    assert sorted(loaded["heldout"] + loaded["train"]) == sorted(REAL_NAMES)
