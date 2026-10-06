"""Chooses the held-out clips: whole games only, fixed by a seed, recorded in a file."""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

DEFAULT_SEED = 20261006


def game_of(clip_id: str) -> str:
    """The game a clip comes from: its name without the trailing clip number."""
    return re.sub(r"_+\d+$", "", clip_id)


def _games_adding_up_to(order: list[str], sizes: dict[str, int], target: int) -> list[str] | None:
    """The first set of games, taken in the given order, whose clip counts sum to target."""

    def search(start: int, remaining: int) -> list[str] | None:
        if remaining == 0:
            return []
        for i in range(start, len(order)):
            size = sizes[order[i]]
            if size <= remaining:
                rest = search(i + 1, remaining - size)
                if rest is not None:
                    return [order[i]] + rest
        return None

    return search(0, target)


def choose_heldout(clip_ids: list[str], n_heldout: int = 5, seed: int = DEFAULT_SEED) -> list[str]:
    clips = sorted(set(clip_ids))
    if len(clips) <= n_heldout:
        raise ValueError(
            f"need more than {n_heldout} clips to hold out {n_heldout}; got {len(clips)}"
        )
    games: dict[str, list[str]] = defaultdict(list)
    for clip in clips:
        games[game_of(clip)].append(clip)
    order = sorted(games)
    random.Random(seed).shuffle(order)
    sizes = {game: len(members) for game, members in games.items()}

    # Whole games only. Hit the target exactly when some set of games allows it;
    # otherwise overshoot by as little as possible, always leaving clips to train on.
    for target in range(n_heldout, len(clips)):
        chosen = _games_adding_up_to(order, sizes, target)
        if chosen is not None:
            return sorted(clip for game in chosen for clip in games[game])
    raise ValueError("holding out whole games would leave no clips for training")


def save_split(path: Path, clip_ids: list[str], heldout: list[str], seed: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    held = sorted(heldout)
    train = sorted(set(clip_ids) - set(held))
    path.write_text(json.dumps({"seed": seed, "heldout": held, "train": train}, indent=2) + "\n")


def load_split(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"split file not found at {path}; run 'tracker split' first")
    return json.loads(path.read_text())
