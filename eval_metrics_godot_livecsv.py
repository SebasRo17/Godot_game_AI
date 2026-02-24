# eval_metrics_godot_livecsv.py
from __future__ import annotations

import csv
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from godot_gym_env import GodotPlatformerEnv


@dataclass
class EpisodeMetrics:
    episode: int
    return_sum: float
    steps: int
    wall_time_sec: float

    # Action counts
    move_0: int
    move_1: int
    move_2: int
    jump_0: int
    jump_1: int
    dash_0: int
    dash_1: int

    # Optional (only if your env starts putting these into info)
    success: Optional[int] = None
    death_cause: Optional[str] = None


def _init_action_counter() -> Dict[str, List[int]]:
    return {"move": [0, 0, 0], "jump": [0, 0], "dash": [0, 0]}


def _update_action_counter(counter: Dict[str, List[int]], action: np.ndarray) -> None:
    move, jump, dash = int(action[0]), int(action[1]), int(action[2])
    if 0 <= move < 3:
        counter["move"][move] += 1
    if 0 <= jump < 2:
        counter["jump"][jump] += 1
    if 0 <= dash < 2:
        counter["dash"][dash] += 1


def random_policy(env: GodotPlatformerEnv, obs: np.ndarray) -> np.ndarray:
    return env.action_space.sample()


def append_episode_to_csv(csv_path: Path, metrics: EpisodeMetrics) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row = asdict(metrics)

    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()  # ensure it’s written immediately


def evaluate_env_live_csv(
    env: GodotPlatformerEnv,
    policy_fn: Callable[[GodotPlatformerEnv, np.ndarray], np.ndarray],
    episodes: int = 20,
    max_steps: int = 2000,
    out_csv: Path = Path("eval_metrics.csv"),
    progress_every: int = 200,
) -> List[EpisodeMetrics]:
    """
    Runs episodes and APPENDS ONE ROW TO CSV AFTER EACH EPISODE.
    Also prints progress during the episode so you know it’s running.
    """
    results: List[EpisodeMetrics] = []

    for ep in range(1, episodes + 1):
        t0 = time.time()
        obs, info = env.reset()

        total_reward = 0.0
        step_count = 0
        counter = _init_action_counter()

        success = None
        death_cause = None

        for _ in range(max_steps):
            action = np.asarray(policy_fn(env, obs), dtype=np.int64)
            _update_action_counter(counter, action)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            step_count += 1

            # Live progress (so you SEE it's recording even if episode doesn't end)
            if progress_every and step_count % progress_every == 0:
                print(f"[Eval] ep={ep:03d} step={step_count} return_so_far={total_reward:.2f}")

            # Optional info capture (only works if Godot sends these later)
            if isinstance(info, dict):
                if success is None and "success" in info:
                    success = int(bool(info["success"]))
                if death_cause is None and "death_cause" in info:
                    death_cause = str(info["death_cause"])

            if terminated or truncated:
                break

        wall = time.time() - t0

        m = EpisodeMetrics(
            episode=ep,
            return_sum=total_reward,
            steps=step_count,
            wall_time_sec=wall,
            move_0=counter["move"][0],
            move_1=counter["move"][1],
            move_2=counter["move"][2],
            jump_0=counter["jump"][0],
            jump_1=counter["jump"][1],
            dash_0=counter["dash"][0],
            dash_1=counter["dash"][1],
            success=success,
            death_cause=death_cause,
        )

        results.append(m)

        # ✅ Write immediately after each episode
        append_episode_to_csv(out_csv, m)

        print(
            f"[Eval] SAVED ep={ep:03d} return={m.return_sum:8.3f} steps={m.steps:5d} "
            f"(move={m.move_0},{m.move_1},{m.move_2} jump={m.jump_0},{m.jump_1} dash={m.dash_0},{m.dash_1}) "
            f"-> {out_csv}"
        )

    return results


if __name__ == "__main__":
    # Godot must be running and connecting to 127.0.0.1:11008
    env = GodotPlatformerEnv(host="127.0.0.1", port=11008, timeout_s=10.0, verbose=True)

    try:
        evaluate_env_live_csv(
            env=env,
            policy_fn=random_policy,   # swap later for your trained-model policy
            episodes=10,
            max_steps=400,             # keep small while testing
            out_csv=Path("eval_metrics.csv"),
            progress_every=100,        # prints every 100 steps
        )
    finally:
        # Ensure sockets close cleanly
        env.close()
