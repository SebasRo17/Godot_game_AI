# eval_metrics_godot.py
from __future__ import annotations

import csv
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# Import your env (adjust the import to your project structure)
# If godot_gym_env.py is in the same folder:
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
    # move has 3 bins, jump 2, dash 2 (as defined by env_info/action_space) :contentReference[oaicite:3]{index=3}
    return {
        "move": [0, 0, 0],
        "jump": [0, 0],
        "dash": [0, 0],
    }


def _update_action_counter(counter: Dict[str, List[int]], action: np.ndarray) -> None:
    move, jump, dash = int(action[0]), int(action[1]), int(action[2])
    # Defensive bounds (in case a policy outputs something weird)
    if 0 <= move < len(counter["move"]):
        counter["move"][move] += 1
    if 0 <= jump < len(counter["jump"]):
        counter["jump"][jump] += 1
    if 0 <= dash < len(counter["dash"]):
        counter["dash"][dash] += 1


def random_policy(env: GodotPlatformerEnv, obs: np.ndarray) -> np.ndarray:
    # Uses env.action_space which is MultiDiscrete([move_n, jump_n, dash_n]) :contentReference[oaicite:4]{index=4}
    return env.action_space.sample()


def make_greedy_move_policy() -> Callable[[GodotPlatformerEnv, np.ndarray], np.ndarray]:
    """
    A simple baseline: always 'move right' (assumes move=2 is right),
    never jump, never dash. Change mapping if your Godot uses different encoding.
    """
    def _policy(env: GodotPlatformerEnv, obs: np.ndarray) -> np.ndarray:
        return np.array([2, 0, 0], dtype=np.int64)
    return _policy


def evaluate_env(
    env: GodotPlatformerEnv,
    policy_fn: Callable[[GodotPlatformerEnv, np.ndarray], np.ndarray],
    episodes: int = 20,
    max_steps: int = 10_000,
    out_csv: Optional[Path] = Path("eval_metrics.csv"),
    sleep_s: float = 0.0,
) -> List[EpisodeMetrics]:
    """
    Runs N episodes and records:
      - return_sum (sum of rewards)
      - steps (episode length)
      - wall_time_sec (real time)
      - action counts by dimension: move/jump/dash
      - optional success/death_cause if you later add them into `info`

    Note: In your current env implementation, `info` is {} :contentReference[oaicite:5]{index=5}
    so success/death_cause will stay None unless you extend Godot -> Python messages.
    """
    results: List[EpisodeMetrics] = []

    for ep in range(1, episodes + 1):
        t0 = time.time()
        obs, info = env.reset()

        total_reward = 0.0
        step_count = 0
        counter = _init_action_counter()

        # Optional fields if you later put them in info
        success = None
        death_cause = None

        for _ in range(max_steps):
            action = policy_fn(env, obs)
            action = np.asarray(action, dtype=np.int64)

            _update_action_counter(counter, action)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            step_count += 1

            # If you later add these keys in Godot messages and pass into info,
            # this will auto-capture them:
            if isinstance(info, dict):
                if success is None and "success" in info:
                    success = int(bool(info["success"]))
                if death_cause is None and "death_cause" in info:
                    death_cause = str(info["death_cause"])

            if terminated or truncated:
                break

            if sleep_s > 0:
                time.sleep(sleep_s)

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

        print(
            f"[Eval] ep={ep:03d} return={m.return_sum:8.3f} steps={m.steps:5d} "
            f"actions(move={m.move_0},{m.move_1},{m.move_2} jump={m.jump_0},{m.jump_1} dash={m.dash_0},{m.dash_1}) "
            f"wall={m.wall_time_sec:6.2f}s"
        )

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
            writer.writeheader()
            for row in results:
                writer.writerow(asdict(row))
        print(f"[Eval] Saved metrics -> {out_csv.resolve()}")

    return results


def summarize(results: List[EpisodeMetrics]) -> None:
    returns = np.array([r.return_sum for r in results], dtype=np.float64)
    steps = np.array([r.steps for r in results], dtype=np.float64)

    def _mean_std(x: np.ndarray) -> Tuple[float, float]:
        return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0

    r_mean, r_std = _mean_std(returns)
    s_mean, s_std = _mean_std(steps)

    print("\n=== Summary ===")
    print(f"episodes: {len(results)}")
    print(f"return:   mean={r_mean:.3f} std={r_std:.3f} min={returns.min():.3f} max={returns.max():.3f}")
    print(f"steps:    mean={s_mean:.2f} std={s_std:.2f} min={steps.min():.0f} max={steps.max():.0f}")

    # Optional success rate if present
    successes = [r.success for r in results if r.success is not None]
    if successes:
        sr = sum(successes) / len(successes)
        print(f"success_rate: {sr*100:.1f}% (n={len(successes)})")


if __name__ == "__main__":
    # IMPORTANT:
    # Your Godot scene must be running and connecting to the env socket.
    # GodotPlatformerEnv starts a server and waits for Godot to connect :contentReference[oaicite:6]{index=6}
    env = GodotPlatformerEnv(host="127.0.0.1", port=11008, timeout_s=10.0, verbose=True)

    try:
        # Choose a policy:
        policy = random_policy
        # policy = make_greedy_move_policy()

        results = evaluate_env(
            env=env,
            policy_fn=policy,
            episodes=20,
            max_steps=5000,
            out_csv=Path("eval_metrics.csv"),
            sleep_s=0.0,  # set small value if you want slower visible gameplay
        )
        summarize(results)
    finally:
        env.close()
