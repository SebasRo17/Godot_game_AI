import os
import time
from pathlib import Path

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from godot_gym_env import GodotPlatformerEnv


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
LOGS_DIR = CHECKPOINT_DIR / "LOGS"
LOGS_DIR.mkdir(exist_ok=True)

LATEST_MODEL_PATH = CHECKPOINT_DIR / "ppo_speedrun_latest.zip"
VECNORM_PATH = CHECKPOINT_DIR / "vecnormalize.pkl"
STEPS_PER_SECOND_CSV = LOGS_DIR / "steps_per_second.csv"


class StepsPerSecondCallback(BaseCallback):
    def __init__(self, csv_path: Path):
        super().__init__()
        self.csv_path = csv_path
        self._last_time = None
        self._last_steps = 0

    def _init_callback(self) -> None:
        self._last_time = time.time()
        self._last_steps = self.num_timesteps

        if not self.csv_path.exists():
            with open(self.csv_path, "w", encoding="utf-8") as f:
                f.write("timestamp,total_timesteps,steps,elapsed_s,steps_per_second\n")

    def _on_rollout_end(self) -> None:
        now = time.time()
        elapsed = now - self._last_time if self._last_time else 0.0
        steps = self.num_timesteps - self._last_steps
        if elapsed <= 0 or steps <= 0:
            return
        sps = steps / elapsed
        with open(self.csv_path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{self.num_timesteps},{steps},{elapsed:.3f},{sps:.3f}\n")

        self._last_time = now
        self._last_steps = self.num_timesteps

    def _on_step(self) -> bool:
        return True


# -----------------------------
# Environment factory
# -----------------------------
def make_env(rank: int, base_port=11008, monitor_dir: Path = None):
    def _init():
        env = GodotPlatformerEnv(
            host="127.0.0.1",
            port=base_port + rank,
            timeout_s=60.0,
            verbose=False
        )
        if monitor_dir:
            monitor_file = monitor_dir / f"monitor_{rank}.csv"
            env = Monitor(env, filename=str(monitor_file))
        else:
            env = Monitor(env)
        return env
    return _init


def main():
    print("START main()")

    NUM_ENVS = 8
    monitor_dir = CHECKPOINT_DIR / "monitor"
    monitor_dir.mkdir(exist_ok=True)
    print("Creating SubprocVecEnv...")
    env = SubprocVecEnv([make_env(i, monitor_dir=monitor_dir) for i in range(NUM_ENVS)])
    print("SubprocVecEnv created")

    # ---------------------------------
    # Load or create VecNormalize (solo al inicio)
    # ---------------------------------
    if VECNORM_PATH.exists():
        print("Loading VecNormalize stats (final-only)...")
        env = VecNormalize.load(str(VECNORM_PATH), env)
        env.training = True
        env.norm_reward = True
    else:
        print("Creating new VecNormalize...")
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # ---------------------------------
    # Model checkpoints (OK tenerlos)
    # ---------------------------------
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_speedrun"
    )

    # ---------------------------------
    # Load "latest" if exists, else create model
    # ---------------------------------
    tb_log_dir = str(CHECKPOINT_DIR / "tensorboard")

    if LATEST_MODEL_PATH.exists():
        print(f"Loading latest model: {LATEST_MODEL_PATH}")
        model = RecurrentPPO.load(str(LATEST_MODEL_PATH), env=env)
        model.set_logger(configure(tb_log_dir, ["stdout", "tensorboard"]))
    else:
        print("Creating new PPO model with LSTM memory...")
        model = RecurrentPPO(
            policy="MlpLstmPolicy",
            env=env,
            verbose=1,
            tensorboard_log=tb_log_dir,
            n_steps=128,
            batch_size=512,
            learning_rate=3e-4,
            gamma=0.997,
            clip_range=0.2,
            ent_coef=0.02,
        )

    # ---------------------------------
    # Train
    # ---------------------------------
    TOTAL_TIMESTEPS = 1_000_000

    sps_callback = StepsPerSecondCallback(STEPS_PER_SECOND_CSV)
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, sps_callback],
    )

    # ---------------------------------
    # Save final model + VecNormalize (solo aquí)
    # ---------------------------------
    model.save(CHECKPOINT_DIR / "ppo_speedrun_latest")  # crea ppo_speedrun_latest.zip
    env.save(str(VECNORM_PATH))  # vecnormalize.pkl SOLO al final

    env.close()
    print("Training finished and model saved")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)
    main()
