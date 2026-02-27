import os
from pathlib import Path

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from godot_gym_env import GodotPlatformerEnv


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

LATEST_MODEL_PATH = CHECKPOINT_DIR / "ppo_speedrun_latest.zip"
VECNORM_PATH = CHECKPOINT_DIR / "vecnormalize.pkl"


# -----------------------------
# Environment factory
# -----------------------------
def make_env(rank: int, base_port=11008):
    def _init():
        env = GodotPlatformerEnv(
            host="127.0.0.1",
            port=base_port + rank,
            timeout_s=60.0,
            verbose=False
        )
        env = Monitor(env)
        return env
    return _init


def main():
    print("START main()")

    NUM_ENVS = 4
    print("Creating SubprocVecEnv...")
    env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])
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
    if LATEST_MODEL_PATH.exists():
        print(f"Loading latest model: {LATEST_MODEL_PATH}")
        model = RecurrentPPO.load(str(LATEST_MODEL_PATH), env=env)
    else:
        print("Creating new PPO model with LSTM memory...")
        model = RecurrentPPO(
            policy="MlpLstmPolicy",
            env=env,
            verbose=1,
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

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback], 
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
