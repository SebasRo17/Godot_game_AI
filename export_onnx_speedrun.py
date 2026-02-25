import argparse
import pathlib
import pickle

import torch
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.type_aliases import RNNStates


class StatelessRecurrentWrapper(torch.nn.Module):
    def __init__(self, model: RecurrentPPO, obs_mean=None, obs_var=None, obs_clip=None, obs_eps=1e-8):
        super().__init__()
        self.policy = model.policy
        lstm_actor = getattr(self.policy, "lstm_actor", None)
        lstm_critic = getattr(self.policy, "lstm_critic", None)
        if lstm_actor is None or lstm_critic is None:
            lstm = getattr(self.policy, "lstm", None)
            if lstm is None:
                raise RuntimeError("No LSTM module found on policy (expected RecurrentPPO).")
            lstm_actor = lstm
            lstm_critic = lstm
        self.lstm_actor = lstm_actor
        self.lstm_critic = lstm_critic
        self.obs_mean = obs_mean
        self.obs_var = obs_var
        self.obs_clip = obs_clip
        self.obs_eps = obs_eps

    def forward(self, obs, state_ins):
        if self.obs_mean is not None and self.obs_var is not None and self.obs_clip is not None:
            obs = (obs - self.obs_mean) / torch.sqrt(self.obs_var + self.obs_eps)
            obs = torch.clamp(obs, -self.obs_clip, self.obs_clip)

        # Stateless inference: reset LSTM each call
        batch = obs.shape[0]
        device = obs.device
        h_actor = torch.zeros((self.lstm_actor.num_layers, batch, self.lstm_actor.hidden_size), device=device)
        c_actor = torch.zeros_like(h_actor)
        h_critic = torch.zeros((self.lstm_critic.num_layers, batch, self.lstm_critic.hidden_size), device=device)
        c_critic = torch.zeros_like(h_critic)
        episode_starts = torch.zeros((batch,), device=device)
        lstm_states = RNNStates((h_actor, c_actor), (h_critic, c_critic))

        actions, _, _, _ = self.policy.forward(obs, lstm_states, episode_starts, deterministic=True)
        actions = actions.float()
        state_outs = torch.zeros((batch, 1), device=device)
        return actions, state_outs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to ppo_speedrun_latest.zip")
    parser.add_argument("--out", required=True, help="Output ONNX path")
    parser.add_argument("--vecnorm", default=None, help="Path to vecnormalize.pkl (optional)")
    args = parser.parse_args()

    model_path = pathlib.Path(args.model)
    if model_path.suffix == "":
        model_path = model_path.with_suffix(".zip")
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vecnorm_path = pathlib.Path(args.vecnorm) if args.vecnorm else model_path.parent / "vecnormalize.pkl"
    obs_mean = None
    obs_var = None
    obs_clip = None
    if vecnorm_path.exists():
        with open(vecnorm_path, "rb") as f:
            vec = pickle.load(f)
        obs_mean = torch.as_tensor(vec.obs_rms.mean, dtype=torch.float32)
        obs_var = torch.as_tensor(vec.obs_rms.var, dtype=torch.float32)
        obs_clip = float(vec.clip_obs)

    model = RecurrentPPO.load(model_path, device="cpu")
    model.policy.to("cpu")
    wrapper = StatelessRecurrentWrapper(
        model,
        obs_mean=obs_mean,
        obs_var=obs_var,
        obs_clip=obs_clip,
    )
    wrapper.eval()

    obs_dim = int(model.observation_space.shape[0])
    dummy_obs = torch.zeros((1, obs_dim), dtype=torch.float32, device="cpu")
    dummy_state = torch.zeros((1,), dtype=torch.float32, device="cpu")

    torch.onnx.export(
        wrapper,
        (dummy_obs, dummy_state),
        out_path.as_posix(),
        input_names=["obs", "state_ins"],
        output_names=["output", "state_outs"],
        opset_version=11,
        dynamic_axes={
            "obs": {0: "batch"},
            "state_ins": {0: "batch"},
            "output": {0: "batch"},
            "state_outs": {0: "batch"},
        },
    )

    print(f"ONNX exported to: {out_path}")


if __name__ == "__main__":
    main()
