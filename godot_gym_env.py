import socket
import json
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# -----------------------------
# Framing: u32 little-endian length + JSON bytes
# -----------------------------

def _recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed while receiving.")
        buf += chunk
    return buf

def _send_godot_msg(conn: socket.socket, obj: Dict[str, Any]) -> None:
    data = json.dumps(obj).encode("utf-8")
    conn.sendall(len(data).to_bytes(4, "little") + data)

def _recv_godot_msg(conn: socket.socket) -> Dict[str, Any]:
    length = int.from_bytes(_recv_exact(conn, 4), "little")
    payload = _recv_exact(conn, length)
    return json.loads(payload.decode("utf-8"))



def _agent0(x: Any):
    # Godot often sends arrays per-agent. We train 1 agent => take [0] when needed
    if isinstance(x, list) and len(x) > 0:
        return x[0]
    return x


class GodotPlatformerEnv(gym.Env):
    """
    Actions: MultiDiscrete([move, jump, dash])
      - move: 0..2
      - jump: 0..1
      - dash: 0..1

    Observations: Box(shape=(obs_dim,), dtype=float32) from Godot env_info
    """

    metadata = {"render_modes": []}

    def __init__(self, host="127.0.0.1", port=11008, timeout_s=60.0, verbose=True):
        super().__init__()

        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.verbose = verbose

        self.server: Optional[socket.socket] = None
        self.conn: Optional[socket.socket] = None
        self.addr: Optional[Tuple[str, int]] = None

        # Temporary placeholders (overwritten by env_info)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32
        )

        self.action_space = spaces.MultiDiscrete(
            np.array([3, 2, 2], dtype=np.int64)
        )

        self._start_server_and_accept()
        self._handshake_and_envinfo()

    # -----------------------------
    # Utilities
    # -----------------------------

    def _log(self, *a):
        if self.verbose:
            print(*a)

    # -----------------------------
    # Connection + handshake
    # -----------------------------

    def _start_server_and_accept(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(1)

        self._log(f"[Env] Listening on {self.host}:{self.port}, waiting for Godot...")

        self.conn, self.addr = self.server.accept()
        self._log(f"[Env] Godot connected from {self.addr}")

        self.conn.settimeout(self.timeout_s)

    def _handshake_and_envinfo(self):
        assert self.conn is not None

        # 1) Handshake
        _send_godot_msg(
            self.conn,
            {"type": "handshake", "major_version": "0", "minor_version": "7"}
        )
        self._log("[Env] Sent handshake")

        # 2) Request env_info
        _send_godot_msg(self.conn, {"type": "env_info"})
        self._log("[Env] Requested env_info")

        # 3) Receive env_info
        env_info = _recv_godot_msg(self.conn)

        if env_info.get("type") != "env_info":
            raise RuntimeError(f"Expected env_info response, got: {env_info}")

        self._log("[Env] Got env_info:", env_info)

        # ---- Observation space ----

        obs_size = env_info["observation_space"]["obs"]["size"]
        obs_dim = int(obs_size[0] if isinstance(obs_size, list) else obs_size)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # ---- Action space ----

        move_n = int(env_info["action_space"]["move"]["size"])
        jump_n = int(env_info["action_space"]["jump"]["size"])
        dash_n = int(env_info["action_space"]["dash"]["size"])

        self.action_space = spaces.MultiDiscrete(
            np.array([move_n, jump_n, dash_n], dtype=np.int64)
        )

        self._enable_ai()

    # -----------------------------
    # Gym API
    # -----------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        assert self.conn is not None

        # 👉 Prepared for curriculum/randomization (optional on Godot side)
        reset_payload = {
            "type": "reset",
            # You can use this in Godot later if you want:
            # "variant": int(np.random.randint(0, 5))
        }

        _send_godot_msg(self.conn, reset_payload)
        msg = _recv_godot_msg(self.conn)
        while msg.get("type") == "call":
            if self.verbose:
                print("[Env] Skipping call reply during reset:", msg)
            msg = _recv_godot_msg(self.conn)

        if msg.get("type") not in ("reset", "step"):
            raise RuntimeError(f"Unexpected reset response: {msg}")

        obs_raw = _agent0(msg.get("obs", []))

        if isinstance(obs_raw, dict) and "obs" in obs_raw:
            obs = np.asarray(obs_raw["obs"], dtype=np.float32)
        else:
            obs = np.asarray(obs_raw, dtype=np.float32)

        # ✅ Stabilization: clip extreme values
        obs = np.clip(obs, -10.0, 10.0)

        return obs, {}

    def step(self, action):
        assert self.conn is not None

        move = int(action[0])
        jump = int(action[1])
        dash = int(action[2])

        _send_godot_msg(
            self.conn,
            {
                "type": "action",
                "action": [
                    {"move": move, "jump": jump, "dash": dash}
                ],
            },
        )

        msg = _recv_godot_msg(self.conn)
        while msg.get("type") == "call":
            if self.verbose:
                print("[Env] Skipping call reply during step:", msg)
            msg = _recv_godot_msg(self.conn)

        if msg.get("type") != "step":
            raise RuntimeError(f"Expected step response, got: {msg}")

        obs_raw = _agent0(msg.get("obs"))

        if isinstance(obs_raw, dict) and "obs" in obs_raw:
            obs = np.asarray(obs_raw["obs"], dtype=np.float32)
        else:
            obs = np.asarray(obs_raw, dtype=np.float32)

        # ✅ Stabilization: clip extreme values
        obs = np.clip(obs, -10.0, 10.0)

        reward = float(_agent0(msg.get("reward", 0.0)))
        done = bool(_agent0(msg.get("done", False)))

        terminated = done
        truncated = False

        info = {}

        return obs, reward, terminated, truncated, info

    # -----------------------------
    # Cleanup
    # -----------------------------

    def close(self):
        try:
            if self.conn:
                try:
                    _send_godot_msg(self.conn, {"type": "close"})
                except Exception:
                    pass
        finally:
            try:
                if self.conn:
                    self.conn.close()
            except Exception:
                pass
            try:
                if self.server:
                    self.server.close()
            except Exception:
                pass

    def _enable_ai(self):
        assert self.conn is not None
        _send_godot_msg(
            self.conn,
            {
                "type": "call",
                "path": "/root/DemoLevel/CanvasLayer",
                "method": "set_ai_enabled",
                "args": [True],
            },
        )

