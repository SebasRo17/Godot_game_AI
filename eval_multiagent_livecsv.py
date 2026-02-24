import csv
import json
import socket
import time
from pathlib import Path
from typing import Dict, List, Any
import random


# --- framing: u32 little endian + JSON bytes (matches your Godot sync.gd) ---
def send_json(conn: socket.socket, obj: Dict[str, Any]) -> None:
    data = json.dumps(obj).encode("utf-8")
    conn.sendall(len(data).to_bytes(4, "little") + data)

def recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed")
        buf += chunk
    return buf

def recv_json(conn: socket.socket) -> Dict[str, Any]:
    length = int.from_bytes(recv_exact(conn, 4), "little")
    payload = recv_exact(conn, length)
    return json.loads(payload.decode("utf-8"))


def append_rows(csv_path: Path, rows: List[Dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not file_exists:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)
        f.flush()


def main(
    host="127.0.0.1",
    port=11008,
    episodes=10,
    max_steps=2000,
    out_csv="eval_multiagent.csv",
):
    out_csv = Path(out_csv)

    # server accepts Godot (same as your godot_gym_env does internally)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print(f"[EvalMA] Listening on {host}:{port}, waiting for Godot...")
    conn, addr = srv.accept()
    print("[EvalMA] Godot connected:", addr)

    try:
        # 1) handshake
        send_json(conn, {"type": "handshake", "major_version": "0", "minor_version": "7"})
        print("[EvalMA] Sent handshake")

        # 2) env_info
        send_json(conn, {"type": "env_info"})
        env_info = recv_json(conn)
        print("[EvalMA] env_info:", env_info)

        action_space = env_info["action_space"]
        n_agents = int(env_info.get("n_agents", 1))

        print(f"[EvalMA] n_agents={n_agents} (Godot side).")

        for ep in range(1, episodes + 1):
            # reset
            send_json(conn, {"type": "reset"})
            msg = recv_json(conn)
            obs_list = msg.get("obs", [])
            if len(obs_list) != n_agents:
                print("[EvalMA] WARNING: obs_list len != n_agents:", len(obs_list), n_agents)

            # per-agent accumulators
            ret = [0.0] * n_agents
            steps = [0] * n_agents
            move_counts = [[0, 0, 0] for _ in range(n_agents)]
            jump_counts = [[0, 0] for _ in range(n_agents)]
            dash_counts = [[0, 0] for _ in range(n_agents)]
            success = [None] * n_agents
            death_cause = [None] * n_agents

            done = [False] * n_agents
            t0 = time.time()

            for t in range(max_steps):
                # build actions for each agent (random baseline)
                actions = []
                for i in range(n_agents):
                    a = {}
                    for key, info in action_space.items():
                        if info["action_type"] == "discrete":
                            a[key] = random.randint(0, int(info["size"]) - 1)
                        else:
                            size = info["size"]
                            if isinstance(size, list):
                                size = size[0]
                            a[key] = [random.uniform(-1.0, 1.0) for _ in range(int(size))]
                    actions.append(a)

                    # count actions
                    move_counts[i][a["move"]] += 1
                    jump_counts[i][a["jump"]] += 1
                    dash_counts[i][a["dash"]] += 1

                send_json(conn, {"type": "action", "action": actions})

                step_msg = recv_json(conn)
                if step_msg.get("type") != "step":
                    raise RuntimeError("Expected step, got: " + str(step_msg))

                # rewards/done are lists per agent
                reward_list = step_msg.get("reward", [0.0] * n_agents)
                done = step_msg.get("done", done)

                # optional info per agent if you added it in sync.gd patch
                info_list = step_msg.get("info", [None] * n_agents)

                for i in range(n_agents):
                    ret[i] += float(reward_list[i])
                    steps[i] += 1

                    if isinstance(info_list, list) and i < len(info_list) and isinstance(info_list[i], dict):
                        if success[i] is None and "success" in info_list[i]:
                            success[i] = int(bool(info_list[i]["success"]))
                        if death_cause[i] is None and "death_cause" in info_list[i]:
                            death_cause[i] = info_list[i]["death_cause"]

                if any(done):
                    break

            wall = time.time() - t0

            # write one row per agent for this episode
            rows = []
            for i in range(n_agents):
                rows.append({
                    "episode": ep,
                    "agent_id": i,
                    "return_sum": ret[i],
                    "steps": steps[i],
                    "wall_time_sec": wall,
                    "move_0": move_counts[i][0],
                    "move_1": move_counts[i][1],
                    "move_2": move_counts[i][2],
                    "jump_0": jump_counts[i][0],
                    "jump_1": jump_counts[i][1],
                    "dash_0": dash_counts[i][0],
                    "dash_1": dash_counts[i][1],
                    "success": success[i],
                    "death_cause": death_cause[i],
                })

            append_rows(out_csv, rows)
            print(f"[EvalMA] SAVED ep={ep:03d} rows={len(rows)} -> {out_csv}")

    except KeyboardInterrupt:
        print("\n[EvalMA] Ctrl+C received. Data up to the last completed episode is already saved.")
    finally:
        try:
            # politely tell Godot we’re done
            send_json(conn, {"type": "close"})
        except Exception:
            pass
        conn.close()
        srv.close()
        print("[EvalMA] Closed.")


if __name__ == "__main__":
    main(episodes=20, max_steps=2000, out_csv="eval_multiagent.csv")
