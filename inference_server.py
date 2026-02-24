import json, socket, argparse, math
from pathlib import Path

# ---------- framed protocol: u32 length + JSON ----------
def send_json(conn, obj):
    data = json.dumps(obj).encode("utf-8")
    conn.sendall(len(data).to_bytes(4, "little") + data)

def recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed.")
        buf += chunk
    return buf

def recv_json(conn):
    n = int.from_bytes(recv_exact(conn, 4), "little")
    payload = recv_exact(conn, n)
    return json.loads(payload.decode("utf-8"))

# ---------- model loader (JSON) ----------
def load_model(path: Path):
    m = json.loads(path.read_text(encoding="utf-8"))
    return m

def matvec(W, x):
    # W: [out][in], x: [in]
    out = []
    for row in W:
        s = 0.0
        for w, xi in zip(row, x):
            s += float(w) * float(xi)
        out.append(s)
    return out

def addvec(a, b):
    return [float(x) + float(y) for x, y in zip(a, b)]

def relu(x):
    return [max(0.0, float(v)) for v in x]

def softmax(logits):
    mx = max(logits)
    exps = [math.exp(v - mx) for v in logits]
    s = sum(exps)
    return [e / s for e in exps]

def argmax(xs):
    best_i = 0
    best_v = xs[0]
    for i, v in enumerate(xs):
        if v > best_v:
            best_v = v
            best_i = i
    return best_i

def forward(model, obs):
    """
    Tries to handle common JSON layouts.
    Expected (typical):
      model["policy"]["W1"], ["b1"], ["W2"], ["b2"] ...
    If your JSON has different keys, tell me and I’ll adapt it.
    """
    # Try common structures
    if "policy" in model:
        pol = model["policy"]
    else:
        pol = model

    # Common key guesses
    W1 = pol.get("W1") or pol.get("w1") or pol.get("weights1")
    b1 = pol.get("b1") or pol.get("B1") or pol.get("bias1")
    W2 = pol.get("W2") or pol.get("w2") or pol.get("weights2")
    b2 = pol.get("b2") or pol.get("B2") or pol.get("bias2")

    if W1 is None or b1 is None or W2 is None or b2 is None:
        raise KeyError("Model JSON keys not found (need W1,b1,W2,b2 or similar).")

    h = relu(addvec(matvec(W1, obs), b1))
    logits = addvec(matvec(W2, h), b2)
    return logits

def logits_to_action_dict(logits):
    """
    THIS PART depends on your output_dim.
    For your Godot actions (move=3, jump=2, dash=2) total = 7 logits.

    We will split:
      move: logits[0:3] -> argmax 0..2
      jump: logits[3:5] -> argmax 0..1
      dash: logits[5:7] -> argmax 0..1
    """
    if len(logits) < 7:
        # fallback: single discrete action
        a = argmax(softmax(logits))
        return {"move": 1, "jump": 0, "dash": 0} if a == 0 else {"move": 2, "jump": 0, "dash": 0}

    move = argmax(softmax(logits[0:3]))
    jump = argmax(softmax(logits[3:5]))
    dash = argmax(softmax(logits[5:7]))
    return {"move": int(move), "jump": int(jump), "dash": int(dash)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=11009)
    ap.add_argument("--model", default="models/demo_speedrun_model.json")
    args = ap.parse_args()

    model = load_model(Path(args.model))
    print("[Infer] Loaded model:", args.model)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    print(f"[Infer] Listening on {args.host}:{args.port}...")

    conn, addr = srv.accept()
    print("[Infer] Godot connected:", addr)

    # 1) Send handshake
    send_json(conn, {"type": "handshake", "major_version": "0", "minor_version": "7"})

    # 2) Request env_info (so Godot replies and sync starts)
    send_json(conn, {"type": "env_info"})
    env_info = recv_json(conn)
    print("[Infer] env_info:", env_info)

    n_agents = int(env_info.get("n_agents", 1))

    # 3) Reset loop
    while True:
        send_json(conn, {"type": "reset"})
        msg = recv_json(conn)  # expecting {"type":"reset","obs":[...]}
        if msg.get("type") != "reset":
            # some implementations reply with "step" on reset; handle both
            if msg.get("type") not in ("reset", "step"):
                raise RuntimeError("Unexpected message: " + str(msg))

        obs_list = msg.get("obs", [])
        # obs_list is list per agent; each item may be {"obs":[...]}
        steps = 0
        done = [False] * n_agents

        while not any(done):
            actions = []
            for i in range(n_agents):
                o = obs_list[i]
                vec = o["obs"] if isinstance(o, dict) and "obs" in o else o
                logits = forward(model, vec)
                actions.append(logits_to_action_dict(logits))

            send_json(conn, {"type": "action", "action": actions})

            step_msg = recv_json(conn)  # {"type":"step","obs":...,"reward":...,"done":...}
            if step_msg.get("type") != "step":
                raise RuntimeError("Expected step, got: " + str(step_msg))

            obs_list = step_msg.get("obs", obs_list)
            done = step_msg.get("done", done)
            steps += 1

        print(f"[Infer] Episode finished in {steps} steps. Restarting...")

if __name__ == "__main__":
    main()
