import socket
import json
import random

HOST = "127.0.0.1"
PORT = 11008  # must match Sync.gd


# ---------- low level: same framing as Godot (u32 length + bytes) ----------
def send_json(conn: socket.socket, obj):
    data = json.dumps(obj).encode("utf-8")
    length = len(data).to_bytes(4, "little")  # Godot uses put_u32 (little endian by default)
    conn.sendall(length + data)


def recv_json(conn: socket.socket):
    # read 4-byte length
    header = conn.recv(4)
    if not header:
        raise ConnectionError("Connection closed while reading length")
    length = int.from_bytes(header, "little")

    # read exactly "length" bytes
    buf = b""
    while len(buf) < length:
        chunk = conn.recv(length - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed while reading body")
        buf += chunk

    return json.loads(buf.decode("utf-8"))


# ---------- main loop ----------
def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"[Server] Listening on {HOST}:{PORT}, waiting for Godot...")

        conn, addr = s.accept()
        with conn:
            print(f"[Server] Godot connected from {addr}")

            # 1) Handshake (Python -> Godot)
            send_json(conn, {
                "type": "handshake",
                "major_version": "0",
                "minor_version": "7",
            })
            print("[Server] Sent handshake")

            # 2) Ask for env info (Python -> Godot, then Godot -> Python)
            send_json(conn, {"type": "env_info"})
            env_info = recv_json(conn)
            print("[Server] Got env_info:", env_info)

            action_space = env_info["action_space"]
            n_agents = env_info.get("n_agents", 1)

            # 3) Reset episode
            send_json(conn, {"type": "reset"})
            reset_msg = recv_json(conn)  # {"type":"reset","obs":[...]}
            print("[Server] First obs after reset")

            step = 0
            while True:
                # 4) Build random actions for each agent
                actions = []
                for _ in range(n_agents):
                    a = {}
                    for key, info in action_space.items():
                        if info["action_type"] == "discrete":
                            a[key] = random.randint(0, info["size"] - 1)
                        else:
                            size = info["size"]
                            if isinstance(size, list):
                                size = size[0]
                            a[key] = [random.uniform(-1.0, 1.0) for _ in range(size)]
                    actions.append(a)

                # send action -> Godot
                send_json(conn, {"type": "action", "action": actions})

                # receive step result <- Godot
                msg = recv_json(conn)  # {"type":"step","obs":...,"reward":...,"done":[...]}
                step += 1
                print(f"[Server] step {step} reward {msg['reward']} done {msg['done']}")

                # if any agent finished the episode, reset
                if any(msg["done"]):
                    print("[Server] Episode done -> resetting")
                    send_json(conn, {"type": "reset"})
                    reset_msg = recv_json(conn)


if __name__ == "__main__":
    main()
