import multiprocessing as mp
from godot_gym_env import GodotPlatformerEnv

def run_env(rank, base_port=11008):
    port = base_port + rank
    print(f"[child {rank}] starting env on port {port}", flush=True)
    env = GodotPlatformerEnv(host="127.0.0.1", port=port, timeout_s=10.0, verbose=True)
    print(f"[child {rank}] connected ok on port {port}", flush=True)
    env.close()

def main():
    procs = []
    for i in range(8):
        p = mp.Process(target=run_env, args=(i,))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)
    main()
