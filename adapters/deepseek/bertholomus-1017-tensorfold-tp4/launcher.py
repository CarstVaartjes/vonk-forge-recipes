"""Bind one rank of bertholomus's v0.3.7 TP4 launch (scripts/serve.sh, config.env.example) to the Controller's model, cache and fabric."""
import argparse
import fcntl
from ipaddress import ip_address
import os
from pathlib import Path
import re

from prepare_cache import prepare

ENGINE = Path("/opt/tensorfold")
RDMA_KERNELS = ENGINE / "fatbin/dsv41_rdma.fatbin"
# config.env.example ENGINE_ENV at acf0d2ba (v0.3.7): every rank's environment, apart from TF_DS_STREAMS and TF_DS_WINDOW
# (bound to --parallel and --pool) and TENSORFOLD_GRAMMAR_LIB (the cache's kit).
ROUND_ROWS = "64"
ROUND_MS = "14.5,17.4,19.9,22.3,24.0,25.9,27.5,29.2,30.8,32.5,34.1,35.7,37.4,39.0,40.7,42.3,43.9,45.6,47.2,48.9,50.5,52.1,53.8,55.4,57.1,58.7,60.3,62.0,63.6,65.3,66.9,68.5,70.2,71.8,73.5,75.1,76.7,78.4,80.0,81.7,83.3,84.9,86.6,88.2,89.9,91.5,93.1,94.8,96.4,98.1,99.7,101.4,103.0,104.6,106.3,107.9,109.6,111.2,112.9,114.5,116.1,117.8,119.4,121.1"
ENGINE_ENV = {
    "TF_DS_2D_GU": "parity", "TF_DS_ROUND_ROWS": ROUND_ROWS, "TF_DS_SERVED_K": "solo", "TF_DS_ROUND_MS": ROUND_MS,
    "TF_DS_IDX_SPLIT": "8192", "TENSORFOLD_PLAIN_SPECIALS": "1", "TENSORFOLD_SSE_KEEPALIVE": "15",
    "TENSORFOLD_STATS_EVERY": "60", "TF_FILL_CHUNKS": "1", "TF_FILL_MANY": "1", "TF_DS_2D_YX": "1", "TF_DS_WAVES": "1",
    "NCCL_MAX_NCHANNELS": "4",
}


def fabric(environ):
    selectors = environ.get("NCCL_IB_HCA", "").removeprefix("=").split(",")
    names = []
    for selector in selectors:
        match = re.fullmatch(r"([A-Za-z0-9_.-]+):1", selector)
        if not match or match[1] in names:
            raise ValueError("TensorFold requires distinct Controller-authorized port-1 RoCE rails")
        names.append(match[1])
    gid = environ.get("NCCL_IB_GID_INDEX", "")
    if not (gid.isascii() and gid.isdigit() and 0 <= int(gid) <= 255):
        raise ValueError("Controller RoCE GID index is absent or invalid")
    return ",".join(names), gid


def launch(arguments, environ):
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--master", required=True)
    parser.add_argument("--master-port", required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--parallel", type=int, required=True)
    parser.add_argument("--pool", type=int, required=True)
    args, server_args = parser.parse_known_args(arguments)
    if args.tp != int(environ["VONK_TF_WORLD"]) or args.tp != 4 or not 0 <= args.rank < args.tp:
        raise ValueError("rank/topology does not match this TensorFold engine")
    if args.pool < args.context:
        raise ValueError("the shared position pool must hold one full-context request")
    master = environ.get(args.master, "") if args.master == "VONK_MASTER_ADDR" else args.master
    port = environ.get(args.master_port, "") if args.master_port == "VONK_MASTER_PORT" else args.master_port
    ip_address(master)
    if not (port.isascii() and port.isdigit() and 1024 <= int(port) <= 65535):
        raise ValueError("invalid Controller rendezvous port")
    rails, gid = fabric(environ)
    revision = environ["VONK_MODEL_REVISION"]
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("missing immutable model identity")
    cache = Path("/outputs/cache/deepseek-1017") / revision / f"tp{args.tp}-rank{args.rank}"
    kit, rank_cache, token_map, model = cache / "kit", cache / "rank-cache", cache / "dsv41_token_map.json", cache / "model"
    env = dict(environ)
    env.update(ENGINE_ENV)
    # serve.sh LINK with the Controller's rails (both RoCE halves when both are authorized) and its RoCE GID index:
    # the engine's v0.3.6 opt-in TF_RDMA_GID, which bertholomus leaves at its default 5.
    env.update(TF_RDMA_DEVICES=rails, TF_RDMA_GID=gid, TF_DS_STREAMS=str(args.parallel), TF_DS_WINDOW=str(args.pool),
               TENSORFOLD_GRAMMAR_LIB=str(kit / "grammar/libtfgrammar.so"))
    if args.rank == 0:
        env.update(TF_TP_WORLD=str(args.tp), TF_TP_MASTER=master, TF_TP_PORT=port, TF_DS_RANK_CACHE=str(rank_cache),
                   TF_DS_KIT=str(kit), TENSORFOLD_CUDA_KERNELS=str(kit / "aot"), TF_DS_ENGRAM="/models/engram",
                   TF_DS_TOKEN_MAP=str(token_map), TF_DS_RDMA_KERNELS=str(RDMA_KERNELS), TF_DS_GRAPHS="1",
                   TF_DS_HC_SIDE="1", TF_DS_L2_PREFETCH="1", TF_DS_ENGRAM_AIO="1")
        command = [str(ENGINE / "bin/tensorfold-native"), "serve", str(model), "--context", str(args.context), "--parallel", str(args.parallel), *server_args, "--backend", "cuda"]
    else:
        # serve.sh `fam` then FOLLOW_FLAGS: the first --pool is the context, the later one (the window) wins.
        command = [str(ENGINE / "bin/tf-dsv41-lanes"), str(model), str(rank_cache), str(args.rank), str(args.tp), master, port, str(kit),
                   "--pool", str(args.context), "--engram", "/models/engram", "--token-map", str(token_map), "--rdma", rails,
                   "--rdma-kernels", str(RDMA_KERNELS), "--graphs", "1", "--side", "1", "--prefetch", "1", "--aio", "1",
                   "--round-rows", ROUND_ROWS, "--streams", str(args.parallel), "--pool", str(args.pool), "--context", str(args.context)]
    return args, cache, command, env


def main(arguments):
    args, cache, command, env = launch(arguments, os.environ)
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(Path("/models/target"), Path("/models/kit-bias/kit/vision/gate_bias_vl.safetensors"), cache, args.rank, args.tp)
    os.execvpe(command[0], command, env)
