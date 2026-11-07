"""Bind one rank of Mia AI Lab's 4x DGX Spark TensorFold launch (start.sh and scripts/config.sh at 8ffa67d5) to the Controller's model, cache and fabric."""
import argparse
import fcntl
from ipaddress import ip_address
import os
from pathlib import Path
import re

from prepare_cache import prepare

ENGINE = Path("/opt/tensorfold")
# scripts/config.sh defaults: a decode round's rows at most, the measured price of a round by its rows and the one-stream chooser's constant.
ROUND_ROWS = "48"
ROUND_MS = "21.5,26.3,31.1,36.0,40.3,44.7,49.0,53.4,57.1,60.8,64.5,68.2,72.0,75.7,79.4,83.1,86.4,89.7,93.0,96.3,99.6,102.9,106.2,109.5,112.8,116.1,119.4,122.7,126.0,129.3,132.6,135.9,138.6,141.3,144.0,146.7,149.4,152.1,154.8,157.5,160.2,162.9,165.6,168.3,171.0,173.7,176.4,179.1"
DRAFT_MS = "25"
# start.sh ENGINE_ENV on a switch (FABRIC_EXPECT=mesh: TF_DS_2D_ORDER 0,1,2,3), apart from TF_DS_STREAMS and TF_DS_WINDOW (bound to
# --parallel and --pool). Every rank gets it.
ENGINE_ENV = {
    "TF_DS_2D_GU": "parity", "TF_DS_2D_ORDER": "0,1,2,3", "TF_DS_ROUND_ROWS": ROUND_ROWS, "TF_DS_SERVED_K": "solo",
    "TF_DS_GRAPHS": "1", "TF_DS_HC_SIDE": "1", "TF_DS_L2_PREFETCH": "1", "TF_DS_ENGRAM_AIO": "1", "TF_DS_ROUND_MS": ROUND_MS,
    "TF_DS_DRAFT_MS": DRAFT_MS,
}
# scripts/nodes.sh nccl_env_n on a switch: the Controller supplies the sockets, rails and GID index, which win.
NCCL_ENV = {
    "NCCL_CROSS_NIC": "1", "NCCL_IB_SUBNET_AWARE_ROUTING": "1", "NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "1",
    "NCCL_MIN_NCHANNELS": "4", "NCCL_MAX_NCHANNELS": "4",
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
    kit, rank_cache, token_map, model, checkpoint = cache / "kit", cache / "rank-cache", cache / "dsv41_token_map.json", cache / "model", "/models/target"
    env = {**NCCL_ENV, **environ, **ENGINE_ENV}
    # The engine's RDMA rings on the Controller's rails and RoCE GID index (the creator leaves the index to each device's first RoCE v2 IPv4 GID).
    env.update(TF_RDMA_DEVICES=rails, TF_DS_RDMA_GID=gid, TF_DS_STREAMS=str(args.parallel), TF_DS_WINDOW=str(args.pool))
    if args.rank == 0:
        # start.sh ENV0 (TF_TP_LISTEN unset: every follower reaches the master's address, as on a switch).
        env.update(TF_TP_WORLD=str(args.tp), TF_TP_MASTER=master, TF_TP_PORT=port, TF_DS_RANK_CACHE=str(rank_cache),
                   TF_DS_KIT=str(kit), TENSORFOLD_CUDA_KERNELS=str(kit / "aot"), TF_DS_ENGRAM=checkpoint,
                   TF_DS_TOKEN_MAP=str(token_map))
        command = [str(ENGINE / "native/bin/tensorfold-native"), "serve", str(model), "--backend", "cuda", "--context", str(args.context),
                   "--parallel", str(args.parallel), *server_args]
    else:
        # start.sh FOLLOW (--pool first, then the context), then the rank's --rdma.
        command = [str(ENGINE / "bin/tf-dsv41-lanes"), checkpoint, str(rank_cache), str(args.rank), str(args.tp), master, port, str(kit),
                   "--pool", str(args.pool), "--context", str(args.context), "--engram", checkpoint, "--token-map", str(token_map),
                   "--drafts", "1", "--round-rows", ROUND_ROWS, "--streams", str(args.parallel), "--graphs", "1", "--side", "1",
                   "--prefetch", "1", "--aio", "1", "--rdma", rails]
    return args, cache, command, env


def main(arguments):
    args, cache, command, env = launch(arguments, os.environ)
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(Path("/models/target"), cache, args.rank, args.tp)
    os.execvpe(command[0], command, env)
