"""Bind one native TensorFold rank to the Controller's model, cache and fabric."""
import argparse
import fcntl
from ipaddress import ip_address
import os
from pathlib import Path
import re

from prepare_cache import prepare


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
    parser.add_argument("--no-drafts", action="store_true")
    args, server_args = parser.parse_known_args(arguments)
    if args.no_drafts:
        server_args.append("--no-drafts")
    if args.tp != int(environ["VONK_TF_WORLD"]) or args.tp != 2 or not 0 <= args.rank < args.tp:
        raise ValueError("rank/topology does not match this TensorFold engine")
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
    env = dict(environ)
    env.update(TF_RDMA_DEVICES=rails, TF_RDMA_GID=gid, TF_RDMA_GID_INDEX=gid,
               TF_TP_WORLD=str(args.tp), TF_TP_MASTER=master, TF_TP_PORT=port,
               TF_DS_RANK_CACHE=str(cache / "rank-cache"), TF_DS_KIT=str(cache / "kit"),
               TENSORFOLD_CUDA_KERNELS=str(cache / "kit/aot"), TF_DS_ENGRAM="/models/engram",
               TF_DS_TOKEN_MAP=str(cache / "dsv41_token_map.json"), TF_DS_GRAPHS="1",
               TF_DS_HC_SIDE="1", TF_DS_L2_PREFETCH="1", TF_DS_ENGRAM_AIO="1",
               TF_DS_STREAMS=str(args.parallel))
    env["TF_DS_REPLAY_FLOOR"] = "1"
    model = cache / "model"
    if args.rank == 0:
        command = ["/opt/tensorfold/bin/tensorfold-native", "serve", str(model), "--context", str(args.context), "--parallel", str(args.parallel), *server_args, "--backend", "cuda"]
    else:
        command = ["/opt/tensorfold/bin/tf-dsv41-lanes", str(model), str(cache / "rank-cache"), str(args.rank), str(args.tp), master, port, str(cache / "kit"), "--engram", "/models/engram", "--token-map", str(cache / "dsv41_token_map.json"), "--pool", str(args.context), "--rdma", rails, "--graphs", "1", "--side", "1", "--prefetch", "1", "--aio", "1", "--streams", str(args.parallel), "--drafts", "0" if args.no_drafts else "1"]
    return args, cache, command, env


def main(arguments):
    args, cache, command, env = launch(arguments, os.environ)
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(Path("/models/target"), cache, args.rank, args.tp)
    os.execvpe(command[0], command, env)
