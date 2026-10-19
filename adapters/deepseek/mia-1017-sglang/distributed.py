"""Bind SGLang ranks and adapter accounting to Controller placement."""
from ipaddress import ip_address

def bind(arguments, environment):
    args = list(arguments)
    def value(name):
        index = args.index(name)
        return index, args[index + 1]
    _, nodes = value("--nnodes")
    _, rank = value("--node-rank")
    at, address = value("--dist-init-addr")
    if nodes not in ("3", "4") or rank not in {str(x) for x in range(int(nodes))}:
        raise ValueError("invalid DeepSeek SGLang topology")
    if address != "VONK_MASTER_ADDR:VONK_MASTER_PORT":
        raise ValueError("Controller rendezvous placeholder is required")
    master = environment["VONK_MASTER_ADDR"]
    port = environment["VONK_MASTER_PORT"]
    ip_address(master)
    ip_address(environment["VONK_LOCAL_ADDR"])
    if not (port.isascii() and port.isdigit() and 1024 <= int(port) <= 65535):
        raise ValueError("invalid Controller rendezvous port")
    for name in ("NCCL_SOCKET_IFNAME", "NCCL_IB_HCA", "NCCL_IB_GID_INDEX", "TP_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME"):
        if not environment.get(name):
            raise ValueError("incomplete Controller fabric")
    args[at + 1] = f"{master}:{port}"
    env = dict(environment, NNODES=nodes, NODE_RANK=rank)
    if nodes == "4":
        env["DSV41_LAUNCHER"] = "tp4"
    return args, env
