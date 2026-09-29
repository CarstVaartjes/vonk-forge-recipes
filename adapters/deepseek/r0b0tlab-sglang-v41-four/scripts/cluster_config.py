"""Explicit, private four-rank inventory for the public TP4 launcher."""
import ipaddress
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(os.environ.get('DSV41_CLUSTER_CONFIG', ROOT / 'cluster.local.json'))
KEYS = {'ssh_hosts', 'hostnames', 'fabric_ips', 'model_path', 'ssh_identity_file',
        'ssh_known_hosts_file', 'nccl_env', 'dist_init_addr'}


def validate(data):
    if not isinstance(data, dict) or set(data) != KEYS:
        raise ValueError('cluster inventory must contain exactly: ' + ', '.join(sorted(KEYS)))
    for key in ('ssh_hosts', 'hostnames', 'fabric_ips'):
        values = data[key]
        if not isinstance(values, list) or len(values) != 4 or len(set(values)) != 4:
            raise ValueError(key + ' must have four distinct rank-ordered strings')
        if any(not isinstance(x, str) or not x or x.startswith('-') or any(c.isspace() for c in x) for x in values):
            raise ValueError('invalid ' + key)
    for value in data['fabric_ips']:
        ipaddress.IPv4Address(value)
    for key in ('model_path', 'ssh_identity_file', 'ssh_known_hosts_file', 'nccl_env'):
        if not isinstance(data[key], str) or not Path(data[key]).is_absolute():
            raise ValueError(key + ' must be an absolute path; identical model path on every rank')
    address, sep, port = str(data['dist_init_addr']).rpartition(':')
    if not sep or address != data['fabric_ips'][0] or not port.isdigit() or not 1 <= int(port) <= 65535:
        raise ValueError('dist_init_addr must be rank-0 fabric IPv4:port (without tcp://)')
    return data


def load():
    if not CONFIG_PATH.exists():
        return None
    return validate(json.loads(CONFIG_PATH.read_text()))


CONFIG = load()


def require():
    if CONFIG is None:
        raise RuntimeError('Missing cluster inventory: set DSV41_CLUSTER_CONFIG or populate cluster.local.json from cluster.example.json')
    return CONFIG
