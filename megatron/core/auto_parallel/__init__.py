import os
import json
import datetime
import threading
import torch

KV_STORE = None
SEARCH_CACHE_PATH = None
MODULE_PATTERN = 'PP{}_TP{}_DP{}_CP{}_MBS{}_RECOMPUTE{}_MODULE.json'

class SingletonType(type):
    single_lock = threading.RLock()

    def __call__(cls, *args, **kwargs):
        with SingletonType.single_lock:
            if not hasattr(cls, "_instance"):
                cls._instance = super(SingletonType, cls).__call__(*args, **kwargs)
        return cls._instance

def get_cache_path():
    global SEARCH_CACHE_PATH
    if SEARCH_CACHE_PATH is None:
        SEARCH_CACHE_PATH = os.getcwd() + os.sep + 'autoparallel_temp_cache' + os.sep
        try:
            os.makedirs(SEARCH_CACHE_PATH, exist_ok=True)
            print(f"Create cache: {SEARCH_CACHE_PATH}")
        except Exception:
            print(f'Create cache directory failed')
            SEARCH_CACHE_PATH = os.getcwd()
    return SEARCH_CACHE_PATH

def analyse_module_profile(profile_file, key):
    if key != 'step_time':
        raise AssertionError('key[{}] not supported. Only step_time is supported.'.format(key))
    if not os.path.exists(path=profile_file):
        return float('inf')
    with open(profile_file, 'r') as file:
        try:
            content = file.read()
            content = json.loads(content)
            return float(content.get(key))
        except Exception:
            return float('inf')

def set_kv_store(args):
    global KV_STORE
    if args.node_rank == 0:
        KV_STORE = torch.distributed.TCPStore(
            host_name=args.master_addr,
            port=int(args.master_port) + 2,
            world_size=args.nnodes,
            is_master=True,
            timeout=datetime.timedelta(seconds=30)
        )
    else:
        KV_STORE = torch.distributed.TCPStore(
            host_name=args.master_addr,
            port=int(args.master_port) + 2,
            world_size=args.nnodes,
            is_master=False
        )

def get_kv_store():
    global KV_STORE
    if KV_STORE is None:
        raise AssertionError('KV_STORE must be initialized')
    return KV_STORE