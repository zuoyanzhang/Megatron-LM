import os
import sys
import subprocess
import signal
import threading
import copy
import time
import json
import stat
import torch
from megatron.training.global_vars import set_args, get_args
from megatron.core.auto_parallel import (
    get_cache_path,
    get_kv_store,
    analyse_module_profile,
    MODULE_PATTERN,
    SingletonType
)

class BaseLaunch:
    def __init__(self):
        self.old_args = None
        self.min_global_batch_size = None
    
    def calculate_min_gbs_from_profiling_configs(self, profiling_configs):
        if not profiling_configs:
            return None
        min_gbs = 0
        for config in profiling_configs:
            config_gbs = config['mb'] * config['dp']
            min_gbs = max(min_gbs, config_gbs)
        return min_gbs if min_gbs > 0 else None
    
    def set_min_global_batch_size(self, profiling_configs):
        self.min_global_batch_size = self.calculate_min_gbs_from_profiling_configs(profiling_configs)
        if self.min_global_batch_size:
            print(f"[Profiling] Calculated min global batch size: {self.min_global_batch_size}", flush=True)

    def launch(self, config):
        def update_or_append_param(argv: list, key, value=None):
            if not value:
                argv.append(key)
                return

            if key in argv:
                argv[argv.index(key) + 1] = value
            else:
                argv.extend([key, value])

        def remove_param(argv: list, key, has_value=False):
            if key in argv:
                pos = argv.index(key)
                argv.pop(pos)
                if has_value:
                    argv.pop(pos)

        def monitor_exit(process):
            while True:
                exit_flag = get_kv_store().get("exit_flag")
                if int(exit_flag) == 1:
                    try:
                        process_group_id = os.getpgid(process.pid)
                        os.killpg(process_group_id, signal.SIGKILL)
                        break
                    except ProcessLookupError:
                        break
                time.sleep(60)

        args = get_args()
        argv: list = sys.argv[1:]
        gbs_to_use = self.min_global_batch_size if self.min_global_batch_size else args.global_batch_size
        update_or_append_param(argv, '--eval-iters', '0')
        update_or_append_param(argv, '--train-iters', '4')
        update_or_append_param(argv, '--global-batch-size', str(gbs_to_use))
        update_or_append_param(argv, '--num-layers', str(args.num_layers))
        update_or_append_param(argv, '--pipeline-model-parallel-size', str(args.pipeline_model_parallel_size))
        update_or_append_param(argv, '--tensor-model-parallel-size', str(args.tensor_model_parallel_size))
        update_or_append_param(argv, '--micro-batch-size', str(args.micro_batch_size))
        update_or_append_param(argv, '--sequence-parallel')
        if args.profile_operator:
            update_or_append_param(argv, '--profile-operator')
        if args.profile_memory:
            update_or_append_param(argv, '--profile-memory')
        if args.module_profile_path:
            update_or_append_param(argv, '--prof-file', str(args.module_profile_path))
        if hasattr(args, 'profile_ranks'):
            ranks = args.profile_ranks
            if isinstance(ranks, (list, tuple)):
                ranks_str = ','.join([str(rank) for rank in ranks])
            else:
                ranks_str = str(ranks)
            update_or_append_param(argv, '--profile-ranks', ranks_str)
        if hasattr(args, 'context_parallel_algo') and args.context_parallel_algo == 'ulysses_cp_algo':
            update_or_append_param(argv, '--context-parallel-algo', 'ulysses_cp_algo')
            update_or_append_param(argv, '--context-parallel-size', str(args.context_parallel_size))
        if hasattr(args, 'recompute_granularity') and args.recompute_granularity:
            update_or_append_param(argv, '--recompute-granularity', args.recompute_granularity)
            update_or_append_param(argv, '--recompute-method', args.recompute_method)
            update_or_append_param(argv, '--recompute-num-layers', str(args.recompute_num_layers))
        remove_param(argv, '--auto-parallel-strategy-generate') # Very important, remember to remove this param !!!
        command = [
            'torchrun', 
            '--nproc_per_node', str(args.nproc_per_node),
            '--nnodes', str(args.nnodes),
            '--node-rank', str(args.node_rank),
            '--master_addr', str(args.master_addr),
            '--master_port', str(args.master_port),
            str(sys.argv[0])
        ] + argv

        get_kv_store().set("exit_flag", "0")
        process = subprocess.Popen(command, shell=False, preexec_fn=lambda: os.setpgrp())
        monitor_thread = threading.Thread(target=monitor_exit, args=(process,))
        monitor_thread.start()
        process.wait()
        get_kv_store().set("exit_flag", "1")
        torch.distributed.barrier()

    def update_args(self, config):
        """Update args with config - only supports dict format now"""
        args = get_args()
        self.old_args = copy.deepcopy(args)

        # Only support dict format (legacy list format removed)
        args.pipeline_model_parallel_size = config['pp']
        args.tensor_model_parallel_size = config['tp']
        args.data_parallel_size = config['dp']
        args.context_parallel_size = config['cp']
        args.ulysses_degree_in_cp = 1  # Always 1 for ulysses_cp_algo
        args.micro_batch_size = config['mb']
        
        # Set recompute parameters
        if config['recompute']:
            args.recompute_granularity = 'full'
            args.recompute_method = 'block'
            args.recompute_num_layers = args.num_layers
        else:
            args.recompute_granularity = None
            args.recompute_method = None
            args.recompute_num_layers = None
            
        # Context parallel settings - only use ulysses_cp_algo
        if config['cp'] > 1:
            args.context_parallel_algo = 'ulysses_cp_algo'
            args.use_cp_send_recv_overlap = True

    def recover_args(self):
        set_args(self.old_args)

class DistributedPerformanceProfiler(BaseLaunch):
    def update_args(self, config):
        super().update_args(config)
        args = get_args()
        profile_args = (config['pp'], config['tp'], config['dp'], 
                      config['cp'], config['mb'], config['recompute'])
        args.module_profile_path = (get_cache_path() + MODULE_PATTERN).format(*profile_args)
        args.prof_file = args.module_profile_path
        args.profile_ranks = [0]

    def launch(self, config):
        self.update_args(config)
        args = get_args()
        if args.node_rank != 0:
            super().launch(config)
            super().recover_args()
            return None

        module_profile_path = get_args().module_profile_path
        if os.path.exists(module_profile_path):
            with open(module_profile_path, 'r') as f:
                obj = json.load(f)
                if 'transformer_act_mem' not in obj:
                    print(f"Found existing profile without transformer_act_mem",
                          f" treat as OOM")
                    super().recover_args()
                    return None
            super().recover_args()
            step_time = analyse_module_profile(module_profile_path, key='step_time')
            print(f"Found existing profile, step_time: {step_time}", flush=True)
            return step_time

        config_list = [config['pp'], config['tp'], config['dp'], config['cp'], config['mb'], config['recompute']]
        gbs = self.min_global_batch_size if self.min_global_batch_size else 0
        buffer = config_list + [gbs, 2]
        torch.distributed.broadcast(torch.tensor(buffer, dtype=torch.int), 0)
        
        # Measure training time as fallback
        start_time = time.time()
        super().launch(config)
        end_time = time.time()
        
        super().recover_args()
        
        # Try to get step_time from profile file first
        step_time = analyse_module_profile(module_profile_path, key='step_time')
        
        # If profile file doesn't exist or returns inf, use measured time as fallback
        if step_time == float('inf') or step_time == 0:
            total_time = end_time - start_time
            # Assuming 4 training iterations (as set in BaseLaunch)
            estimated_step_time = total_time / 4.0
            print(f"Profile file missing or invalid, using estimated step_time: {estimated_step_time}", flush=True)
            # Write the estimated time to the profile file for consistency
            try:
                os.makedirs(os.path.dirname(module_profile_path), exist_ok=True)
                with open(module_profile_path, 'w') as f:
                    json.dump({'step_time': estimated_step_time}, f)
                step_time = estimated_step_time
            except Exception as e:
                print(f"Failed to write profile file: {e}", flush=True)
                step_time = estimated_step_time
        else:
            print(f"Successfully read step_time from profile: {step_time}", flush=True)
            
        return step_time


class SimpleProfiler(metaclass=SingletonType):
    """Simplified profiler for measuring step_time only"""
    def __init__(self, args, warmup_step=2, stop_step=4):
        self.args = args
        self.warmup_step = warmup_step
        self.stop_step = stop_step
        self.curr_step = 0
        self.context = {'step_time': 0}

    def _normalize_allowed_ranks(self):
        if torch.distributed.is_initialized():
            world = torch.distributed.get_world_size()
        else:
            world = 1
        allowed = getattr(self.args, 'profile_ranks', None)
        if isinstance(allowed, int):
            if allowed == -1:
                return list(range(world))
            return [allowed]
        if isinstance(allowed, str):
            allowed = [int(x) for x in allowed.split(',') if x != '']
        if not allowed:
            return [0]
        return list(allowed)

    def should_profiling(self):
        allowed = self._normalize_allowed_ranks()
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        return (rank in allowed) and (self.warmup_step <= self.curr_step < self.stop_step)

    def hook_train_step(self, train_step):
        def custom_train_step(*args, **kwargs):
            start_time = time.time()
            result = train_step(*args, **kwargs)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            else:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
            step_time = time.time() - start_time
            
            if self.should_profiling():
                cur_step_time = self.context.get('step_time')
                cur_step_time += (step_time - cur_step_time) / (self.curr_step - self.warmup_step + 1)
                self.context['step_time'] = cur_step_time
                
            self.export_to_file()
            self.curr_step += 1
            return result
        return custom_train_step
    
    def export_to_file(self):
        if torch.distributed.is_initialized():
            cur_rank = torch.distributed.get_rank()
        else:
            cur_rank = 0
        allowed = self._normalize_allowed_ranks()
        path = getattr(self.args, 'prof_file', None) or getattr(self.args, 'module_profile_path', None)
        
        if path is None:
            print("[export] prof_file/module_profile_path missing; skip write", flush=True)
            return
            
        if cur_rank in allowed:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            modes = stat.S_IWUSR | stat.S_IRUSR
            with os.fdopen(os.open(path, flags, modes), 'w') as fout:
                fout.write(json.dumps(self.context))