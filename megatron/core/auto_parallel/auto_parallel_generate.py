import itertools
from collections import defaultdict, OrderedDict
import torch
import json
import time
import math
from megatron.training.global_vars import get_args

from megatron.core.auto_parallel import set_kv_store
from megatron.core.auto_parallel.auto_parallel_profiling import DistributedPerformanceProfiler

def extract_params(args):
    params={}
    tp_match = args.tensor_model_parallel_size
    pp_match = args.pipeline_model_parallel_size
    cp_match = args.context_parallel_size
    npus_match = args.nproc_per_node
    nnodes_match = args.nnodes
    num_layers_match = args.num_layers
    hidden_size_match = args.hidden_size
    ffn_hidden_size_match = args.ffn_hidden_size
    num_attention_heads_match = args.num_attention_heads
    num_query_groups_match = args.num_query_groups
    seq_length_match = args.seq_length
    micro_batch_size_match = args.micro_batch_size
    global_batch_size_match = args.global_batch_size
    if args.model_type == "llama":
        if args.vocab_size is not None:
            padded_vocab_size_match = args.vocab_size
        else:
            padded_vocab_size_match = 32000
    elif args.model_type == "baichuan":
        if args.vocab_size is not None:
            padded_vocab_size_match = args.vocab_size
        else:
            padded_vocab_size_match = 125696
    elif args.model_type == "qwen":
        if args.vocab_size is not None:
            padded_vocab_size_match = args.vocab_size
        else:
            padded_vocab_size_match = 152064
    elif args.model_type == "mistral":
        if args.vocab_size is not None:
            padded_vocab_size_match = args.vocab_size
        else:
            padded_vocab_size_match = 32000
    else:
        raise ValueError("Invalid model type, now just only support llama, gpt, baichuan and qwen")
    recompute_match = args.recompute_granularity
    npu_memory = args.gpu_peak_memory
    num_layer_list_match = args.num_layer_list
    params['tp'] = int(tp_match) if tp_match else 1
    params['pp'] = int(pp_match) if pp_match else 1
    params['cp'] = int(cp_match) if cp_match else 1
    params['npus_per_node'] = int(npus_match) if npus_match else 1
    params['nnodes'] = int(nnodes_match) if nnodes_match else 1
    params['num_layers'] = int(num_layers_match) if num_layers_match else 32
    params['hidden_size'] = int(hidden_size_match) if hidden_size_match else 4096
    params['ffn_hidden_size'] = int(ffn_hidden_size_match) if ffn_hidden_size_match else 11008
    params['num_attention_heads'] = int(num_attention_heads_match) if num_attention_heads_match else 32
    params['num_query_groups'] = int(num_query_groups_match) if num_query_groups_match else params['num_attention_heads']
    params['seq_length'] = int(seq_length_match) if seq_length_match else 2048
    params['micro_batch_size'] = int(micro_batch_size_match) if micro_batch_size_match else 1
    params['global_batch_size'] = int(global_batch_size_match) if global_batch_size_match else 1
    params['vocab_size'] = int(padded_vocab_size_match)
    params['recompute'] = recompute_match if recompute_match else None
    params['npu_mem'] = npu_memory if npu_memory else 64
    if num_layer_list_match:
        params['num_layer_list'] = [int(x) for x in num_layer_list_match .split(',')]
    else:
        base_layers = params['num_layers'] // params['pp']
        params['num_layer_list'] = [base_layers] * params['pp']
    total_npus = params['npus_per_node'] * params['nnodes']
    params['dp'] = total_npus // (params['tp'] * params['pp'] * params['cp'])
    return params

def state_mem(params, dp, tp, cp, layers_in_stage, emb_layer, last_stage):
    hidden = params['hidden_size']
    ffn_h = params['ffn_hidden_size']
    num_head = params['num_attention_heads']
    kv_head = params['num_query_groups'] 
    vocab = params['vocab_size']
    beta = 2 + 4 + 12 / (dp * cp) # use-distributed-optimizer equal to use zero stage-1
    alpha_shard = 1 + kv_head / num_head + 1.5 * ffn_h / hidden
    alpha_unshard = 1 / hidden
    base = 2 * layers_in_stage * (hidden ** 2) * (alpha_shard / tp + alpha_unshard)
    if emb_layer:
        base += hidden * vocab / tp          # embedding weights
    if last_stage:
        base += hidden                       # final RMSNorm weight
        base += hidden * vocab / tp          # LM head state memory is zero, because shared it with embedding, but considering void communication need to add.
    return beta * base                       # bytes

def act_mem(params, tp, cp, seq, mb, layers_in_stage, micro_parallel, first_stage, last_stage, pp, recompute):
    hidden = params['hidden_size']
    ffn_h = params['ffn_hidden_size']
    num_head = params['num_attention_heads']
    kv_head = params['num_query_groups']
    vocab = params['vocab_size']
    gamma = 12 + 4 * kv_head / num_head + 8 * ffn_h / hidden
    chi = 4 * (1 + vocab / hidden)
    layer_term = gamma * layers_in_stage * micro_parallel
    if recompute: 
        boundary_act = micro_parallel * (2 * mb * seq * hidden) / (tp * cp)
        emb_act = pp * (2 * seq * mb * hidden) / (tp * cp) if first_stage else 0
        lm_head_act = (2 * seq * mb * hidden + 4 * seq * mb * vocab) / (tp * cp) if last_stage else 0
        activation_memory = boundary_act + emb_act + lm_head_act
        return activation_memory
    else:
        term = layer_term
        emb_act = pp * (2 * seq * mb * hidden) / (tp * cp) if first_stage else 0
        lm_head_act = chi * seq * mb * hidden / (tp * cp) if last_stage else 0
        activation_memory = seq * mb * hidden * term / (tp * cp) + emb_act + lm_head_act
        return activation_memory

def get_layer_distribution(num_layers, pp):
    base = num_layers // pp
    rem = num_layers % pp
    return [base + 1 if i >= pp - rem else base for i in range(pp)]

def extract_search_space(total_npus, params):
    # generate search space according to the total npus and model params
    dp_candidates = generate_power_of_two_values(lambda x: x <= total_npus)
    pp_candidates = generate_power_of_two_values(lambda x: x <= min(total_npus, params['num_layers']))
    tp_candidates = generate_power_of_two_values(lambda x: x <= min(total_npus, params['hidden_size'] // 16))
    cp_candidates = generate_power_of_two_values(lambda x: x <= total_npus)
    def maybe_add(cur_val, candidates):
        if cur_val not in candidates and cur_val <= total_npus:
            candidates.append(cur_val)
    maybe_add(params['dp'], dp_candidates)
    maybe_add(params['pp'], pp_candidates)
    maybe_add(params['tp'], tp_candidates)
    maybe_add(params['cp'], cp_candidates)
    return (
        sorted(dp_candidates),
        sorted(pp_candidates),
        sorted(tp_candidates),
        sorted(cp_candidates),
    )

def generate_power_of_two_values(condition_func):
    values = []
    for i in range(20):  # 上限防止死循环
        val = 2 ** i
        if condition_func(val):
            values.append(val)
        else:
            break
    return values

def get_microbatch_candidates(current_mb):
    mb_candidates = [current_mb]
    for factor in [2, 4]:
        if current_mb // factor >= 1:
            mb_candidates.append(current_mb // factor)
        mb_candidates.append(current_mb * factor)
    return sorted(list(set(mb_candidates)))

def compute_max_memory(params, dp, tp, cp, pp, mb, recompute, npu_limit_bytes):
    max_memory = 0.0
    micro_parallel = [pp - i for i in range(pp)]
    layer_distribution = get_layer_distribution(params['num_layers'], pp)
    for i, layers_in_stage in enumerate(layer_distribution):
        is_first = (i == 0)
        is_last = (i == pp - 1)
        st = state_mem(params, dp, tp, cp, layers_in_stage, emb_layer=is_first, last_stage=is_last)
        act = act_mem(params, tp, cp, params['seq_length'], mb, layers_in_stage,
                      micro_parallel=micro_parallel[i],
                      first_stage=is_first, last_stage=is_last,
                      pp=pp, recompute=recompute)
        total_memory = st + act
        max_memory = max(max_memory, total_memory)
    return max_memory <= npu_limit_bytes

def print_extraction_info(params, total_npus):
    print("Extracted parameters:")
    print(f"Total number of model layers: {params['num_layers']}")
    print(f"Hidden layer size: {params['hidden_size']}")
    print(f"FFN hidden layer size: {params['ffn_hidden_size']}")
    print(f"Number of attention heads: {params['num_attention_heads']}")
    print(f"Query group number: {params['num_query_groups']}")
    print(f"Sequence length: {params['seq_length']}")
    print(f"Vocab size: {params['vocab_size']}")
    print(f"Total device number: {total_npus}")
    print(f"Memory limit per device: {params['npu_mem']} GB")
    print()

def print_search_space(dp, pp, tp, cp, mb, recompute):
    print("Search space:")
    print(f"DP = {dp}, PP = {pp}, TP = {tp}, CP = {cp}")
    print(f"Microbatch sizes = {mb}")
    print(f"Recompute options = {recompute}")
    print("\nStarting configuration search...\n")
        
def print_all_configs(configs):
    print("\nAll viable configurations:")
    for idx, config in enumerate(configs):
        print(f"{idx+1}. DP={config['dp']}, PP={config['pp']}, TP={config['tp']}, "
              f"CP={config['cp']}, MBS={config['mb']}, Recompute={config['recompute']}")
        
def group_configs_by_axes(configs):
    bucket = defaultdict(lambda: {True: set(), False: set()})
    for c in configs:
        key = (c['dp'], c['pp'], c['tp'], c['cp'])
        bucket[key][bool(c['recompute'])].add(int(c['mb']))
    ordered = OrderedDict()
    for key in sorted(bucket.keys()):
        ordered[key] = {
            True: sorted(bucket[key][True]),
            False: sorted(bucket[key][False])
        }
    return ordered

def print_grouped_configs(configs):
    grouped = group_configs_by_axes(configs)
    idx = 1
    for (dp, pp, tp, cp), m in grouped.items():
        print(f"{idx}. DP={dp}, PP={pp}, TP={tp}, CP={cp}")
        if m[True]:
            print(f" - Recompute=True, MBS={m[True]}")
        if m[False]:
            print(f" - Recompute=False, MBS={m[False]}")
        idx += 1

def generate_all_viable_configs(args):
    params = extract_params(args)
    total_npus = params['dp'] * params['pp'] * params['tp'] * params['cp']
    npu_limit_bytes = params['npu_mem'] * (1024 ** 3) * 0.85
    print_extraction_info(params, total_npus)
    dp_list, pp_list, tp_list, cp_list = extract_search_space(total_npus, params)
    mb_list = get_microbatch_candidates(params['micro_batch_size'])
    recompute_list = [True, False]
    print_search_space(dp_list, pp_list, tp_list, cp_list, mb_list, recompute_list)
    configs = []
    for dp, pp, tp, cp, recompute in itertools.product(dp_list, pp_list, tp_list, cp_list, recompute_list):
        if dp * pp * tp * cp != total_npus:
            continue
        for mb in mb_list:
            if compute_max_memory(params, dp, tp, cp, pp, mb, recompute, npu_limit_bytes):
                configs.append({
                    "dp": dp,
                    "pp": pp,
                    "tp": tp,
                    "cp": cp,
                    "mb": mb,
                    "recompute": recompute,
                })
    # print_grouped_configs(configs)
    return configs

def filter_configs_by_tp_num_query_groups(configs):
    args = get_args()
    try:
        ngroups = int(getattr(args, 'num_query_groups', 0) or 0)
    except Exception:
        ngroups = 0
    if ngroups <= 0:
        return configs
    filtered = [c for c in configs if int(c.get('tp', 1)) <= ngroups]
    removed = len(configs) - len(filtered)
    if removed > 0:
        print(f"[Filter] Removed {removed} configs: tp > num_query_groups ({ngroups})")
    return filtered

def filter_configs_by_head_divisibility(configs):
    args = get_args()
    n_heads = int(getattr(args, 'num_attention_heads', 0) or 0)
    if n_heads <= 0:
        return configs
    def ok(c):
        cp = int(c.get('cp', 1))
        tp = int(c.get('tp', 1))
        return (cp * tp) > 0 and (n_heads % (cp * tp) == 0)
    filtered = [c for c in configs if ok(c)]
    removed = len(configs) - len(filtered)
    if removed > 0:
        print(f"[Filter] Removed {removed} configs: cp*tp does not divide num_attention_heads ({n_heads})")
    return filtered

def monitor_train_task():
    while True:
        message = torch.tensor([0 for _ in range(8)], dtype=torch.int)
        torch.distributed.broadcast(message, 0)
        task_type = message[-1].item()
        gbs = message[-2].item()
        config_list = [m.item() for m in message[:-2]]
        if task_type == -1:
            break
        elif task_type == 2:
            # Convert list format back to dict format for DistributedPerformanceProfiler
            # New format: [pp, tp, dp, cp, mb, recompute]
            config_dict = {
                'pp': config_list[0], 
                'tp': config_list[1], 
                'dp': config_list[2], 
                'cp': config_list[3], 
                'mb': config_list[4],
                'recompute': bool(config_list[5])
            }
            profiler = DistributedPerformanceProfiler()
            profiler.min_global_batch_size = gbs if gbs > 0 else None
            profiler.launch(config_dict)

def export_results(config):
    results = {}
    results['optimal_parallel_strategy'] = {}
    results['optimal_parallel_strategy']['pipeline-model-parallel-size'] = config['pp']
    results['optimal_parallel_strategy']['tensor-model-parallel-size'] = config['tp']
    results['optimal_parallel_strategy']['data-parallel-size'] = config['dp']
    results['optimal_parallel_strategy']['micro-batch-size'] = config['mb']
    results['optimal_parallel_strategy']['recompute'] = config['recompute']
    if config['cp'] > 1:
        results['optimal_parallel_strategy']['context-parallel-algo'] = 'ulysses_cp_algo'
        results['optimal_parallel_strategy']['context-parallel-size'] = config['cp']
    return json.dumps(results)

def search_optimal_parallel_configurations(args):
    set_kv_store(args)
    init_method = 'tcp://{}:{}'.format(args.master_addr, int(args.master_port) + 1)
    torch.distributed.init_process_group(
        backend=torch.distributed.Backend.GLOO,
        init_method=init_method,
        rank=args.node_rank,
        world_size=args.nnodes
    )
    if args.node_rank == 0:
        start_time = time.time()
        configs = generate_all_viable_configs(get_args())
        configs = filter_configs_by_tp_num_query_groups(configs)
        configs = filter_configs_by_head_divisibility(configs)
        from megatron.core.auto_parallel.auto_parallel_optimizer import HeuristicSearcher
        print(f"Total viable configurations: {len(configs)}")
        best_config, step_time = HeuristicSearcher().run(get_args(), configs)
        if best_config is None or step_time is None:
            print("No valid configuration found (all OOM)")
            torch.distributed.broadcast(torch.tensor([-1 for _ in range(7)], dtype=torch.int), 0)
            print(f"Total search time: {time.time() - start_time:.2f} seconds")
            return 
        print(f"Best configuration: {best_config}, step_time: {step_time}")
        torch.distributed.broadcast(torch.tensor([-1 for _ in range(7)], dtype=torch.int), 0)
        results = export_results(best_config)
        print(f"Results: {results}")
        print(f"Total search time: {time.time() - start_time:.2f} seconds")
    else:
        monitor_train_task()
    

