import ast
import math
from typing import Dict, List, Tuple, Optional
from megatron.core.auto_parallel.auto_parallel_profiling import (
    DistributedPerformanceProfiler
)
from megatron.core.auto_parallel import get_cache_path, MODULE_PATTERN
import os
import json

def _profile_file_path(cfg: Dict) -> str:
    return (get_cache_path() + MODULE_PATTERN).format(
        cfg['pp'], cfg['tp'], cfg['dp'], cfg['cp'], cfg['mb'], cfg['recompute']
    )
    
def _is_oom_or_invalid_profile(cfg: Dict) -> bool:
    path = _profile_file_path(cfg)
    try:
        if not os.path.exists(path):
            return True
        with open(path, 'r') as f:
            obj = json.load(f)
        return 'transformer_act_mem' not in obj
    except Exception:
        return True
    
class SearchByGreyBox:
    def __init__(self):
        self.config_performances = {}

    def search(self, args, search_spaces):
        profiler = DistributedPerformanceProfiler()
        profiler.set_min_global_batch_size(search_spaces)
        for config_dict in search_spaces:
            print(f"Profiling configuration: {config_dict}", flush=True)
            duration_time = profiler.launch(config_dict)
            if _is_oom_or_invalid_profile(config_dict):
                print(f"[GreyBox] OOM profile detected, skipping: {config_dict}")
                continue
            self.config_performances[duration_time] = config_dict
            print(f"Configuration {config_dict} step_time: {duration_time}", flush=True)
        if not self.config_performances:
            print(f"[GreyBox] No valid (non-OOM) profiles found")
            return None, None
        min_key = min(self.config_performances.keys())
        best_config = self.config_performances.get(min_key)
        print(f"Best configuration: {best_config} with step_time: {min_key}", flush=True)
        return best_config, min_key
    
class HeuristicSearcher:
    def __init__(self, lambda_penalty: float = 0.1, eps: float = 1e-6):
        self.profiler = DistributedPerformanceProfiler()
        self.lambda_penalty = float(lambda_penalty)
        self.eps = float(eps)
        
    @staticmethod
    def _key(cfg: Dict) -> Tuple:
        return (cfg['dp'], cfg['pp'], cfg['tp'], cfg['cp'], cfg['mb'], cfg['recompute'])
    
    @staticmethod
    def _dims() -> List[str]:
        return ['dp', 'pp', 'tp', 'cp']
    
    @staticmethod
    def _has_variation(configs: List[Dict], dim: str) -> bool:
        vals = {c[dim] for c in configs}
        return len(vals) > 1

    def _split_by_recompute(self, configs: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        no_rec = [c for c in configs if not c['recompute']]
        with_rec = [c for c in configs if c['recompute']]
        return no_rec, with_rec
    
    def _balance_ratio(self, cfg: Dict) -> float:
        vals = [max(1, int(cfg[d])) for d in self._dims()]
        mn = max(1, min(vals))
        mx = max(vals)
        return mx / mn if mn > 0 else float('inf')
    
    def _choose_base(self, configs: List[Dict]) -> Dict:
        best = sorted(
            configs, 
            key = lambda c: (self._balance_ratio(c), -int(c['mb']), -int(c['dp']), 
                             -int(c['pp']), -int(c['tp']),
                            -int(c['cp']))
        )[0]
        print(f"[Heuristic] Base config selected: {best}", flush=True)
        return best
    
    @staticmethod
    def _log2(v: float) -> float:
        return math.log2(max(1.0, float(v)))
    
    def _purity(self, base: Dict, cand: Dict, target: str) -> float:
        d_target = abs(self._log2(cand[target]) - self._log2(base[target]))
        d_others = 0.0
        for d in self._dims():
            if d == target:
                continue
            d_others += abs(self._log2(cand[d]) - self._log2(base[d]))
        return d_target / (d_others + self.eps)
    
    def _find_anchor_for_dim(self, configs: List[Dict], base: Dict, target: str) -> Optional[Dict]:
        candidates = [c for c in configs if self._key(c) != self._key(base) and c[target] != base[target]]
        if not candidates:
            return None
        ranked = sorted(
            candidates,
            key=lambda c: (self._purity(base, c, target), int(c[target]), int(c['mb'])),
            reverse=True
        )
        anchor = ranked[0]
        print(f"[Heuristic] Anchor for {target}: {anchor} (purity={self._purity(base, anchor, target):.3f})", 
              flush=True)
        return anchor
    
    def _profile_configs(self, configs: List[Dict]) -> Dict[Tuple, float]:
        perf: Dict[Tuple, float] = {}
        seen = set()
        uniq: List[Dict] = []
        for c in configs:
            k = self._key(c)
            if k in seen:
                continue
            seen.add(k)
            uniq.append(c)
        print(f"[Heuristic] Profiling {len(uniq)} configs...", flush=True)
        self.profiler.set_min_global_batch_size(uniq)
        for c in uniq:
            t = self.profiler.launch(c)
            if _is_oom_or_invalid_profile(c):
                print(f"[Heuristic] OOM profile detected, skipping: {c}")
                continue
            if t is None:
                continue
            perf[self._key(c)] = float(t)
            print(f"[Heuristic] Profiled {c} -> step_time={t}", flush = True)
        return perf
    
    @staticmethod
    def _throughput_per_sample(step_time: float, mb: int) -> float:
        if step_time <= 0:
            return 0.0
        return 1.0 / (step_time * max(1, int(mb)))
    
    def _sensitivity_scores(
        self,
        base: Dict,
        anchors: Dict[str, Dict],
        perf: Dict[Tuple, float]
    ) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        base_t = perf.get(self._key(base), float('inf'))
        if base_t == float('inf') or base_t <= 0:
            print("[Heuristic] Warning: base performance missing; using neutral baseline.", flush=True)
            base_t = 1.0
        thr_base = self._throughput_per_sample(base_t, base['mb'])
        for dim, cfg in anchors.items():
            if cfg is None:
                continue
            t = perf.get(self._key(cfg), float('inf'))
            if t == float('inf') or t <= 0:
                continue
            thr = self._throughput_per_sample(t, cfg['mb'])
            d_target = abs(self._log2(cfg[dim]) - self._log2(base[dim]))
            d_others = 0.0
            for d in self._dims():
                if d == dim:
                    continue
                d_others += abs(self._log2(cfg[d]) - self._log2(base[d]))
            gain = (thr - thr_base) / (d_target + self.eps) - self.lambda_penalty * d_others
            scores[dim] = gain
            print(f"[Heuristic] Score {dim}: gain={gain:.6f}, thr_base={thr_base:.6f}, thr={thr:.6f},"
                  f"d_target={d_target:.3f}, d_others={d_others:.3f}", flush=True)
        return scores

    def _rank_dims(self, configs: List[Dict], scores: Dict[str, float]) -> List[str]:
        present = [d for d in self._dims() if self._has_variation(configs, d)]
        missing = [d for d in self._dims() if d not in present]
        ranked_present = sorted(present, key=lambda d: scores.get(d, -1e9), reverse=True)
        return ranked_present + missing

    def _greedy_select(self, configs: List[Dict], ranking: List[str]) -> Optional[Dict]:
        if not configs:
            return None
        candidates = list(configs)
        for dim in ranking:
            vals = [c[dim] for c in candidates]
            if not vals:
                continue
            max_val = max(vals)
            next_cands = [c for c in candidates if c[dim] == max_val]
            if next_cands:
                candidates = next_cands
        best = max(candidates, key=lambda c: (int(c['mb'])))
        return best

    def _pick_group(self, configs: List[Dict]) -> Tuple[List[Dict], bool]:
        no_rec, with_rec = self._split_by_recompute(configs)
        if no_rec:
            return no_rec, False
        return with_rec, True

    def run(self, args, all_configs: List[Dict]) -> Tuple[Optional[Dict], Optional[float]]:
        if not all_configs:
            print("[Heuristic] No viable configs (non-OOM) found.", flush=True)
            return None, None
        self.profiler.set_min_global_batch_size(all_configs)
        group, used_recompute = self._pick_group(all_configs)
        if not group:
            print("[Heuristic] No viable configs in any recompute group.", flush=True)
            return None, None
        print(f"[Heuristic] Working in recompute={used_recompute} group with {len(group)} configs.",
              flush=True)

        if len(group) <= 6:
            print("[Heuristic] Small search space, using GreyBox brute-force on this group.", flush=True)
            searcher = SearchByGreyBox()
            best, t = searcher.search(args, group)
            return best, t

        base = self._choose_base(group)
        anchors: Dict[str, Optional[Dict]] = {}
        for dim in self._dims():
            anchors[dim] = (
                self._find_anchor_for_dim(group, base, dim) 
                if self._has_variation(group, dim) 
                else None
            )

        to_profile: List[Dict] = [base] + [c for c in anchors.values() if c is not None]
        perf = self._profile_configs(to_profile)

        scores = self._sensitivity_scores(base, anchors, perf)
        ranking = self._rank_dims(group, scores)
        print(f"[Heuristic] Sensitivity ranking (high→low): {ranking}", flush=True)

        best_cfg = self._greedy_select(group, ranking)
        if best_cfg is None:
            print("[Heuristic] Greedy selection failed, fallback to max-mb in group.", flush=True)
            best_cfg = max(group, key=lambda c: int(c['mb']))

        step_time = perf.get(self._key(best_cfg))
        if step_time is None:
            step_time = self.profiler.launch(best_cfg)
            if _is_oom_or_invalid_profile(best_cfg):
                print(f"[Heuristic] Best candidate profile is OOM, giving up")
                return None, None

        print(f"[Heuristic] Selected best config: {best_cfg} with step_time={step_time}", flush=True)
        return best_cfg, step_time