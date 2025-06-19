import torch
import torch.distributed as dist
from torch.optim import Optimizer
from typing import Type, Any, List, Tuple

class OptimizerStateSharding(Optimizer):
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.optimizer_cls = optimizer_cls
        self.kwargs = kwargs
        self.inner_optimizer = None

        self.param_to_rank = {}

        # 父類初始化，延遲建立 inner_optimizer
        super().__init__(params, defaults={})

    def step(self, closure=None, **kwargs):
        self.inner_optimizer.step(closure, **kwargs)

        # 同步參數
        for param, src in self.param_to_rank.items():
            dist.broadcast(param.data, src=src)

    def add_param_group(self, param_group):
        params = list(param_group['params'])

        split_indices = self._compute_split_indices(len(params))

        # 維護 param_to_rank
        self._update_param_to_rank(params, split_indices)

        # 提取本 rank 的 local params
        local_start, local_end = split_indices[self.rank]
        local_params = params[local_start:local_end]

        if local_params:
            new_group = dict(param_group)
            new_group['params'] = local_params

            if self.inner_optimizer is None:
                self.inner_optimizer = self.optimizer_cls([new_group], **self.kwargs)
            else:
                self.inner_optimizer.add_param_group(new_group)

            super().add_param_group(new_group)

    def _compute_split_indices(self, num_params: int) -> List[Tuple[int, int]]:
        params_per_rank = num_params // self.world_size
        remainder = num_params % self.world_size

        split_indices = []
        start_idx = 0
        for r in range(self.world_size):
            n = params_per_rank + (1 if r < remainder else 0)
            split_indices.append((start_idx, start_idx + n))
            start_idx += n

        return split_indices

    def _update_param_to_rank(self, params: List[torch.nn.Parameter], split_indices: List[Tuple[int, int]]):
        for r, (start, end) in enumerate(split_indices):
            for p in params[start:end]:
                self.param_to_rank[p] = r
