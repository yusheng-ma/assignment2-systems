import torch
import torch.distributed as dist
from torch.optim import Optimizer
from typing import Type, Any

class OptimizerStateSharding(Optimizer):
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        all_params = list(params)
        self._all_params = all_params
        self._param_to_idx = {id(p): i for i, p in enumerate(all_params)}

        shard_params = all_params[self.rank::self.world_size]
        self.wrapped_optimizer = optimizer_cls(shard_params, **kwargs)
        super().__init__(shard_params, defaults={})

    def step(self, closure=None, **kwargs):
        loss = self.wrapped_optimizer.step(closure=closure, **kwargs)

        # 只需要一次 broadcast，每個 rank 都調用這行
        for p in self._all_params:
            owner = self._param_to_rank(p)
            dist.broadcast(p.data, src=owner)

        return loss

    def add_param_group(self, param_group: dict[str, Any]):
        params = param_group['params']
        if isinstance(params, torch.Tensor):
            params = [params]

        new_params = []
        for p in params:
            if id(p) not in self._param_to_idx:
                new_params.append(p)
                self._all_params.append(p)
                self._param_to_idx[id(p)] = len(self._all_params) - 1

        shard_params = [p for p in new_params if self._param_to_rank(p) == self.rank]

        if shard_params:
            new_group = dict(param_group)
            new_group['params'] = shard_params
            self.wrapped_optimizer.add_param_group(new_group)

    def _param_to_rank(self, p):
        idx = self._param_to_idx[id(p)]
        return idx % self.world_size
