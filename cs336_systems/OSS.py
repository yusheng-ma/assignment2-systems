import torch
from torch.optim import Optimizer
from typing import Type, Any

class OptimizerStateSharding(Optimizer):
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        params = list(params)
        self.inner_optimizer = optimizer_cls(params, **kwargs)

        super().__init__(params, defaults={}) # params cant be empty, we give full params to fake in add_param_group

    def step(self, closure=None, **kwargs):
        self.inner_optimizer.step(closure, **kwargs)
        
    def add_param_group(self, param_group):
        # update father
        # self.param_groups = self.inner_optimizer.param_groups
        super().add_param_group(param_group)