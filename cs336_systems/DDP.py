import torch
import torch.nn as nn
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

class DDPIndividualParameters(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.handles = []
        self.world_size = dist.get_world_size()

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook( # Note that, unlike other autograd hooks, this hook operates on the tensor that requires grad and not the grad itself. The hook can in-place modify and access its Tensor argument, including its .grad field.
                    lambda param: self.handles.append((
                        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True),
                        param.grad
                    ))
                )

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle, grad in self.handles:
            handle.wait()
            grad.div_(self.world_size)
        self.handles = []


class DDPBucketed(nn.Module):
    def __init__(self, module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
        self.handles = []
        self.world_size = dist.get_world_size()

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        
        # bucket state
        self.current_bucket_grads = []
        self.current_bucket_bytes = 0

        for param in reversed(list(self.module.parameters())):
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._hook)
    
    def _hook(self, param):
        grad = param.grad
        if grad is None:
            return
        
        grad_bytes = grad.numel() * grad.element_size()

        # 如果加上這個梯度會超出 bucket size，先 flush
        if self.current_bucket_bytes + grad_bytes > self.bucket_size_bytes:
            self._flush_bucket()

        self.current_bucket_grads.append(grad)
        self.current_bucket_bytes += grad_bytes

        # 如果剛好等於 bucket size，直接 flush
        if self.current_bucket_bytes == self.bucket_size_bytes:
            self._flush_bucket()

    def _flush_bucket(self):
        if not self.current_bucket_grads:
            return

        flat = _flatten_dense_tensors(self.current_bucket_grads)
        handle = dist.all_reduce(flat, op=dist.ReduceOp.SUM, async_op=True)
        self.handles.append((handle, flat, list(self.current_bucket_grads)))

        # 重置 bucket
        self.current_bucket_grads = []
        self.current_bucket_bytes = 0

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        self._flush_bucket()

        for handle, flat, grads in self.handles:
            handle.wait()
            flat.div_(self.world_size)
            synced_grads = _unflatten_dense_tensors(flat, grads)
            for grad, synced in zip(grads, synced_grads):
                grad.copy_(synced)

        self.handles = []
