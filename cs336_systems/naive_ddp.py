import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist

# 简单模型
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 10)
        self.linear2 = nn.Linear(10, 5)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def naive_ddp(rank, world_size, ref_model_state):
    setup(rank, world_size)

    torch.manual_seed(0)
    model = ToyModel()
    model.load_state_dict(copy.deepcopy(ref_model_state))  # 每个进程同样初始化
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    # 模拟数据 (20, 10) 输入，(20, 5) 标签
    data = torch.randn(20, 10)
    target = torch.randn(20, 5)

    local_bs = 20 // world_size
    local_data = data[rank * local_bs : (rank + 1) * local_bs]
    local_target = target[rank * local_bs : (rank + 1) * local_bs]

    for epoch in range(5):
        optimizer.zero_grad()
        output = model(local_data)
        loss = loss_fn(output, local_target)
        loss.backward()

        # Naive all-reduce 每个参数的梯度
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= world_size

        optimizer.step()

    # 收集 rank 0 的最终参数
    params = [p.detach().cpu() for p in model.parameters()]
    if rank == 0:
        torch.save(params, "ddp_params.pt")

    cleanup()

def single_process_training(ref_model_state):
    torch.manual_seed(0)
    model = ToyModel()
    model.load_state_dict(copy.deepcopy(ref_model_state))
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    data = torch.randn(20, 10)
    target = torch.randn(20, 5)

    for epoch in range(5):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

    return [p.detach().cpu() for p in model.parameters()]

def main():
    world_size = 2

    # 初始模型参数
    torch.manual_seed(0)
    ref_model = ToyModel()
    ref_model_state = ref_model.state_dict()

    # 启动 DDP
    mp.spawn(naive_ddp, args=(world_size, ref_model_state), nprocs=world_size, join=True)

    # 单进程训练
    single_params = single_process_training(ref_model_state)

    # 加载 DDP 训练的参数
    ddp_params = torch.load("ddp_params.pt")

    # 验证
    for sp, dp in zip(single_params, ddp_params):
        assert torch.allclose(sp, dp, atol=1e-6), "Parameters mismatch!"
    print("✅ Naive DDP training matches single-process training!")

if __name__ == "__main__":
    main()
