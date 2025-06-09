import torch
import torch.nn as nn
import torch.nn.functional as F

# 定義模型
class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        print("fc1 output dtype:", x.dtype)
        x = self.relu(x)
        x = self.ln(x)
        print("LayerNorm output dtype:", x.dtype)
        x = self.fc2(x)
        print("fc2 (logits) output dtype:", x.dtype)
        return x

# 初始化模型與資料
device = "cuda"
model = ToyModel(in_features=5, out_features=3).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler()  # 用 torch.amp 而非 torch.cuda.amp

# 模擬輸入與標籤
inputs = torch.randn(8, 5, device=device)
targets = torch.randint(0, 3, (8,), device=device)

# 混合精度訓練區塊，完全使用 torch 的新版介面
with torch.autocast(device_type="cuda", dtype=torch.float16):
    outputs = model(inputs)
    loss = F.cross_entropy(outputs, targets)
    print("Loss dtype:", loss.dtype)

# 反向與更新
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# 查看參數的梯度 dtype
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name} grad dtype: {param.grad.dtype}")
