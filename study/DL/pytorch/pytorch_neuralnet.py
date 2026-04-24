import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

print(0b100)  # 二进制整数
print(0o100)  # 八进制整数
print(100)    # 十进制整数
print(0x100)  # 十六进制整数
# 定义两层神经网络
class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNet, self).__init__()
        # 对应鱼书的 Affine1 + Relu
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            # 对应鱼书的 Affine2
            nn.Linear(hidden_size, output_size)
        )
        # 注意：PyTorch 的 CrossEntropyLoss 内部集成了 Softmax，
        # 所以在网络末尾不需要显式添加 Softmax 层。

    def forward(self, x):
        # 展平图像 (batch_size, 1, 28, 28) -> (batch_size, 784)
        x = x.view(x.size(0), -1)
        return self.layers(x)


# 1. 配置超参数
input_size = 784
hidden_size = 50
output_size = 10
batch_size = 100
learning_rate = 0.1
epochs = 10  # 鱼书里的 iters_num 很大，这里用 epoch 概念更常用

# 2. 准备数据 (MNIST)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 3. 初始化模型、损失函数和优化器
model = TwoLayerNet(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()  # 对应 SoftmaxWithLoss
optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # 对应 SGD 更新参数

# 4. 训练循环
for epoch in range(epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        print(batch_idx)
        # 梯度清零 (对应鱼书里每一步计算新梯度的过程)
        optimizer.zero_grad()

        # 前向传播 (Forward)
        output = model(data)
        loss = criterion(output, target)

        # 反向传播 (Backward) - PyTorch 自动完成所有求导
        loss.backward()

        # 更新参数 (Update)
        optimizer.step()

    # 5. 计算精度 (每个 epoch 结束后)
    model.eval()
    correct = 0
    with torch.no_grad():  # 测试时不计算梯度，节省内存
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = correct / len(test_dataset)
    print(f"Epoch {epoch + 1}: Test Accuracy: {accuracy:.4f}")
