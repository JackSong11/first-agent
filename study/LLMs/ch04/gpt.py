from importlib.metadata import version

import matplotlib
import tiktoken
import torch

print("matplotlib version:", version("matplotlib"))
print("torch version:", version("torch"))
print("tiktoken version:", version("tiktoken"))
# 加载并确认版本

GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False  # Query-Key-Value bias
}
# 初始化定义需要的各种超参数

import torch
import torch.nn as nn


class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # 词嵌入层，将输入索引转换为词向量，词表大小由字典大小和特征维度决定。
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # 位置信息嵌入层，基于文本长度和特征维度生成位置信息。
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        # Dropout 层，用于随机丢弃一部分嵌入信息以减少过拟合。

        # 使用多个 Transformer 块（占位符）
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        # Transformer 模块的堆叠，模型核心部分。

        # 使用归一化层（占位符）
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        # 最终归一化层，用于调整特征分布。

        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
        # 输出层，将特征映射到词表分布，最终预测输出单词。

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        # 获取批次大小和序列长度。

        tok_embeds = self.tok_emb(in_idx)
        # 根据输入索引生成词嵌入。
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        # 生成对应的位置信息嵌入。

        x = tok_embeds + pos_embeds
        # 将词嵌入和位置信息嵌入相加。
        x = self.drop_emb(x)
        # 应用 Dropout 随机丢弃部分信息。
        x = self.trf_blocks(x)
        # 通过多个 Transformer 块处理特征。
        x = self.final_norm(x)
        # 应用最终的归一化层。
        logits = self.out_head(x)
        # 将隐藏状态映射到词表分布，生成预测结果。
        return logits


class DummyTransformerBlock(nn.Module):
    # Transformer 块的占位类。
    def __init__(self, cfg):
        super().__init__()
        # 占位，实际模型应实现注意力机制和前馈网络。

    def forward(self, x):
        # 此块不执行任何操作，仅返回输入。
        return x


class DummyLayerNorm(nn.Module):
    # 归一化层的占位类。
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        # 参数用于模拟 LayerNorm 的接口。

    def forward(self, x):
        # 此层不执行任何操作，仅返回输入。
        return x


import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
# 召唤gpt大神
batch = []

txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
# 编码输入文本
batch = torch.stack(batch, dim=0)
# 按照横向来叠加两个向量
print(batch)
print(batch.shape)

torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)

logits = model(batch)
print("Output shape:", logits.shape)
print(logits)

torch.manual_seed(123)

# create 2 training examples with 5 dimensions (features) each
batch_example = torch.randn(2, 5)

layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
# 一个按照顺序执行的神经网络
# 具体: 全链接层跟,激活函数
out = layer(batch_example)
print(out)

mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)

print("Mean:\n", mean)
print("Variance:\n", var)

out_norm = (out - mean) / torch.sqrt(var)
# 执行归一化操作
print("Normalized layer outputs:\n", out_norm)

mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
print("Mean:\n", mean)  # 这里对归一化之后的值求均值，均值肯定为0。
print("Variance:\n", var)  # 这里对归一化之后的值求方差，方差肯定为1。

torch.set_printoptions(sci_mode=False)
print("Mean:\n", mean)
print("Variance:\n", var)


class LayerNorm(nn.Module):
    # layer归一化的函数,可以避免信息泄露也可以稳定
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 避免0的产生导致崩溃
        self.scale = nn.Parameter(torch.ones(emb_dim))  # 动态的缩放参数
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 动态的偏移参数

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # 算平均值
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 算方差
        norm_x = (x - mean) / torch.sqrt(var + self.eps)  # 归一化
        return self.scale * norm_x + self.shift  # 通过Ω和  œ 调整归一化后的值范围和位置


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            # 这一步把它变得平滑了很多
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


# import matplotlib
#
# matplotlib.use('MacOSX')  # 或者 'Qt5Agg'，取决于你系统安装了哪个
# import matplotlib.pylab as plt
#
# gelu, relu = GELU(), nn.ReLU()  # 先把函数给个小名
#
# # Some sample data
# x = torch.linspace(-3, 3, 100)  # 初定义一个张量
# y_gelu, y_relu = gelu(x), relu(x)  # 两种激活函数
#
# plt.figure(figsize=(8, 3))
# for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
#     plt.subplot(1, 2, i)
#     plt.plot(x, y)
#     plt.title(f"{label} activation function")
#     plt.xlabel("x")
#     plt.ylabel(f"{label}(x)")
#     plt.grid(True)
#
# plt.tight_layout()
# plt.show()


# 一个经典的作图


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    # 运行一次就线性两次激活一次
    def forward(self, x):
        return self.layers(x)


print(GPT_CONFIG_124M["emb_dim"])

import torch
import torch.nn as nn


class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        # 定义多层网络，包含 5 层线性层和激活函数 GELU
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])
        # 定义一个五层的神经网络块，其中每层包含一个线性变换和一个激活函数 GELU，
        # 类似 ResNet 的结构，支持添加残差连接。

    def forward(self, x):
        # 遍历每一层
        for layer in self.layers:
            # 当前层的输出
            layer_output = layer(x)
            # 检查是否可以应用残差连接
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output  # 如果输入和输出维度匹配，添加残差连接
            else:
                x = layer_output  # 否则直接输出当前层结果
        return x  # 返回最终结果


def print_gradients(model, x):
    # 前向传播
    output = model(x)
    target = torch.tensor([[0.]])  # 定义目标值
    # 计算损失，使用均方误差损失函数
    loss = nn.MSELoss()
    loss = loss(output, target)

    # 反向传播，计算梯度
    loss.backward()

    # 打印每层权重的梯度均值
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")


layer_sizes = [3, 3, 3, 3, 3, 1]

sample_input = torch.tensor([[1., 0., -1.]])

torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=False
)
print_gradients(model_without_shortcut, sample_input)
# 一次一次输出梯度

torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=True
)
print_gradients(model_with_shortcut, sample_input)


# 引入了残差链接,发现梯度消失的缺点明显改善了


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"
        # 确保是可以被整除的

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim
        # 初始化头的维度、数量
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        # 头的输出结合线性层
        self.dropout = nn.Dropout(dropout)
        # 进行dropout防止过拟合
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )
        # 上三角掩码，确保因果性

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)
        # 把输出的维度拆成头*头大小
        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        # 转制维度,听说是为了更好的计算注意力
        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        # 计算缩放点积注意力
        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head
        # 将掩码缩减到当前 token 数量，并转换为布尔型
        # 进而实现动态遮蔽,所以不用另开好几个数组
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # 遮蔽矩阵
        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        # 归一化
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)
        # 头的合并
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        # 对上下文向量的形状进行调整，确保输出的形状
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],  # 输入特征维度
            d_out=cfg["emb_dim"],  # 输出特征维度
            context_length=cfg["context_length"],  # 上下文长度
            num_heads=cfg["n_heads"],  # 注意力头的数量
            dropout=cfg["drop_rate"],  # Dropout 比例
            qkv_bias=cfg["qkv_bias"]  # 查询、键和值的偏置
        )  # 多头注意力模块，结合各种参数
        self.ff = FeedForward(cfg)  # 前馈神经网络模块
        self.norm1 = LayerNorm(cfg["emb_dim"])  # 第一归一化层
        self.norm2 = LayerNorm(cfg["emb_dim"])  # 第二归一化层
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])  # 残差连接的 Dropout

    def forward(self, x):
        # 对注意力模块的快捷连接
        shortcut = x
        x = self.norm1(x)  # 应用第一归一化层
        x = self.att(x)  # 通过多头注意力模块，形状为 [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)  # 应用 Dropout
        x = x + shortcut  # 将原始输入加回，实现残差连接

        # 对前馈网络模块的残差连接
        shortcut = x
        x = self.norm2(x)  # 应用第二归一化层
        x = self.ff(x)  # 通过前馈神经网络模块
        x = self.drop_shortcut(x)  # 应用 Dropout
        x = x + shortcut  # 将原始输入加回，实现残差连接

        return x


torch.manual_seed(123)

x = torch.rand(2, 4, 768)  # Shape: [batch_size, num_tokens, emb_dim]
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)


# 经典的一系列操作


class GPTModel(nn.Module):  # 召唤GPT!
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        # 新建字典、位置信息、还有dropout的比率设置
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        # 解包操作

        self.trf_blocks = nn.Sequential(
            TransformerBlock(cfg),
            TransformerBlock(cfg),
            TransformerBlock(cfg)
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        # 归一化
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
        # 输出头保证维度

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)
# 经典操作

total_params = sum(p.numel() for p in model.parameters())
# 模型的总参数数量
print(f"Total number of parameters: {total_params:,}")

print("Token embedding layer shape:", model.tok_emb.weight.shape)
print("Output layer shape:", model.out_head.weight.shape)
# 输出格式让我们更好地理解

total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")
# Parameter- sharing

# Calculate the total size in bytes (assuming float32, 4 bytes per parameter)
total_size_bytes = total_params * 4

# Convert to megabytes
total_size_mb = total_size_bytes / (1024 * 1024)

print(f"Total size of the model: {total_size_mb:.2f} MB")


# 计算总的容量

def generate_text_simple(model, idx, max_new_tokens, context_size):
    # 预测单词的模块
    # idx 是当前上下文中的（batch, n_tokens）索引数组
    for _ in range(max_new_tokens):
        # 每次生成一个单词后，重新将其加入序列中
        # 如果当前上下文长度超过模型支持的最大上下文长度，则截取
        # 例如，如果LLM只支持5个token，而上下文长度为10
        # 那么只使用最后5个token作为上下文
        idx_cond = idx[:, -context_size:]
        # 如果idx的长度超过模型支持的上下文长度size，只保留最后size个token
        # 避免溢出
        # 获取预测结果
        with torch.no_grad():  # 在推理阶段，不需要计算梯度，因为没有反向传播
            # 这样可以减少存储开销
            logits = model(idx_cond)
            # 模型输出结果
        # 只关注最后一个时间步的输出
        # (batch, n_tokens, vocab_size) 变为 (batch, vocab_size)
        logits = logits[:, -1, :]
        # 关注最后一个时间步
        # 使用softmax函数计算概率
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)
        # 归一化
        # 获取具有最高概率值的词汇索引
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)
        # 获取概率最高的词汇索引
        # 将采样的索引添加到序列中
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


start_context = "Hello, I am"
# 模拟
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
# 进行语义理解
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", encoded_tensor.shape)
# 最终输出格式

model.eval() # disable dropout
#在检验的时候不需要正则化了
out = generate_text_simple(
    model=model,
    #左边的参数名字,右边是函数传入的实际模型
    idx=encoded_tensor, #上下文的索引
    max_new_tokens=6, #最多运行六次,然后取结果概率最高的
    #初始文本➕6
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output:", out)
print("Output length:", len(out[0]))
#输出长度还有每个单词的id
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)