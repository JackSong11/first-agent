from importlib.metadata import version

pkgs = ["matplotlib",
        "numpy",
        "tiktoken",
        "torch",
        # "tensorflow"  # For OpenAI's pretrained weights
        ]
for p in pkgs:
    print(f"{p} version: {version(p)}")
# 同样导入库并检查版本


import torch
from study.LLMs.ch04.gpt_clean import GPTModel

GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 256,  # Shortened context length (orig: 1024)
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False  # Query-key-value bias
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval();  # Disable dropout during inference
# 导入模型, 设定一系列参数, 设定随机种子确保可复现


import tiktoken
from study.LLMs.ch04.gpt_clean import generate_text_simple


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


# 给输入的字符进行编码并实现一个Batch维度的向量,符合模型的输入形式
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


# 反向编码,去掉移除张量中的批次维度, 变成普通的链表
start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")
# 举个例子
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    # 初始上下文的Token ID张量，是上一步 text_to_token_ids 的输出
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)
# 输出最长单词度为10的句子
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

inputs = torch.tensor([[16833, 3626, 6100],  # ["every effort moves",
                       [40, 1107, 588]])  # "I really like"]
# 用向量的形式展现输入的文本
targets = torch.tensor([[3626, 6100, 345],  # [" effort moves you",
                        [1107, 588, 11311]])  # " really like chocolate"]
# 用向量的形式展现要输出的东西

with torch.no_grad():
    logits = model(inputs)
# 不用梯度计算的计算inputes并储存
probas = torch.softmax(logits, dim=-1)  # Probability of each token in vocabulary
# 用soft Max整理logits
print(probas.shape)  # Shape: (batch_size, num_tokens, vocab_size)

token_ids = torch.argmax(probas, dim=-1, keepdim=True)
# 相当于用贪心算法给出最有可能的答案
print("Token IDs:\n", token_ids)

print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
# 给出答案
print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")
# 给出事实上的结论

text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 1:", target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 2:", target_probas_2)

# Compute logarithm of all token probabilities
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)
print(log_probas.shape)
# 用对数输出他最大的可能数值

# Calculate the average probability for each token
avg_log_probas = torch.mean(log_probas)
print(avg_log_probas)
print(avg_log_probas.shape)
# 对数概率平均值

neg_avg_log_probas = avg_log_probas * -1
# 最大化对数等价为最小化负对数
print(neg_avg_log_probas)

# Logits have shape (batch_size, num_tokens, vocab_size)
print("Logits shape:", logits.shape)

# Targets have shape (batch_size, num_tokens)
print("Targets shape:", targets.shape)

logits_flat = logits.flatten(0, 1)
# 将张量 logits 的 第0维和第1维合并为一个维度，展平成一个二维张量
targets_flat = targets.flatten()
# 将张量 targets 展平为一维张量

print("Flattened logits:", logits_flat.shape)
print("Flattened targets:", targets_flat.shape)

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(loss)
# 封装函数出马,一个代替好几行

perplexity = torch.exp(loss)
# 指数化loss作为P值
print(perplexity)

import os
import urllib.request

file_path = "the-verdict.txt"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
# 引入数据集
if not os.path.exists(file_path):
    with urllib.request.urlopen(url) as response:
        text_data = response.read().decode('utf-8')
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text_data)
else:
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
# 一系列经典的读取数据操作

# First 100 characters
print(text_data[:99])

# Last 100 characters
print(text_data[-99:])

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
# 统计一下文本的长度,编码文本内容并输出文本个数
print("Characters:", total_characters)
print("Tokens:", total_tokens)

from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    # 让GPT初始化一个类型
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})  # id是文本内容编码过来的

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


# 从一个库导入之前的文章
# Train/validation ratio
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]
# 这边可以手动定义训练集跟测试剂的比例

torch.manual_seed(123)
# 依旧保持可复现
train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)
# 初始化输入训练模型,给出批处理的大小、给出最大文本容量防止溢出
# 给出不畅,丢弃最后一批不足的文本,打开随机防止拟合过度
val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)
# 验证数据集仅仅修改了是否丢弃跟随抽取

# Sanity check
# 神圣性,看一下一批次够了没

if total_tokens * (train_ratio) < GPT_CONFIG_124M["context_length"]:
    print("Not enough tokens for the training loader. "
          "Try to lower the `GPT_CONFIG_124M['context_length']` or "
          "increase the `training_ratio`")

if total_tokens * (1 - train_ratio) < GPT_CONFIG_124M["context_length"]:
    print("Not enough tokens for the validation loader. "
          "Try to lower the `GPT_CONFIG_124M['context_length']` or "
          "decrease the `training_ratio`")

print("Train loader:")
for x, y in train_loader:
    print(x.shape, y.shape)

print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)

train_tokens = 0
for input_batch, target_batch in train_loader:
    train_tokens += input_batch.numel()

val_tokens = 0
for input_batch, target_batch in val_loader:
    val_tokens += input_batch.numel()
# 每次加一下训练数据集所有元素的种类
print("Training tokens:", train_tokens)
print("Validation tokens:", val_tokens)
print("All tokens:", train_tokens + val_tokens)


# 在 PyTorch 中，调用 .numel() 方法会返回张量中所有元素的总数，无论张量的形状或维度如何

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # 呼唤GPU
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    # 用交叉熵函数对于logits进行计算并且拉伸到二维长度
    return loss


# 一个计算批损失的函数

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # 如果指定的批次数超过数据加载器中的总批次数，则将批次数减少到与数据加载器的总批次数匹配。
        num_batches = min(num_batches, len(data_loader))
        # 减少需要处理的数量,同时也是防止溢出
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
        # 一点点加上去的损失
    return total_loss / num_batches


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 如果支持，则调用 GPU

# 注意：
# 如果取消注释以下代码块，代码可以在 Apple Silicon 芯片上运行（如果适用），
# 在 M3 MacBook Air 上测量速度大约是 Apple CPU 的两倍。
# 然而，计算得到的损失值可能会略有不同。

# if torch.cuda.is_available():
#    device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#    device = torch.device("mps")
# else:
#    device = torch.device("cpu")
#
# print(f"Using {device} device.")

model.to(device)  # 对于 nn.Module 类，不需要赋值 model = model.to(device)

torch.manual_seed(123)  # 固定随机种子，保证数据加载器打乱数据的结果可复现

with torch.no_grad():  # 禁用梯度跟踪以提高效率，因为此时尚未开始训练
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)

# 推理阶段不计算梯度
print("Training loss:", train_loss)
print("Validation loss:", val_loss)


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    # 评价模块
    model.eval()
    # 检验模式
    with torch.no_grad():
        # 我认为的双保险,防止梯度更新
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    #	在评估结束后切换回训练模式，确保模型能继续用于训练。
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    # 初始化训练模型而且给了空的队列
    # Main training loop
    for epoch in range(num_epochs):  # 训练次数
        model.train()  # Set model to training mode
        # 转移到训练模块
        for input_batch, target_batch in train_loader:
            # 从loader里面调出输入跟目标
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            # 清空所有函数的梯度
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            # 计算损失函数
            loss.backward()  # Calculate loss gradients
            # 反向传播优化
            optimizer.step()  # Update model weights using loss gradients
            # 更新权重
            tokens_seen += input_batch.numel()
            # 加一下一共有多少
            global_step += 1
            # 看一下一共训练了多少步
            # Optional evaluation step
            if global_step % eval_freq == 0:
                # 按照一定的步数进行记录
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                # 计算损失函数
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                # 加到list中
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


# Note:
# Uncomment the following code to calculate the execution time
# 下面可以看一下计算了多久
# import time
# start_time = time.time()

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
# 经典操作
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
# 用Adam进行优化,其中学习rate为0.004,动量衰减是0.1
num_epochs = 10
# 10论学习
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)
# 记录了开始文本、检验的频率
# 注意：
# 如果需要显示执行时间，请取消注释以下代码
# end_time = time.time()
# execution_time_minutes = (end_time - start_time) / 60
# print(f"训练完成耗时 {execution_time_minutes:.2f} 分钟。")


# import matplotlib
#
# matplotlib.use('MacOSX')  # 或者 'Qt5Agg'，取决于你系统安装了哪个
# import matplotlib.pylab as plt
# from matplotlib.ticker import MaxNLocator
#
#
# def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
#     fig, ax1 = plt.subplots(figsize=(5, 3))
#
#     # Plot training and validation loss against epochs
#     ax1.plot(epochs_seen, train_losses, label="Training loss")
#     ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
#     ax1.set_xlabel("Epochs")
#     ax1.set_ylabel("Loss")
#     ax1.legend(loc="upper right")
#     ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis
#
#     # Create a second x-axis for tokens seen
#     ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
#     ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
#     ax2.set_xlabel("Tokens seen")
#
#     fig.tight_layout()  # Adjust layout to make room
#     plt.savefig("loss-plot.pdf")
#     plt.show()
#
# print()
# epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
# plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
# #一个经典的plot画图函数


model.to("cpu")
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"]
)
#经典的载入
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

vocab = {
    "closer": 0,
    "every": 1,
    "effort": 2,
    "forward": 3,
    "inches": 4,
    "moves": 5,
    "pizza": 6,
    "toward": 7,
    "you": 8,
}

inverse_vocab = {v: k for k, v in vocab.items()}
#插入
# Suppose input is "every effort moves you", and the LLM
# returns the following logits for the next token:
next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)

probas = torch.softmax(next_token_logits, dim=0)
#softmax归一化
next_token_id = torch.argmax(probas).item()
#选个可能性最大
# The next generated token is then as follows:
print(inverse_vocab[next_token_id])


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    # 生成模块
    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        # 计算预测值,但是切最后一个
        # New: Filter logits with top_k sampling
        # top K采样
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # New: Apply temperature scaling
        # 温度校正
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
            # 从概率分布中采样下一个 token

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)
            # 如果未启用采样，选择概率最高的 token 作为下一个 token
        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx


torch.manual_seed(123)

token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
# 经典的操作

torch.save(model.state_dict(), "../ch04/model.pth")
#训练完的数据保存一下

# model = GPTModel(GPT_CONFIG_124M)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))
# model.eval();


# torch.save({
#     "model_state_dict": model.state_dict(),
#     "optimizer_state_dict": optimizer.state_dict(),
#     },
#     "model_and_optimizer.pth"
# )
#全家整整齐齐地保存

# checkpoint = torch.load("model_and_optimizer.pth", weights_only=True)
# #保存检查点
# model = GPTModel(GPT_CONFIG_124M)
# model.load_state_dict(checkpoint["model_state_dict"])
#
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)
# optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
# model.train();
# #调整到训练模式