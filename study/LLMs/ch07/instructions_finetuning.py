from importlib.metadata import version

pkgs = [
    "matplotlib",  # 绘图库
    "tiktoken",  # 分词器
    "torch",  # 深度学习库
    "tqdm",  # 进度条
    # "tensorflow",  # 用于加载OpenAI的预训练权重
]
for p in pkgs:
    print(f"{p} version: {version(p)}")

import json
import os
import urllib
import urllib.request  # 修改这里


def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


# 在网上下载并打开数据库

file_path = "instruction-data.json"
url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)

data = download_and_load_file(file_path, url)
print("Number of entries:", len(data))


# 看一下数据一共有多少条

def format_input(entry):
    # 使用数据库的提示词
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    # 如果没有输入的格式，将如何处理
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


model_input = format_input(data[50])
desired_response = f"\n\n### Response:\n{data[50]['output']}"
# 先使用五十条数据进行测试
print(model_input + desired_response)

model_input = format_input(data[999])
desired_response = f"\n\n### Response:\n{data[999]['output']}"

print(model_input + desired_response)

# 自定义训练集、测试集和验证集的大小
train_portion = int(len(data) * 0.85)  # 85% 作为训练集
test_portion = int(len(data) * 0.1)  # 10% 作为测试集
val_portion = len(data) - train_portion - test_portion  # 剩下的 5% 作为验证集

# 划分数据集
train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

print("Training set length:", len(train_data))
print("Validation set length:", len(val_data))
print("Test set length:", len(test_data))

import torch
from torch.utils.data import Dataset


class InstructionDataset(Dataset):
    # 指示数据类的构建
    def __init__(self, data, tokenizer):
        self.data = data
        # 实例化数据
        # 对文本进行预编码
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    # 链表访问
    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


import tiktoken

device = "cpu"
# gpt2作为编码模型
tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))


def custom_collate_draft_1(
        batch,
        pad_token_id=50256,
        device = "cpu"
):
    # 找到批次中最长的序列
    # 并将最大长度增加1，这样会在后面添加一个额外的填充 token
    batch_max_length = max(len(item) + 1 for item in batch)

    inputs_lst = []
    # 准备输入
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        # 复制后进行填充
        padded = (
                new_item + [pad_token_id] *
                (batch_max_length - len(new_item))
        )
        # 去掉最后一个表示并保存
        inputs = torch.tensor(padded[:-1])

        inputs_lst.append(inputs)
    # 堆积起来并输送给gpu
    inputs_tensor = torch.stack(inputs_lst).to(device)

    return inputs_tensor


inputs_1 = [0, 1, 2, 3, 4]
inputs_2 = [5, 6]
inputs_3 = [7, 8, 9]

batch = (
    inputs_1,
    inputs_2,
    inputs_3
)

print(custom_collate_draft_1(batch))


def custom_collate_draft_2(
        batch,
        pad_token_id=50256,
        device = "cpu"
):
    # 找到最大的序列长度
    batch_max_length = max(len(item) + 1 for item in batch)
    # 准备一个空列表
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = (
                new_item + [pad_token_id] *
                (batch_max_length - len(new_item))
        )
        # 输入值是第一个到倒数第二个
        inputs = torch.tensor(padded[:-1])
        # 目标值是第二个到最后一个，这样子保证了长度一样
        targets = torch.tensor(padded[1:])

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


inputs, targets = custom_collate_draft_2(batch)
print(inputs)
print(targets)


def custom_collate_fn(
        batch,
        pad_token_id=50256,
        ignore_index=-100,
        allowed_max_length=None,
        device = "cpu"
):
    # 找到批次中最长的序列
    batch_max_length = max(len(item) + 1 for item in batch)

    # 填充并准备输入和目标
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # 添加一个 <|endoftext|> token
        new_item += [pad_token_id]
        # 将序列填充到最大长度
        padded = (
                new_item + [pad_token_id] *
                (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])  # 截断最后一个 token 作为输入
        targets = torch.tensor(padded[1:])  # 向右移1个位置作为目标

        # 新增：将目标中除了第一个填充 token 外的所有填充 token 替换为 ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # 新增：根据需要，限制序列的最大长度
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # 将输入和目标的列表转换为张量，并转移到目标设备
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


inputs, targets = custom_collate_fn(batch)
print(inputs)
print(targets)

logits_1 = torch.tensor(
    [[-1.0, 1.0],  # 1st training example
     [-0.5, 1.5]]  # 2nd training example
)
# 两个训练的实例
targets_1 = torch.tensor([0, 1])

loss_1 = torch.nn.functional.cross_entropy(logits_1, targets_1)
# 计算交叉熵
print(loss_1)

logits_2 = torch.tensor(
    [[-1.0, 1.0],
     [-0.5, 1.5],
     [-0.5, 1.5]]  # 新增第3个训练实例
)
targets_2 = torch.tensor([0, 1, 1])

loss_2 = torch.nn.functional.cross_entropy(logits_2, targets_2)
print(loss_2)

targets_3 = torch.tensor([0, 1, -100])

loss_3 = torch.nn.functional.cross_entropy(logits_2, targets_3)
print(loss_3)
print("loss_1 == loss_3:", loss_1 == loss_3)
# 综上所述、交叉熵会忽略-100

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 注意：
# 如果适用，取消注释以下行将使代码能够在Apple Silicon芯片上运行，
# 这比在Apple CPU上运行要快得多（在M3 MacBook Air上测得）。
# 然而，计算得到的loss可能会略有不同。


print("Device:", device)

from functools import partial

# 初始化定义
customized_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=1024
)

from torch.utils.data import DataLoader

num_workers = 0
batch_size = 8

torch.manual_seed(123)
# 初始化训练
train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

# 初始化验证与测试
val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

print("Train loader:")
for inputs, targets in train_loader:
    print(inputs.shape, targets.shape)

print(inputs[0])

print(targets[0])

from importlib.metadata import version
import torch

# allowed model names
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

# 1. 指定模型名称
model_name = "openai-community/gpt2-medium"
hf_token = os.getenv("HF_TOKEN")
# 2. 加载预训练权重和模型结构
# GPT2LMHeadModel 包含了预测下一个词所需的线性输出层
model = GPT2LMHeadModel.from_pretrained(model_name)

# 3. 加载对应的分词器
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# 设置为评估模式
model.eval()

torch.manual_seed(123)

input_text = format_input(val_data[0])
print(input_text)


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    # 生成模块
    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            # logits = model(idx_cond)
            # 【修复点】Hugging Face 模型返回的是对象，需要取 .logits
            outputs = model(idx_cond)
            logits = outputs.logits
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


def text_to_token_ids(text, tokenizer):
    # 【修复点】GPT2Tokenizer 不需要 allowed_special 参数
    encoded = tokenizer.encode(text)
    # encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


# 给输入的字符进行编码并实现一个Batch维度的向量,符合模型的输入形式
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


token_ids = generate(
    model=model,
    idx=text_to_token_ids(input_text, tokenizer),
    max_new_tokens=35,
    context_size=256,
    eos_id=50256,
)
generated_text = token_ids_to_text(token_ids, tokenizer)

response_text = (
    # 从生成的文本开始计数
    generated_text[len(input_text):]
    # 如果生成的文本包含 `### Response:`，则删除它
    .replace("### Response:", "")
    # 去掉空格
    .strip()
)
print(response_text)


def calc_loss_batch(input_batch, target_batch, model, device):
    # 1. 搬运数据并对齐类型
    input_batch = input_batch.to(device).to(torch.int64)
    target_batch = target_batch.to(device).to(torch.int64)

    # 2. 前向传播
    # 此时 model 已经是 float64，所以 outputs.logits 也会自动是 float64
    outputs = model(input_batch)
    logits = outputs.logits

    # 3. 计算 Loss (全程在 float64 下进行，精度极高，无 NaN 风险)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target_batch.view(-1),
        ignore_index=-100
    )

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


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            # logits = model(idx_cond)
            # 【修改点】Hugging Face 模型返回的是对象，需要取 .logits
            outputs = model(idx_cond)
            logits = outputs.logits

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    # context_size = model.pos_emb.weight.shape[0]
    context_size = model.config.n_ctx
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


model.to(device)
model.to(torch.float64)
torch.manual_seed(123)

with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

# 先看一次没有微调的结果
print("Training loss:", train_loss)
print("Validation loss:", val_loss)

import time

start_time = time.time()

torch.manual_seed(123)

# 用Adam训练,并定义了学习率、权重衰减等参数
# 第一步：把模型搬到设备并强制转为 float64
model.to(device)
model.to(torch.float64)

# 第二步：【关键】在模型转换完 float64 后再定义优化器
# 这样 optimizer 里的参数指针才是 float64 类型的
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0.1)
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

num_epochs = 2

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context=format_input(val_data[0]), tokenizer=tokenizer
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

torch.manual_seed(123)

for entry in test_data[:3]:
    input_text = format_input(entry)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=256,
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )

    print(input_text)
    print(f"\nCorrect response:\n>> {entry['output']}")
    print(f"\nModel response:\n>> {response_text.strip()}")
    print("-------------------------------------")

from tqdm import tqdm

for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
    input_text = format_input(entry)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=256,
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

    test_data[i]["model_response"] = response_text

with open("instruction-data-with-response.json", "w") as file:
    json.dump(test_data, file, indent=4)  # "indent"设置用于美化输出

print(test_data[0])

import re

file_name = f"{re.sub(r'[ ()]', '', "gpt2-medium")}-sft.pth"
torch.save(model.state_dict(), file_name)
print(f"Model saved as {file_name}")

# Load model via
# model.load_state_dict(torch.load("gpt2-medium355M-sft.pth"))