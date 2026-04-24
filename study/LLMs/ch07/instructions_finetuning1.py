import json
import urllib
import urllib.request  # 修改这里
import os


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


# 自定义训练集、测试集和验证集的大小
train_portion = int(len(data) * 0.85)  # 85% 作为训练集
test_portion = int(len(data) * 0.1)  # 10% 作为测试集
val_portion = len(data) - train_portion - test_portion  # 剩下的 5% 作为验证集

# 划分数据集
train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

import torch

torch.set_num_threads(1)  # 限制单线程，排除多线程竞争导致的数值错误
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
from importlib.metadata import version

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# allowed model names
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

# 1. 指定模型名称
# model_name = "openai-community/gpt2"
model_name = "openai-community/gpt2-medium"
hf_token = os.getenv("HF_TOKEN")
# 2. 加载预训练权重和模型结构
# GPT2LMHeadModel 包含了预测下一个词所需的线性输出层
model = GPT2LMHeadModel.from_pretrained(model_name)
model = model.float()
# 3. 加载对应的分词器
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
print(f"Model vocab size: {model.config.vocab_size}")
print(f"Tokenizer vocab size: {len(tokenizer)}")

# gpt2作为编码模型
# tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))


def custom_collate_fn(
        batch,
        pad_token_id=50256,
        ignore_index=-100,
        allowed_max_length=None,
        device="cpu"
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


# if torch.cuda.is_available():
#     device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")
device = torch.device("cpu")
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


#
# print("Train loader:")
# for inputs, targets in train_loader:
#     print(inputs.shape, targets.shape)
#
# print(inputs[0])
#
# print(targets[0])


def verify_model_health(model):
    print("开始深度验证模型权重...")
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"❌ {name} 包含 NaN")
        if torch.isinf(param).any():
            print(f"❌ {name} 包含 Inf")
        if param.abs().max() > 1e6:  # 正常权重不应该超过这个数
            print(f"⚠️ {name} 包含异常极大值: {param.abs().max().item()}")


# 在你的 model = GPT2LMHeadModel.from_pretrained(model_name) 之后调用
verify_model_health(model)
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

import torch


def calc_loss_batch(input_batch, target_batch, model, device):
    # 确保输入数据在正确的设备上
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    # 1. 检查全屏蔽风险 (CrossEntropy 零分母检查)
    if (target_batch == -100).all():
        print("⚠️ 警告：检测到当前 Batch 的 Targets 全部为 -100，跳过计算。")
        return torch.tensor(0.0, device=device, requires_grad=True)

    # 2. 生成 Attention Mask
    # 注意：填充 ID 需与你的 Tokenizer 一致，GPT-2 通常是 50256
    attention_mask = (input_batch != 50256).long().to(device)

    # 3. 提升模型精度至 float64 (数值稳定性最强模式)
    # 注意：这会显著消耗 CPU 内存和计算时间
    model.to(torch.float64)

    # 4. 前向传播
    # 显式转换输入类型以匹配模型精度
    outputs = model(
        input_ids=input_batch.to(torch.int64),
        attention_mask=attention_mask.to(torch.float64),
        labels=target_batch.to(torch.int64)
    )

    logits = outputs.logits
    loss = outputs.loss

    # 5. 诊断 NaN
    if torch.isnan(loss):
        print("\n--- [双精度模式] 依然检测到 Loss 为 NaN ---")
        print(f"Logits Max: {logits.max().item()}")
        print(f"Logits 包含 NaN 的比例: {torch.isnan(logits).float().mean().item():.2%}")

        # 检查是否权重本身已经带了 NaN (即使在 float64 下)
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"🚨 发现损坏权重: {name}")
                break

        # 尝试通过 CPU 手动 CrossEntropy 抢救（强制 float32 降噪）
        logits_f32 = logits.detach().to("cpu").to(torch.float32)
        targets_f32 = target_batch.to("cpu")
        loss = torch.nn.functional.cross_entropy(
            logits_f32.view(-1, logits_f32.size(-1)),
            targets_f32.view(-1),
            ignore_index=-100
        ).to(device)

    # 6. 重要：将模型还原回 float32 供优化器使用
    # 如果不还原，后续 optimizer.step() 可能会报错（如果优化器初始化时是 fp32）
    model.to(torch.float32)

    return loss


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
            logits = model(idx_cond)

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

torch.manual_seed(123)

with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

# 先看一次没有微调的结果
print("Training loss:", train_loss)
print("Validation loss:", val_loss)
