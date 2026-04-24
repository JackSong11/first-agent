import urllib.request
import zipfile
import os
from pathlib import Path

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

# 完成导入数据
# def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
#     if data_file_path.exists():
#         print(f"{data_file_path} already exists. Skipping download and extraction.")
#         return
#
#     # 下载文件
#     with urllib.request.urlopen(url) as response:
#         with open(zip_path, "wb") as out_file:
#             out_file.write(response.read())
#
#     # 解压文件
#     with zipfile.ZipFile(zip_path, "r") as zip_ref:
#         zip_ref.extractall(extracted_path)
#     # 添加.tsv文件扩展名
#     original_file_path = Path(extracted_path) / "SMSSpamCollection"
#     os.rename(original_file_path, data_file_path)
#     print(f"File downloaded and saved as {data_file_path}")
#
#
# download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

import pandas as pd

df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
print(df["Label"].value_counts())


# 单元模型启动

def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    # 算一下类别为spam的样本出现的次数
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    # 随机采样 “ham” 使其数量与“spam”样本数量一致
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    # 合并 “spam” 和采样后的 “ham”数据
    return balanced_df


balanced_df = create_balanced_dataset(df)
print(balanced_df["Label"].value_counts())

balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})


# print(balanced_df)

def random_split(df, train_frac, validation_frac):
    # 把数据集Dataframe打乱
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # 计算分割系数
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # 分割数据集Dataframe
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df


train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
# 剩余部分为测试集，占总数据集的比例为 0.2

train_df.to_csv("train.csv", index=None)
validation_df.to_csv("validation.csv", index=None)
test_df.to_csv("test.csv", index=None)
# DataFrame 被随机划分为训练集、验证集和测试集并保存


import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
# 打印<|endoftext|>对应的词元id
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

import torch
from torch.utils.data import Dataset


class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        # 先读入文本
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]
        # 编码成对应的词元id
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # 如果序列长度超过 max_length，则进行截断
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # 填充到最长序列的长度
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        # 获取指定索引的数据样本
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)
        # 输出序列长度

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


train_dataset = SpamDataset(
    csv_file="train.csv",
    max_length=None,
    tokenizer=tokenizer
)

print(train_dataset.max_length)

val_dataset = SpamDataset(
    csv_file="validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)
test_dataset = SpamDataset(
    csv_file="test.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)
# 将验证集和测试集数据填充到最长序列的长度

from torch.utils.data import DataLoader

num_workers = 0
batch_size = 8

# 设置种子确保可复现
torch.manual_seed(123)
# 初始化数据加载器
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)
# 对齐超参数,但是dataset需要区别一下训练集、验证集和测试集

print("Train loader:")
for input_batch, target_batch in train_loader:
    pass
    # 如果这个数据在训练集出现过,则跳过

print("Input batch dimensions:", input_batch.shape)
print("Label batch dimensions", target_batch.shape)

print(f"{len(train_loader)} training batches")
print(f"{len(val_loader)} validation batches")
print(f"{len(test_loader)} test batches")

from importlib.metadata import version
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# allowed model names
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

# 1. 指定模型名称
model_name = "openai-community/gpt2"
hf_token = os.getenv("HF_TOKEN")
# 2. 加载预训练权重和模型结构
# GPT2LMHeadModel 包含了预测下一个词所需的线性输出层
model = GPT2LMHeadModel.from_pretrained(model_name)

# 3. 加载对应的分词器
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# 设置为评估模式
model.eval()


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


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            outputs = model(idx_cond)
            # outputs = model(input_ids)
            logits = outputs.logits  # 先提取出张量
            # logits = logits[:, -1, :]  # 再进行切片操作

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


# text_1 = "Every effort moves you"
#
# token_ids = generate_text_simple(
#     model=model,
#     idx=text_to_token_ids(text_1, tokenizer),
#     max_new_tokens=200,
#     context_size=1024
# )
# # 生成文本
# print(token_ids_to_text(token_ids, tokenizer))
#
# text_2 = (
#     "Is the following text 'spam'? Answer with 'yes' or 'no':"
#     " 'You are a winner you have been specially"
#     " selected to receive $1000 cash or a $2000 award.'"
# )
#
# token_ids = generate_text_simple(
#     model=model,
#     idx=text_to_token_ids(text_2, tokenizer),
#     max_new_tokens=200,
#     context_size=1024
# )
#
# print(token_ids_to_text(token_ids, tokenizer))


print(model)

# --- 核心修改部分开始 ---

# Step 1: 冻结所有参数
for param in model.parameters():
    param.requires_grad = False

# Step 2: 替换输出层 (lm_head)
# 注意：GPT2 的 embedding 维度是 768
torch.manual_seed(123)
num_classes = 2
model.lm_head = torch.nn.Linear(in_features=768, out_features=num_classes)

# Step 3: 解冻最后一部分层 (按照课本逻辑)
# 对应课本的 model.trf_blocks[-1]
for param in model.transformer.h[-1].parameters():
    param.requires_grad = True

# 对应课本的 model.final_norm
for param in model.transformer.ln_f.parameters():
    param.requires_grad = True

# --- 核心修改部分结束 ---

# 验证模型结构
print("Modified Model lm_head:", model.lm_head)

# 4. 测试模型输出
inputs_text = "Do you have time"
inputs = tokenizer.encode(inputs_text, return_tensors="pt")
print(inputs)

model.eval()  # 预测模式
with torch.no_grad():
    outputs1 = model(inputs)  # 因为这是一个CausalLMOutputWithCrossAttentions类
    outputs = outputs1.logits  # 形状: [batch_size, sequence_length, num_classes]
    print(outputs)
    print(outputs.shape)

# 分类任务通常取最后一个词元的 Logits
# 因为 GPT-2 是自回归模型，最后一个词元包含了前面所有词的信息
last_token_logits = outputs[:, -1, :]

print("\nInputs dimensions:", inputs.shape)
print("Outputs (last token) shape:", last_token_logits.shape)  # 应该是 [1, 2]
print("Logits:", last_token_logits)

probas = torch.softmax(outputs[:, -1, :], dim=-1)
label = torch.argmax(probas)
print("Class label:", label.item())

logits = outputs[:, -1, :]
label = torch.argmax(logits)
print("Class label:", label.item())


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0
    # 先改成评估模式,在初始化两个储存器
    if num_batches is None:
        num_batches = len(data_loader)
        # num_batches设为None则取全部数据
    else:
        num_batches = min(num_batches, len(data_loader))
        # 取num_batches和len(data_loader)中较小的那个
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            # 仅处理前 num_batches 批次的数据
            with torch.no_grad():
                outputs = model(input_batch)
                logits = outputs.logits[:, -1, :]  # 获取最后一个词元的预测结果
                # logits = model(input_batch)[:, -1, :]  # 最后一个输出词元的概率
            # 预测内容取概率最大值
            predicted_labels = torch.argmax(logits, dim=-1)
            # 当前批次的样本数
            num_examples += predicted_labels.shape[0]
            # 预测正确的样本数量
            correct_predictions += (predicted_labels == target_batch).sum().item()

        else:
            break
    return correct_predictions / num_examples


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Note:

# 取消注释以下行将允许代码在 Apple Silicon 芯片上运行（如果适用），
# 这比在 Apple CPU 上运行快大约 2 倍（根据 M3 MacBook Air 的测量结果）。
# 截至目前，在 PyTorch 2.4 版本中，通过 CPU 和 MPS 得到的结果是相同的。
# 然而，在 PyTorch 的早期版本中，使用 MPS 时，可能会观察到不同的结果。
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Running on {device} device.")

model.to(device)  # 对于 nn.Module 类，无需 model = model.to(device)赋值操作

torch.manual_seed(123)  # 由于训练数据加载器中的随机打乱，因此设置随机种子以确保可重复性

train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)
test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)
# 各种计算
print(f"Training accuracy: {train_accuracy * 100:.2f}%")
print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # 修改这里：先获取输出对象，再提取 logits
    outputs = model(input_batch)
    logits = outputs.logits[:, -1, :]
    # logits = model(input_batch)[:, -1, :]  # Logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss
    # 用交叉熵计算损失函数


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # 如果 num_batches 超过数据加载器中的批次数，则减少批次数以匹配数据加载器中的总批次数
        # num_batches = min(num_batches, len(data_loader))
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                # 总损失值
                total_loss += loss.item()

            else:
                break
    return total_loss / num_batches


with torch.no_grad():  # 因为我们不进行训练，所以为了提高效率禁用梯度跟踪
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
    test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)

print(f"Training loss: {train_loss:.3f}")
print(f"Validation loss: {val_loss:.3f}")
print(f"Test loss: {test_loss:.3f}")


# 跟第五章的一摸一样那就当重新复习了
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


def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1
    # 初始化分类头
    for epoch in range(num_epochs):
        model.train()
        # 每次都进入训练模块
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            # 清零梯度
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            # 对损失值执行反向传播记录梯度
            optimizer.step()
            # 用梯度优化权重
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy * 100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


import time

start_time = time.time()

torch.manual_seed(123)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

num_epochs = 5
train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=50, eval_iter=5,
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")
# 输出每一次的损失值跟在一定频次下进行准确率输出

# import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('MacOSX')  # 或者 'Qt5Agg'，取决于你系统安装了哪个
import matplotlib.pylab as plt


def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)

    fig.tight_layout()
    plt.savefig(f"{label}-plot.pdf")
    plt.show()


# 一个经典的画图操作

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))

plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy")

train_accuracy = calc_accuracy_loader(train_loader, model, device)
val_accuracy = calc_accuracy_loader(val_loader, model, device)
test_accuracy = calc_accuracy_loader(test_loader, model, device)

print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")
# 输出总的训练集、验证集和测试集的准确率
