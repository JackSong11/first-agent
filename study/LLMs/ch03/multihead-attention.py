# NBVAL_IGNORE_OUTPUT
from importlib.metadata import version
from torch.utils.data import Dataset, DataLoader
import torch
import tiktoken

print("torch version:", version("torch"))


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []  # 输入ID列表
        self.target_ids = []  # 目标ID列表

        # 对整个文本进行分词
        token_ids = tokenizer.encode(txt, allowed_special={'<|endoftext|>'})

        # 使用滑动窗口将文本分割成重叠的最大长度序列
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]  # 输入片段
            target_chunk = token_ids[i + 1: i + max_length + 1]  # 目标片段（右移一个位置）
            self.input_ids.append(torch.tensor(input_chunk))  # 将输入片段转换为张量
            self.target_ids.append(torch.tensor(target_chunk))  # 将目标片段转换为张量

    def __len__(self):
        return len(self.input_ids)  # 返回数据集的大小

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]  # 获取特定索引的输入和目标


def create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True):
    # 初始化分词器
    tokenizer = tiktoken.get_encoding("gpt2")

    # 创建数据集
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader  # 返回数据加载器


with open("small-text-sample.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()  # 读取文本文件

tokenizer = tiktoken.get_encoding("gpt2")  # 初始化分词器
encoded_text = tokenizer.encode(raw_text)  # 对文本进行编码

vocab_size = 50257  # 词汇表大小
output_dim = 256  # 输出维度
max_len = 1024  # 最大序列长度
context_length = max_len  # 上下文长度

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)  # 创建词嵌入层
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)  # 创建位置嵌入层

max_length = 4  # 每个输入片段的最大长度
dataloader = create_dataloader(raw_text, batch_size=8, max_length=max_length, stride=max_length)  # 创建数据加载器


