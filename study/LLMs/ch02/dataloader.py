from importlib.metadata import version
import torch

print(torch.cuda.is_available())
print(torch.__version__)

print("torch version:", version("torch"))
print("tiktoken version:", version("tiktoken"))
# 确认库已安装并显示当前安装的版本

import os  ##导入os库
import urllib.request  ##导入request库

if not os.path.exists("the-verdict.txt"):  ##如果文件不存在则创建，防止因文件已存在而报错
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)  ##从指定的地点读取文件

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()  ##读入文件按照utf-8

print("Total number of character:", len(raw_text))  ##先输出总长度
print(raw_text[:99])  ##输出前一百个内容

import re

text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text)  ##正则表达式按照空白字符进行分割

print(result)

result = re.split(r'([,.]|\s)', text)  ##只是按照, .分割
print(result)

##把上述结果去掉空格
result = [item for item in result if item.strip()]
print(result)

text = "Hello, world. Is this-- a test?"

result = re.split(r'([,.:;?_!"()\']|--|\s)', text)  ##就是按照常用的符号分割
result = [item.strip() for item in result if item.strip()]  ##去掉两端的空白字符 也是去掉了空字符串与仅包含空白字符的项
print(result)

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)  ##按照符号继续把原文件给分割了
preprocessed = [item.strip() for item in preprocessed if item.strip()]  ##去掉两端的空白字符 也是去掉了空字符串和仅包含空白字符的项
print(preprocessed[:30])

print(len(preprocessed))

all_words = sorted(set(preprocessed))  # 从去掉重复的字符
vocab_size = len(all_words)  # 计总的单词书

print(vocab_size)

vocab = {token: integer for integer, token in enumerate(all_words)}  ##先把word进行编号,再按照单词或者标点为索引(有HashList那味道了)

for i, item in enumerate(vocab.items()):
    # print(item)
    if i >= 50:
        break  ##遍历到前五十个


class SimpleTokenizerV1:  # 一个实例的名字创立
    def __init__(self, vocab):  ## 初始化一个字符串
        self.str_to_int = vocab  # 单词到整数的映射
        self.int_to_str = {i: s for s, i in vocab.items()}
        # 方便解码,进行整数到词汇的反向映射

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)  ##正则化分词标点符号

        preprocessed = [
            item.strip() for item in preprocessed if item.strip()  ## 去掉两端空格与全部的空句
        ]
        ids = [self.str_to_int[s] for s in preprocessed]  ##整理完的额字符串列表对应到id,从字典出来
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])  # 映射整数id到字符串。join是用前面那个(" ")联结成一个完整的字符串
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)  # 使用正则表达式，去除标点符号前的多余空格
        # \s+匹配一个或者多个空白  \1 替换到匹配
        return text


tokenizer = SimpleTokenizerV1(vocab)  # 用vocab创造一个实例

text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)  # 按照这个例子里的encode函数处理text
print(ids)

print(tokenizer.decode(ids))  # 按照这个例子里的decode函数处理text
print(tokenizer.decode(tokenizer.encode(text)))  # 按照这个例子里的decode函数处理(#按照这个例子里的encode函数处理text)

tokenizer = SimpleTokenizerV1(vocab)  ##用vocab创造一个实例

text = "Hello, do you like tea. Is this-- a test?"

# tokenizer.encode(text) # 会报错

all_tokens = sorted(list(set(preprocessed)))  # set去重 list把处理后的重新变为列表,然后排序
all_tokens.extend(["<|endoftext|>", "<|unk|>"])  # 加上未知的表示

vocab = {token: integer for integer, token in enumerate(all_tokens)}
# 遍历 enumerate(all_tokens) 中的每个元组 (integer, token)，以 token 作为键，integer 作为值创建字典条目。

print(len(vocab.items()))
for i, item in enumerate(list(vocab.items())[-5:]):  # 输出后五个内容与其标号
    print(item)


class SimpleTokenizerV2:  ##版本2.0,启动!
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}  # s为单词,i是key

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)  # 正则化按照标点分类
        preprocessed = [item.strip() for item in preprocessed if item.strip()]  # 去掉两头与所有空余句
        preprocessed = [
            item if item in self.str_to_int
            else "<|unk|>" for item in preprocessed
            # 遍历 preprocessed 中的每个 item，如果 item 存在于 self.str_to_int（即词汇表）中，就保留 item
            # 如果不存在（即该单词或符号未定义在词汇表中），就替换为特殊标记 <|unk|>。
            # 拓展:推导式（如列表推导式）是一种紧凑的语法，专门用于生成新列表（或其他容器）
            # 与普通 for 循环相比，它更加简洁和高效，但逻辑复杂时可能会降低可读性。
        ]

        ids = [self.str_to_int[s] for s in preprocessed]  # 单词或标点映射为整数列表
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


tokenizer = SimpleTokenizerV2(vocab)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."

text = " <|endoftext|> ".join((text1, text2))  # 用句子分隔符链接两个句子

print(text)  # 跟第一个一样,但不会报错了

print(tokenizer.encode(text))
print(tokenizer.decode(tokenizer.encode(text)))

import importlib
import tiktoken

print("tiktoken version:", importlib.metadata.version("tiktoken"))  # 验证下载并输出版本信息

tokenizer = tiktoken.get_encoding("gpt2")  # 初始化GPT2!
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})  # 输出分词的id,可以允许endoftext
print("-------------------------")
print(integers)

strings = tokenizer.decode(integers)
# 按照数字解码回去

print(strings)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)  # 读入了一个text并编码到enc_text里面
print(len(enc_text))

enc_sample = enc_text[50:]  # 从第五十一个开始向后

context_size = 4  # sliding windows4

x = enc_sample[:context_size]  # 开头四个
y = enc_sample[1:context_size + 1]  # 第二个开始的四个

print(f"x: {x}")
print(f"y:      {y}")

for i in range(1, context_size + 1):
    # 文本成输入 context,先输出有什么,然后输出下一个是什么编号
    context = enc_sample[:i]
    desired = enc_sample[i]

    print(context, "---->", desired)

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


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
dataloader = create_dataloader_v1(  # raw_text 中创建一个数据加载器 但是所批次
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
)

data_iter = iter(dataloader)  # 数据加载器 dataloader 转换为一个迭代器
first_batch = next(data_iter)
print(first_batch)

second_batch = next(data_iter)
print(second_batch)

input_ids = torch.tensor([2, 3, 5, 1])  # 要加入2,3,5,1的字符

vocab_size = 6  # 嵌入层需要支持的唯一标记的总数
output_dim = 3  # 嵌入向量的维度

torch.manual_seed(123)  # 用于设置随机数生成器的种子，确保结果的可复现性
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)  # 每行表示一个标记的嵌入向量。

print(embedding_layer.weight)

print(embedding_layer(torch.tensor([3])))

print(embedding_layer(input_ids))

print("--------------------------------------")

vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)  # 映射为tensor

max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

token_embeddings = token_embedding_layer(inputs)  # 调用token_embedding_layer将输入inputs映射为对应的嵌入向量。
print(token_embeddings.shape)

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
# 目的是为输入序列中的每个位置生成一个向量,表明位置信息

pos_embeddings = pos_embedding_layer(torch.arange(max_length))  # 生成一个连续整数的1D tensor

print(pos_embeddings.shape)
# print(pos_embeddings)

input_embeddings = token_embeddings + pos_embeddings  # 特征是词语信息跟位置信息的结合
print(input_embeddings.shape)