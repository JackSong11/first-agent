import torch

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
print(inverse_vocab)
# 插入
# Suppose input is "every effort moves you", and the LLM
# returns the following logits for the next token:
next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)

probas = torch.softmax(next_token_logits, dim=0)
print(probas)
# softmax归一化
next_token_id = torch.argmax(probas).item()
print(next_token_id)
# 选个可能性最大
# The next generated token is then as follows:
print(inverse_vocab[next_token_id])

torch.manual_seed(123)
next_token_id = torch.multinomial(probas, num_samples=1).item()
print(inverse_vocab[next_token_id])


def print_sampled_tokens(probas):
    torch.manual_seed(123)  # Manual seed for reproducibility
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
    # 从概率分布 probas 中按照权重进行一次采样,并生成索引
    sampled_ids = torch.bincount(torch.tensor(sample))
    # 然后变成单词
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")


# 统计采样过程中每个词的出现频率
print_sampled_tokens(probas)


def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)


# 温度校正
# Temperature values
temperatures = [1, 0.1, 5]  # Original, higher confidence, and lower confidence
# 初始校正系数
# Calculate scaled probabilities
scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]
print(scaled_probas)

import matplotlib

matplotlib.use('MacOSX')  # 或者 'Qt5Agg'，取决于你系统安装了哪个
import matplotlib.pylab as plt
from matplotlib.ticker import MaxNLocator

# Plotting
# x = torch.arange(len(vocab))
# bar_width = 0.15
#
# fig, ax = plt.subplots(figsize=(5, 3))
# for i, T in enumerate(temperatures):
#     rects = ax.bar(x + i * bar_width, scaled_probas[i], bar_width, label=f'Temperature = {T}')
#
# ax.set_ylabel('Probability')
# ax.set_xticks(x)
# ax.set_xticklabels(vocab.keys(), rotation=90)
# ax.legend()
#
# plt.tight_layout()
# plt.savefig("temperature-plot.pdf")
# plt.show()
# # 一套经典的画图


top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k)
# topK采样
print("Top logits:", top_logits)
print("Top positions:", top_pos)

new_logits = torch.where(
    condition=next_token_logits < top_logits[-1],
    input=torch.tensor(float("-inf")),
    other=next_token_logits
)
# 不是前K遮蔽掉
print(new_logits)

topk_probas = torch.softmax(new_logits, dim=0)
print(topk_probas)



