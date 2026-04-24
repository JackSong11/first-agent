from importlib.metadata import version

print("torch version:", version("torch"))
# 导入并确认库

import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],  # Your     (x^1)
     [0.55, 0.87, 0.66],  # journey  (x^2)
     [0.57, 0.85, 0.64],  # starts   (x^3)
     [0.22, 0.58, 0.33],  # with     (x^4)
     [0.77, 0.25, 0.10],  # one      (x^5)
     [0.05, 0.80, 0.55]]  # step     (x^6)
)
# 对于一句话中的每个单词定义了一个三维的向量

query = inputs[1]  # 2nd input token is the query

attn_scores_2 = torch.empty(inputs.shape[0])
print(attn_scores_2)
# 建立一个未初始化的张量来记录注意力得分
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
    # 相似性度量计算attention分数
    # 从公式上看也就是点乘

print(attn_scores_2)

# attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
# # 归一化,这里是属于加权式的归一化
#
# print("Attention weights:", attn_weights_2_tmp)
# print("Sum:", attn_weights_2_tmp.sum())


# def softmax_naive(x):
#     return torch.exp(x) / torch.exp(x).sum(dim=0)
#
#
# attn_weights_2_naive = softmax_naive(attn_scores_2)
#
# print("Attention weights:", attn_weights_2_naive)
# print("Sum:", attn_weights_2_naive.sum())
# ##用SoftMax做归一化, 处理好极端值
# # 有合理的梯度数据表现力

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())
# 用torch优化过的softmax对边缘值也挺友好的

query = inputs[1]  # 2nd input token is the query

context_vec_2 = torch.zeros(query.shape)
# 创造一个内容的零向量
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
    # 把不同内容的向量+起来

print(context_vec_2)

# attn_scores = torch.empty(6, 6)
# 建立个空表来储存相关联程度

# for i, x_i in enumerate(inputs):
#     for j, x_j in enumerate(inputs):
#         attn_scores[i, j] = torch.dot(x_i, x_j)
#         # 一点点计算相关性并输入表格
# print(attn_scores)
# # 事实上就是实现了两个单词之间的关联度列表输出

attn_scores = inputs @ inputs.T  # 牛逼，真牛逼，可以让AI举个数值的例子就懂了。
print(attn_scores)
# 有简单的方法整合方法计算

attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)
# 归一化处理

all_context_vecs = attn_weights @ inputs  # 这个也牛逼，真牛逼，可以让AI举个数值的例子就懂了。
print(all_context_vecs)
# 重复了上一个操作
print("Previous 2nd context vector:", context_vec_2)

x_2 = inputs[1]  # second input element
d_in = inputs.shape[1]  # the input embedding size, d=3
d_out = 2  # the output embedding size, d=2

torch.manual_seed(123)
# 固定随机种子确保可复现性

W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
# 初始化三个矩阵来存放
# 不要求梯度降低了复杂度

query_2 = x_2 @ W_query  # _2 because it's with respect to the 2nd input element
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
# 点积计算
print(query_2)

keys = inputs @ W_key
values = inputs @ W_value

print("keys.shape:", keys.shape)
print("values.shape:", values.shape)
# 中途检验下


keys_2 = keys[1]  # Python starts index at 0
print(query_2.shape, keys_2.shape)
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)

attn_scores_2 = query_2 @ keys.T  # All attention scores for given query
print(attn_scores_2)
# 计算注意力跟query值

d_k = keys.shape[1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k ** 0.5, dim=-1)
# 压缩函数, 有利于储存与比较
print(attn_weights_2)

context_vec_2 = attn_weights_2 @ values
print(context_vec_2)

import torch.nn as nn


class SelfAttention_v1(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
        # 定义QKV的随机矩阵

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T  # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        # 模型的训练传递
        context_vec = attn_weights @ values
        return context_vec


torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))


class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # 权重初始化

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        # Query跟Key的计算 得出初始的分数传递到后面进行归一化操作
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        context_vec = attn_weights @ values
        # 直接基于注意力对于文本计算
        return context_vec


torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))

# Reuse the query and key weight matrices of the
# SelfAttention_v2 object from the previous section for convenience
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T

attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
print(attn_weights)

# 用mask的数据重新算了一次
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length,
                                    context_length))  # torch.ones创建一个 全为 1 的矩阵，torch.tril意为 “下三角矩阵”，它会保留矩阵的主对角线及其下方的元素，而将对角线以上的所有元素置为 0。
# Mask矩阵,直接保留Diagonal下部分的,上部分掩盖掉
print(mask_simple)

masked_simple = attn_weights * mask_simple
print(masked_simple)
# 简单的效果图


row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)
# 掩码之后的softmax


mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
# 创建一个全1的三角,去上部分变成0
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
# 有掩码的地方变为负无穷
print(masked)

attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=-1)
print(attn_weights)

torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
# dropout rate of 50%丢包率doge
example = torch.ones(6, 6)
# create a matrix of ones满的6*6矩阵被1包圆了

print(dropout(example))
# 输出需要被放大相应的倍数,为了维持恒定

torch.manual_seed(123)
print(dropout(attn_weights))

batch = torch.stack((inputs, inputs), dim=0)
# 相同的tensor按照指定维度堆叠
print(batch.shape)  # 2 inputs with 6 tokens each, and each token has embedding dimension 3


class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        # 初始化定义网络结构和参数
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)  # New
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))  # New
        # 定义QKV并对进行dropout防止过拟合
        # 注册mask向量, 对未来进行负无穷的拟合

    def forward(self, x):
        b, num_tokens, d_in = x.shape  # New batch dimension b
        # 提取batch的大小、token的数量、跟宽度
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        # 进行运算计算
        # 结果 attn_scores 的形状是 (batch, num_tokens, num_tokens)。
        attn_scores = queries @ keys.transpose(1, 2)  # Changed transpose
        # 通过点积来计算attention的数值
        attn_scores.masked_fill_(  # New, _ ops are in-place
            self.mask.bool()[:num_tokens, :num_tokens],
            -torch.inf)  # `:num_tokens` to account for cases where the number of tokens in the batch is smaller than the supported context_size
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1  ## 缩放因子 √d，用于稳定梯度
        )
        # 在时间顺序上进行mask确保信息不会被泄露
        attn_weights = self.dropout(attn_weights)  # New
        # 防止过拟合的dropout处理方式
        context_vec = attn_weights @ values
        # 根据注意力权重计算上下文向量
        return context_vec


torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)

context_vecs = ca(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)


class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # 多个实例,每个都是一个头
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
    # 模型的训练


torch.manual_seed(123)

context_length = batch.shape[1]  # This is the number of tokens
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads=2
)

context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)


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


torch.manual_seed(123)

batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)

context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
