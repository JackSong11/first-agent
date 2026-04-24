import torch
import torch.nn as nn


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, num_classes):
        super(TransformerClassifier, self).__init__()

        # 1. 词嵌入层：将单词索引转换为向量
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 2. 位置编码 (简化版：直接使用可学习的参数)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 100, embed_dim))

        # 3. Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. 全连接分类层
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x 形状: (batch_size, seq_len)
        batch_size, seq_len = x.shape

        # 生成词向量并加上位置信息
        out = self.embedding(x) + self.pos_embedding[:, :seq_len, :]

        # Transformer 需要的输入形状通常是 (seq_len, batch_size, embed_dim)
        out = out.permute(1, 0, 2)

        # 经过 Transformer 编码
        out = self.transformer_encoder(out)

        # 取最后一个时间步或者平均值作为句子的表示
        sentence_vector = out.mean(dim=0)

        # 最后分类
        logits = self.fc(sentence_vector)
        return logits


# --- 模拟使用 ---
# 假设词汇表大小为1000，每个词映射为32维，2个注意力头，2层Transformer
model = TransformerClassifier(vocab_size=1000, embed_dim=32, num_heads=2, hidden_dim=64, num_layers=2, num_classes=2)

# 模拟一个 batch 的数据：2个句子，每句长5个词
dummy_input = torch.randint(0, 1000, (2, 5))
output = model(dummy_input)

print(f"输出形状: {output.shape}")  # (2, 2) -> 对应两个句子的二分类结果
