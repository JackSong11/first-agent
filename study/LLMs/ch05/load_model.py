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

token_ids = generate(
    model=model.to(device),
    idx=text_to_token_ids("Every effort moves", tokenizer).to(device),
    max_new_tokens=30,
    context_size=256,
    top_k=1,
    temperature=1.0
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
