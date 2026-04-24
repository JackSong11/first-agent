import torch
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

# 1. 配置路径和参数
model_path = "Qwen/Qwen3-0.6B"
output_dir = "./models/qwen3_gsm8k_lora"

# 检测设备: Mac 使用 mps, 否则用 cpu
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# 2. 加载 Tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 显式使用 float32 以确保 MPS 上的数值稳定性
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map={"": device},
    trust_remote_code=True
)

# 3. 配置 LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, peft_config)

# 关键：确保 LoRA 层也是 float32，防止混合精度导致 nan
for param in model.parameters():
    param.data = param.data.to(torch.float32)

model.print_trainable_parameters()

# 4. 加载并预处理 GSM8K 数据集
dataset = load_dataset("gsm8k", "main", split="train[:1000]")


def process_func(example):
    MAX_LENGTH = 384

    # 解析 GSM8K 答案格式
    raw_answer = example["answer"]
    if "####" in raw_answer:
        reasoning, final_answer = raw_answer.split("####")
        reasoning = reasoning.strip()
        final_answer = final_answer.strip()
    else:
        reasoning = raw_answer
        final_answer = ""

    # 构造 Prompt：包含 ChatML 标记和思维链触发词
    instruction = f"<|im_start|>user\nQuestion: {example['question']}\n\nLet's solve this step by step:<|im_end|>\n<|im_start|>assistant\n"
    response = f"{reasoning}\n\nFinal Answer: {final_answer}<|im_end|>"

    model_inputs = tokenizer(instruction, add_special_tokens=False)
    labels_ids = tokenizer(response, add_special_tokens=False)

    input_ids = model_inputs["input_ids"] + labels_ids["input_ids"]
    attention_mask = model_inputs["attention_mask"] + labels_ids["attention_mask"]

    # 只对回答部分计算 Loss，指令部分用 -100 屏蔽
    labels = [-100] * len(model_inputs["input_ids"]) + labels_ids["input_ids"]

    # 截断处理
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


tokenized_ds = dataset.map(process_func, remove_columns=dataset.column_names)

# 打印一条数据检查 labels 是否全为 -100（如果是，说明截断太严重，需调大 MAX_LENGTH）
print(f"Check labels (should contain non -100 values): {tokenized_ds[0]['labels'][-10:]}")

# 5. 增强稳定性的训练参数
args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    logging_steps=1,
    num_train_epochs=1,
    learning_rate=2e-5,  # 进一步降低学习率
    lr_scheduler_type="cosine",  # 使用余弦退火，后期学习率更小更稳
    warmup_steps=50,  # 增加预热步数，让模型慢慢进入状态
    gradient_checkpointing=True,
    fp16=False,  # MPS 不支持标准的 fp16 训练，设为 False
    bf16=False,  # 同上，坚持使用 fp32
    optim="adamw_torch",
    adam_beta2=0.95,  # 优化器稳定性调整
    adam_epsilon=1e-5,  # 防止分母过小导致 nan
    save_total_limit=1,
    dataloader_pin_memory=False,
    report_to="none"
)

# 6. 开始训练
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

print("Starting training on MPS...")
trainer.train()

# 7. 保存
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"训练完成，模型已保存至 {output_dir}")