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

# 1. 配置路径
model_path = "Qwen/Qwen3-0.6B"
output_dir = "./models/qwen3_gsm8k_full_lora"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# 2. 加载 Tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 保持 float32 保证 Mac 训练不爆或不产生 NaN
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
    r=16,  # 全量数据可以将 rank 稍微调大(8->16)以提升表达能力
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # 覆盖更多层，全量微调效果更好
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 4. 加载并预处理全量 GSM8K 数据集
dataset = load_dataset("gsm8k", "main", split="train")  # 获取全部训练数据


def process_func(example):
    MAX_LENGTH = 512  # GSM8K 部分解题过程较长，建议增加到 512

    raw_answer = example["answer"]
    reasoning, final_answer = raw_answer.split("####") if "####" in raw_answer else (raw_answer, "")

    # 保持 ChatML 格式
    instruction = f"<|im_start|>user\nQuestion: {example['question']}\n\nLet's solve this step by step:<|im_end|>\n<|im_start|>assistant\n"
    response = f"{reasoning.strip()}\n\nFinal Answer: {final_answer.strip()}<|im_end|>"

    model_inputs = tokenizer(instruction, add_special_tokens=False)
    labels_ids = tokenizer(response, add_special_tokens=False)

    input_ids = (model_inputs["input_ids"] + labels_ids["input_ids"])[:MAX_LENGTH]
    attention_mask = (model_inputs["attention_mask"] + labels_ids["attention_mask"])[:MAX_LENGTH]
    labels = ([-100] * len(model_inputs["input_ids"]) + labels_ids["input_ids"])[:MAX_LENGTH]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# 使用 num_proc 加速处理（根据 Mac 核心数调整，如 8）
tokenized_ds = dataset.map(process_func, remove_columns=dataset.column_names, num_proc=4)

# 5. 修正后的训练参数
args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,      # 18G 内存跑 0.6B 模型，2 是安全的
    gradient_accumulation_steps=16,     # 等效 batch 为 32
    logging_steps=10,
    num_train_epochs=3,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    gradient_checkpointing=True,
    fp16=False,                         # Mac MPS 必须为 False
    bf16=False,                         # Mac MPS 必须为 False
    optim="adamw_torch",
    weight_decay=0.01,
    # group_by_length=True,             # <-- 删除或注释掉这一行，避免 TypeError
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
    dataloader_pin_memory=False         # Mac 上建议设为 False
)

# 6. 开始训练
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

print(f"Total training samples: {len(tokenized_ds)}")
trainer.train()

# 7. 保存
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"训练完成，全量模型已保存至 {output_dir}")