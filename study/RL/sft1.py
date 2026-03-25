import torch
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
model_path = "Qwen/Qwen3-0.6B"  # 或者是你的本地绝对路径
output_dir = "./models/qwen3_gsm8k_lora"

# 检测设备: Mac 使用 mps, 否则用 cpu
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# 2. 加载 Tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Qwen 默认可能需要设置 pad_token

# 1. 明确加载为 float32
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32, # 改为 float32 增加稳定性
    device_map={"": "mps"},    # 这种写法更兼容
    trust_remote_code=True
)

# 3. 配置 LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,  # 秩
    lora_alpha=32,  # 缩放系数
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]  # Qwen 常见的注意力层命名
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 4. 加载并预处理 GSM8K 数据集
dataset = load_dataset("gsm8k", "main", split="train[:1000]")  # 先取1000条做演示


# 2. 改进数据预处理函数
def process_func(example):
    MAX_LENGTH = 384  # 缩短长度节省内存
    # 严格按照 Qwen 的 ChatML 格式
    instruction = f"<|im_start|>user\n{example['question']}<|im_end|>\n<|im_start|>assistant\n"
    response = f"{example['answer']}<|im_end|>"

    # 编码输入和输出
    model_inputs = tokenizer(instruction, add_special_tokens=False)
    labels_ids = tokenizer(response, add_special_tokens=False)

    # 拼接
    input_ids = model_inputs["input_ids"] + labels_ids["input_ids"]
    attention_mask = model_inputs["attention_mask"] + labels_ids["attention_mask"]

    # 关键：Label 屏蔽掉指令部分，只计算回答部分的 Loss
    labels = [-100] * len(model_inputs["input_ids"]) + labels_ids["input_ids"]

    # 截断
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

# 5. 修正后的训练参数
args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,      # 压到最低，保证不崩
    gradient_accumulation_steps=16,     # 累积步数增加，保持总 Batch Size (1*16=16) 不变
    logging_steps=1,
    num_train_epochs=1,
    learning_rate=5e-5,                 # 调低学习率，防止出现之前日志里的 nan
    lr_scheduler_type="constant",
    warmup_steps=20,
    gradient_checkpointing=True,        # 关键！开启梯度检查点，大幅节省内存
    fp16=False,
    bf16=False,
    optim="adamw_torch",                # 明确指定优化器
    dataloader_pin_memory=False,
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

# 6. 开始训练
trainer.train()

# 7. 保存 LoRA 权重
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"训练完成，模型已保存至 {output_dir}")