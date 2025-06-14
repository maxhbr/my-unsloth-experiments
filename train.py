from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

model, tokenizer = FastLanguageModel.from_pretrained(
  model_name="unsloth/Qwen3-14B",
  max_seq_length=2048,
  load_in_4bit=True,      # fits efficiently in VRAM
  full_finetuning=False   # LoRA mode
)

# Add LoRA adapter support
model = model.add_peft_adapter(
  r=8, alpha=32, target_modules=["q_proj","v_proj","k_proj","o_proj"]
)

training_args = TrainingArguments(
  output_dir="qwen3-lora",
  per_device_train_batch_size=2,
  gradient_accumulation_steps=4,
  learning_rate=2e-4,
  num_train_epochs=3,
  fp16=True,
  logging_steps=10,
  save_total_limit=2
)

trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=tokenized_dataset["train"],
  tokenizer=tokenizer
)

trainer.train()



# lora_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,
#     r=8,
#     lora_alpha=32,
#     lora_dropout=0.05,
#     bias="none",
#     target_modules=["q_proj", "v_proj"]  # Qwen uses these naming conventions
# )

# model_name = "Qwen/Qwen1.5-7B-Chat"
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# peft_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,
#     r=8,
#     lora_alpha=32,
#     lora_dropout=0.05,
#     bias="none"
# )

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     load_in_8bit=True,
#     device_map="auto",
#     trust_remote_code=True  # Needed for Qwen's custom model class
# )

# # Load your formatted dataset here (e.g., JSONL or Hugging Face dataset)
# dataset = load_dataset("json", data_files={"train": "your_chats.json"})

# # Tokenization logic here

# training_args = TrainingArguments(
#     output_dir="./qwen-lora-out",
#     per_device_train_batch_size=4,
#     gradient_accumulation_steps=2,
#     num_train_epochs=3,
#     learning_rate=2e-4,
#     fp16=True,
#     logging_dir="./logs",
#     save_strategy="epoch",
#     save_total_limit=2
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset["train"]
# )

# trainer.train()