from utils import process_web_content
from dotenv import load_dotenv
import yaml
import requests
import re
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import os


# Load HF Token from .env file if it exists
if os.path.exists(".env"):
    load_dotenv()
    print("[INFO] Loaded environment variables from .env file.")
else:
    print("[INFO] No .env file found. Proceeding without loading environment variables.")

# ==========================================
# 0. Model
# ==========================================
model_id = "Qwen/Qwen3.5-4B-Base"
output_dir="./qwen3.5_lora_finetuning"
save_pretrained = "./qwen3.5_lora_adapter"
epochs = 10

# ==========================================
# 1. LOAD YOUR CLEANED BS4 DATA
# ==========================================
scraped_data = process_web_content()
cleaned_data = [{"text": doc} for doc in scraped_data]
print(f"[INFO] Number of cleaned documents: {len(cleaned_data)}")

dataset = Dataset.from_list(cleaned_data)
print(f"[INFO] Dataset: {dataset}")

# ==========================================
# 2. CONFIGURATOR TOKENIZER & MODEL
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Fix Qwen padding conflict with <|endoftext|>
if "<|vision_pad|>" in tokenizer.get_vocab():
    tokenizer.pad_token = "<|vision_pad|>"
else:
    tokenizer.pad_token = "<|extra_0|>"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    dtype="bfloat16", # Essential for Qwen 3.5 architecture
)

# ==========================================
# 3. DEFINE LORA CONFIGURATION
# ==========================================
peft_config = LoraConfig(
    r=16,                       # Rank: higher means more capacity, 16 or 32 is optimal for pre-training
    lora_alpha=32,              # Scaling factor (usually 2x Rank)
    target_modules=[            # Target all linear layers for comprehensive domain learning
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA parameters to the base model
model = get_peft_model(model, peft_config)
# Displays the percentage of parameters being trained
model.print_trainable_parameters()

# ==========================================
# 4. DEFINE TRAINING ARGUMENTS
# ==========================================
sft_config = SFTConfig(
    output_dir=output_dir,
    dataset_text_field="text",
    max_length=4096,        # Context window packing size
    learning_rate=2e-4,         # LoRA requires a slightly higher learning rate than full fine-tuning
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=epochs,
    weight_decay=0.01,
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    bf16=True,                  # Set to False and fp16=True if your GPU is older than Ampere (e.g., V100)
    fp16=False,
    packing=True,              # Packs raw texts without padding
)

# ==========================================
# 5. INITIALIZE TRL SFTTRAINER WITH PACKING
# ==========================================
# Initialize the trainer
print("[INFO] Initializing SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    args=sft_config, # Pass SFTConfig into the args parameter
)

# ==========================================
# 6. START PRE-TRAINING
# ==========================================
print("[INFO] Starting LoRA fine-tuning...")
trainer.train()
print("[INFO] LoRA fine-tuning completed.")

# Save the adapter weights separately at the end
trainer.model.save_pretrained(save_pretrained)
print(f"[INFO] LoRA adapter weights saved to {save_pretrained}.")