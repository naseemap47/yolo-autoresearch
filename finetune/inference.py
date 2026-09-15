from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch


# 1. Define paths
model_id = "Qwen/Qwen3.5-4B-Base"
adapter_model_dir = "./qwen3.5_lora_adapter"
prompt = "which yolo model is better for me and how can I find out ?"

# 2. Load the tokenizer from the base model
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 3. Load the actual BASE model in bfloat16 (Crucial for Qwen!)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,                      # <-- Must be the base model ID, not the adapter dir
    dtype=torch.bfloat16,    # <-- Native precision for Qwen 3.5
    device_map="auto"
)

# 4. Load and attach the LoRA adapter weights over the base model
model = PeftModel.from_pretrained(base_model, adapter_model_dir)
model.eval()  # Put the model in evaluation mode to turn off dropout layers

# 5. Prepare your prompt using Qwen's chat template
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# 6. Tokenize and move to the designated device
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 7. Generate text safely
with torch.no_grad():  # <-- Saves VRAM by preventing unnecessary gradient tracking
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=True,  # <-- REQUIRED to make temperature and top_p actually work!
        temperature=0.7,
        top_p=0.9
    )

# 8. Decode the output
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"Response:\n{response}")