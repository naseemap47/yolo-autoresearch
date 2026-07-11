from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import live_terminal, update_mtp_num_hidden_layers, create_modelfile
from peft import PeftModel
import torch
import subprocess
import os


model_id = "Qwen/Qwen3.5-4B-Base"
adapter_model_dir = "./qwen3.5_lora_adapter"
merged_output_dir = "./qwen3.5_merged"
outfile = "qwen3.5_q8.gguf"
outtype = "q8_0"

# Define your new updated values for the Modelfile
gguf = "./qwen3.5_q8.gguf"
temp = "0.7"
top_p = "0.9"
system_prompt = "You are a helpful AI assistant for a yolo training research task."

ollama_out_model_name = "qwen3.5:4b-yolo"

print("[INFO] Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto"  # Safe for memory; change to "auto" if you have plenty of VRAM
)

print("[INFO] Merging weights... (this might take a minute)")
model = PeftModel.from_pretrained(base_model, adapter_model_dir)
merged_model = model.merge_and_unload()  # Fuses adapter layers into base layers

print(f"[INFO] Saving merged model to {merged_output_dir}...")
merged_model.save_pretrained(merged_output_dir)
tokenizer.save_pretrained(merged_output_dir)
print("[INFO] Merge complete!")

# update the configuration file to set mtp_num_hidden_layers to 0
config_path = os.path.join(merged_output_dir, "config.json")
if os.path.exists(config_path):
    print(f"[INFO] Updating {config_path} to set mtp_num_hidden_layers to 0...")
    update_mtp_num_hidden_layers(config_path, new_value=0)
    print("[INFO] Configuration update complete!")
else:
    print(f"[WARNING] {config_path} not found. Skipping configuration update.")
    raise FileNotFoundError(f"{config_path} not found. Cannot update mtp_num_hidden_layers.")

# Convert the Merged Model to GGUF Format
if not os.path.exists("llama.cpp"):
    print("[INFO] Cloning llama.cpp repository...")
    live_terminal(["git", "clone", "https://github.com/ggml-org/llama.cpp.git"])
    print("[INFO] Building llama.cpp...")
    os.chdir("llama.cpp")
    print("[INFO] Installing dependencies...")
    live_terminal(["uv", "add", "-r", "requirements.txt", "--active"])
else:
    os.chdir("llama.cpp")
    print("[INFO] llama.cpp already exists. Skipping clone and build.")
# Convert your merged model to a compressed 8-bit GGUF file
print("[INFO] Converting merged model to GGUF format...")
live_terminal(["python", "convert_hf_to_gguf.py", f"../{merged_output_dir}", "--outfile", f"../{outfile}", "--outtype", outtype])
os.chdir("..")  # Ensure we're in the correct directory before running the conversion
print(f"[INFO] GGUF conversion complete! The model is saved as {outfile}.")

## Initialize and Create Ollama Model
# Create Modelfile for Ollama
create_modelfile(gguf, temp, top_p, system_prompt)

print(f"[INFO] Creating Ollama model '{ollama_out_model_name}' using the Modelfile...")
live_terminal(["ollama", "create", ollama_out_model_name, "-f", "Modelfile"])
print(f"[INFO] Ollama model '{ollama_out_model_name}' created successfully!")
