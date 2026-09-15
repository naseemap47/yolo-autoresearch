from bs4 import BeautifulSoup
from tqdm import tqdm
import re
import yaml
import requests
import subprocess
import sys
import json


# Remove Unwanted HTML Tags
def strip_html_noise(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Define tags that do not contain core textual knowledge
    unwanted_tags = [
        "script", "style", "noscript", "header", "footer", 
        "nav", "aside", "form", "button", "svg", "iframe"
    ]
    
    for tag in soup(unwanted_tags):
        tag.decompose()  # Completely removes the element from the parse tree
        
    return soup

# Extract Text and Normalize Whitespace
def extract_and_normalize(soup):
    # Extract plain text with a separator to keep layout boundaries
    # Pass strip=True to remove outer padding from HTML block elements before joining them
    # eg: removes "\n\n\n\n\n....."
    raw_text = soup.get_text(separator=" ", strip=True)

    # 1. Replace multiple spaces/tabs with a single space
    cleaned = re.sub(r'[ \t]+', ' ', raw_text)
    
    # 2. Collapse three or more consecutive newlines down to just two
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    # 3. Strip out blank spaces leading or trailing the text block
    cleaned = cleaned.strip()
    
    # 4. Remove excessive newlines
    cleaned = re.sub(r'\n+', '\n', cleaned)

    return cleaned

# Filter Boilerplate and Low-Quality Text
def is_high_quality(text):
    # Filter out sentences that are too short (likely menu buttons or dates)
    if len(text.split()) < 5:
        return False
        
    # Filter out common legal/web boilerplate patterns
    boilerplate_patterns = [
        r"copyright ©", r"all rights reserved", r"cookie policy", 
        r"terms of service", r"privacy policy", r"sign up for our newsletter"
    ]
    
    for pattern in boilerplate_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False
            
    return True

def filter_paragraphs(cleaned_text):
    # Split text into paragraphs and keep only high-quality ones
    paragraphs = cleaned_text.split('\n\n')
    filtered_paragraphs = [p for p in paragraphs if is_high_quality(p)]
    
    return "\n\n".join(filtered_paragraphs)

# Deduplicate and Standardize Text
import unicodedata

def final_polish(text):
    # Normalize Unicode characters (e.g., converts '\xa0' to normal spaces)
    text = unicodedata.normalize("NFKC", text)
    
    # Remove bracketed web references like [1], [2], [citation needed]
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[citation needed\]', '', text, flags=re.IGNORECASE)

    # Option A: Collapse everything into a single space (Best for continuous text streams)
    # clean_text = re.sub(r'\s{2,}', ' ', text)
    
    # Option B: Collapse multiple newlines into a single standard line break
    # clean_text = re.sub(r'\n{2,}', '\n', text)
    
    return text

def process_web_content(yaml_path: str = "../config/settings.yaml"):
    # Load Configuration from YAML File
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Configuration loaded URLs: {config['urls']}")

    scraped_data = []
    for url in tqdm(config['urls'], desc="Processing URLs"):
        response = requests.get(url)
        if response.status_code == 200:
            # soup = BeautifulSoup(response.text, 'html.parser')
            soup = strip_html_noise(response.text)
            # print(soup)
            cleaned = extract_and_normalize(soup)
            # print(cleaned)
            filter_para = filter_paragraphs(cleaned)
            # print(filter_para)
            text = final_polish(filter_para)
            # print(text)
            scraped_data.append(text)

    return scraped_data


def live_terminal(command: list):
    """
    Execute a command in the terminal and stream its output live.
    Args:
        command (list): The command to execute, provided as a list of strings.
    """
    # Start the process with stdout and stderr piped
    # Note: Git/cmd often sends progress updates to stderr
    process = subprocess.Popen(
        command, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True,
        bufsize=1  # Line buffered
    )

    # Read the output line by line as it happens
    for line in process.stdout:
        print(line, end="")  # end="" prevents double line breaks
        sys.stdout.flush()   # Forces the terminal to update immediately

    # Wait for the process to fully complete and get the exit code
    return_code = process.wait()
    
    if return_code == 0:
        print("\nCommand executed successfully!")
    else:
        print(f"\nCommand failed with exit code: {return_code}")

def update_mtp_num_hidden_layers(config_path, new_value: int = 0):
    # Open the file and decode the JSON data
    with open(config_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Update the value
    data['mtp_num_hidden_layers'] = new_value

    # Write the updated data back to the file
    with open(config_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

    print(f"Updated 'mtp_num_hidden_layers' to {new_value} in {config_path}")

def create_modelfile(gguf_path, temp, top_p, system_prompt):
    # Content wrapped in triple single-quotes to safely allow internal triple double-quotes
    modelfile_base_content = r'''# Point to your newly generated GGUF file
FROM {GGUF_PLACEHOLDER}

# Set Qwen's native ChatML template format so it understands system/user prompt tags
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""

# Set your default runtime parameters
PARAMETER temperature {TEMP_PLACEHOLDER}
PARAMETER top_p {TOP_P_PLACEHOLDER}
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"

# Optional: Give your model a default personality or instructions
SYSTEM """{SYSTEM_PLACEHOLDER}"""'''

    updated_content = (modelfile_base_content
                    .replace("{GGUF_PLACEHOLDER}", gguf_path)
                    .replace("{TEMP_PLACEHOLDER}", str(temp))
                    .replace("{TOP_P_PLACEHOLDER}", str(top_p))
                    .replace("{SYSTEM_PLACEHOLDER}", system_prompt))

    # Write the content to a file named 'Modelfile'
    with open('Modelfile', 'w', encoding='utf-8') as file:
        file.write(updated_content)

    print("Modelfile created successfully!")
