from bs4 import BeautifulSoup
from tqdm import tqdm
import re
import yaml
import requests


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

def process_web_content():
    # Load Configuration from YAML File
    with open("../config/settings.yaml", 'r') as f:
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