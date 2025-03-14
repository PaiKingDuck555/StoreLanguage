from openai import OpenAI
from pdf2image import convert_from_path
import base64
import pandas as pd 
import os
from dotenv import load_dotenv 

load_dotenv() 

OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")

# Maximum batch size per API request
BATCH_SIZE = 5  # Adjust as needed
# Maximum token budget for the entire script
MAX_TOKENS = 1000  # Change this based on your budget
# Track total tokens used
total_tokens_used = 0 

# Initialize OpenAI Client (Required for new API version)
client = OpenAI(api_key=OPEN_AI_API_KEY)

# Sub-Function to encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# Function to track tokens and prevent overuse
def update_token_usage(response):
    global total_tokens_used
    tokens_used = response.usage.total_tokens
    total_tokens_used += tokens_used
    print(f"✅ Tokens Used: {tokens_used} | Total: {total_tokens_used}/{MAX_TOKENS}")

# Function to extract multiple Hindi texts in one batch using GPT-4 Vision
def extract_hindi_batch(image_paths):
    """Extracts Hindi text from multiple images in a single OpenAI API call while tracking token usage."""
    
    global total_tokens_used
    if total_tokens_used >= MAX_TOKENS:
        print("❌ Token limit reached! Stopping execution.")
        return []

    if not image_paths:
        return []

    # Change this line in extract_hindi_batch function
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Extract only Hindi text from the following images, keeping each extraction separate."},
            {"role": "user", "content": [
                {"type": "image_url", 
                "image_url": {"url": f"data:image/png;base64,{encode_image(img)}"}} 
                for img in image_paths
            ]}
        ]
    )

    update_token_usage(response)  # Track token usage

    # Extract and split responses
    response_text = response.choices[0].message.content.strip()
    hindi_texts = response_text.split("\n\n")  # Assume responses are separated by two newlines

    return hindi_texts

# Function to transliterate multiple Hindi texts in one batch while tracking token usage
def transliterate_hindi_batch(hindi_texts):
    """Transliterates multiple Hindi texts in a single API call to optimize costs while tracking token usage."""
    
    global total_tokens_used
    if total_tokens_used >= MAX_TOKENS:
        print("❌ Token limit reached! Stopping execution.")
        return []

    if not hindi_texts:
        return []

    formatted_prompt = "\n\n".join([f"Text {i+1}: {text}" for i, text in enumerate(hindi_texts)])

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Transliterate the following Hindi texts into Romanized Hindi, keeping them separated."},
            {"role": "user", "content": formatted_prompt}
        ]
    )

    update_token_usage(response)  # Track token usage

    # Extract and split responses
    response_text = response.choices[0].message.content.strip()
    hindi_texts = response_text.split("\n\n")  # Split multi-text responses


    return romanized_texts

# Convert PDF to images
pdf_path = "3-Idiots-Screenplay-Book.pdf"  # Change this to your PDF file path
images = convert_from_path(pdf_path)

hindi_texts = []
romanized_texts = []

# Process images in batches with token limit enforcement
for i in range(0, len(images), BATCH_SIZE):
    if total_tokens_used >= MAX_TOKENS:
        print("❌ Token limit reached! Stopping further API calls.")
        break  # Stop processing if we exceed the token budget

    batch_images = images[i:i + BATCH_SIZE]  # Select a batch of images

    # Save images temporarily
    image_paths = []
    for j, img in enumerate(batch_images):
        img_path = f"page_{i + j + 1}.png"
        image_paths.append(img_path)

    # Extract Hindi text in batch
    hindi_batch = extract_hindi_batch(image_paths)
    hindi_texts.extend(hindi_batch)

    # Transliterate Hindi text in batch
    romanized_batch = transliterate_hindi_batch(hindi_batch)
    romanized_texts.extend(romanized_batch)

# Print transliterated text
for item in romanized_texts:
    print(item)

# Save results to CSV for Anki
df = pd.DataFrame({"Hindi": hindi_texts, "Romanized Hindi": romanized_texts})
df.to_csv("hindi_translations.csv", index=False, encoding="utf-8")

print(f"✅ Done! Used {total_tokens_used}/{MAX_TOKENS} tokens.")
print("✅ Translations saved to hindi_translations.csv")
