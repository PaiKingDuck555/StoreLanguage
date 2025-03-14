import openai
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")

# Initialize OpenAI Client
client = OpenAI(api_key=OPEN_AI_API_KEY)

# Fetch available models
models = client.models.list()

# Print available model names
print("Available models:")
for model in models:
    print(model.id)