import os
from dotenv import load_dotenv
from google import genai

# 1. Load your key
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# 2. Connect to Google
client = genai.Client(api_key=API_KEY)

# 3. Ask Google for all model IDs and print the Gemma ones
print("🔍 Searching Google Servers for exact Gemma 4 IDs...\n")
for model in client.models.list():
    if "gemma" in model.name.lower():
        print(f"Dashboard Name: {model.display_name}")
        print(f"👉 EXACT STRING TO USE: '{model.name}'\n")