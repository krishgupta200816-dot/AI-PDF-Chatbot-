import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

client = genai.Client(api_key=api_key)

response = client.interactions.create(
    model=model_name,
    input="Explain Python in two simple sentences."
)

print(response)