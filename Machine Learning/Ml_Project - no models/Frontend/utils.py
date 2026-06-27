import requests
from dotenv import load_dotenv
import os 
load_dotenv()  # Load environment variables from .env file

base_url = os.getenv("backend_url")

def call_backend_api(text):
    url = f"{base_url}/predict"
    payload = {"text": text}
    response = requests.post(url, json=payload)
    return response.json()