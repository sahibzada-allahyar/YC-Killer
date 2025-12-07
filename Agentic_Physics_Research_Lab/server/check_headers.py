import os
import requests
from dotenv import load_dotenv

load_dotenv()
app_id = os.getenv("WOLFRAM_ALPHA_APPID")

if not app_id:
    print("No App ID found")
    exit(1)

url = "http://api.wolframalpha.com/v2/query"
params = {"input": "integral of x^2", "appid": app_id}

try:
    resp = requests.get(url, params=params)
    print(f"Status: {resp.status_code}")
    print(f"Content-Type: '{resp.headers.get('Content-Type')}'")
except Exception as e:
    print(f"Error: {e}")
