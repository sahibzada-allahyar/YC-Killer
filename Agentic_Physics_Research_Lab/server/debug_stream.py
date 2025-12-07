import httpx
import json

def debug_stream():
    url = "http://localhost:8000/api/chat"
    data = {"message": "What is the integral of x^2 from 0 to 5?"}
    
    print(f"Connecting to {url}...")
    try:
        with httpx.stream("POST", url, json=data, timeout=60.0) as response:
            if response.status_code != 200:
                print(f"Error: Status code {response.status_code}")
                return

            print("Connected. Listening for events...")
            for line in response.iter_lines():
                if line:
                    print(f"RAW: {line}")
                    # Try to parse data
                    if line.startswith("data: "):
                        content = line[6:]
                        try:
                            json_content = json.loads(content)
                            print(f"PARSED: {json_content.get('type')} - {json_content.get('content')}")
                        except:
                            pass
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    debug_stream()
