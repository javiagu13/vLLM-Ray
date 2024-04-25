import requests

# Sending a request to each model
base_url = "http://127.0.0.1:8000"

# Request to Model A
response_a = requests.post(f"{base_url}/llama", json={"text": "How do I make pizza in 10 steps"})
print("Response from Model A:", response_a.text)#response_a.json()
