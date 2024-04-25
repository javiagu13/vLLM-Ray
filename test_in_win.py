import requests

# API endpoint URL
url = 'http://localhost:8000/query'

# Data to be sent in the request (input text)
data = {'text': 'Your input text goes here.'}

# Send POST request
response = requests.post(url, json=data)

# Print the response
print(response.json())
