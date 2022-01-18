import time
import requests
url = 'https://us-central1-silver-asset-337912.cloudfunctions.net/function-test'
payload = {'message': 'Hello, General Yoda'}

for _ in range(10):
   r = requests.get(url, params=payload)
   print(r.content)

