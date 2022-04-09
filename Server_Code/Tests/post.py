
import requests

f = open("assets/example_images/example01.jpg", "rb")

url = 'http://localhost:8000/produce_detector/'
myobj = {'somekey': f}

x = requests.post(url, data = [])

print(x.text)