
import requests

f = open("assets/Apple02.jpg", "rb")

url = 'http://localhost:80'
myobj = {'somekey': f}

x = requests.post(url, data = f)

print(x.text)