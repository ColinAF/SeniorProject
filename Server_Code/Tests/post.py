
import requests

f = open("../assets/Apple02.jpg", "rb")

url = 'http://10.0.0.118:80'
myobj = {'somekey': f}

x = requests.post(url, data = f)

print(x.text)