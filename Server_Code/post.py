
import requests

url = 'http://localhost:80'
myobj = {'somekey': 'somevalue'}

x = requests.post(url, data = myobj)

print(x.text)