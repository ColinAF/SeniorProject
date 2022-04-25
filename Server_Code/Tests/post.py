
import requests

f = open("../../assets/example_images/example01.jpg", "rb")
#f = open("../../assets/datasets/fruit_train01/images/bowl044.jpg", "rb")

url = 'http://10.0.0.118:8000/produce_detector/'
myobj = {'somekey': 'f'}

x = requests.post(url, data = f)

print(x.text)