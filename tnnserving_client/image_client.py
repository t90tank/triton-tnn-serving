import requests
import json

r = requests.get('http://localhost:8000/v2/models/test')
print(r.text)
f = open("data.txt", 'r')
ss = f.read()
d = '{"inputs":[{"name":"IN","datatype":"UINT8","shape":[224,224,3],"data":'+ss+'}]}'
r = requests.post('http://localhost:8000/v2/models/test/infer', data=d)
print(r.text)

data = json.loads(r.text)

# print(data)

answer = data['outputs'][0]['data']

best = 0
for i in range(len(answer)):
  if answer[i] > answer[best]: 
    best = i

label = open("synset.txt", 'r')
result = ""
print(best)
for x in range(best+1):
  result = label.readline()
print(result)