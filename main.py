import json
with open("Data/english-train.json") as f:
    content = json.loads(f.read())

print(type(content))
print(len(content))