import json

path = "/Utilisateurs/umushtaq/emorec_work/mdlt_er/datasets/json_datasets/llama3_vision_ds_train_lfc.jsonl"

def verify_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                print(f"❌ Error on line {i}: {e}")
                return
    print(f"✅ JSONL file {path} is valid!")

verify_jsonl(path)


with open(path, "r", encoding="utf-8") as f:
    count = sum(1 for _ in f)

print("Total records:", count)


required_keys = {"messages", "images"}

with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        obj = json.loads(line)

        if not required_keys.issubset(obj):
            print(f"❌ Missing keys on line {i}")
            
data = []

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

print("Loaded:", len(data))

