import ast
import json
import tqdm

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)

with open("/Utilisateurs/umushtaq/emorec_work/unsloth_setup/llama3_vision_dataset_test_uns_noPIL_full.jsonl", "r", encoding="utf-8") as f:
    test_data = [json.loads(line) for line in f if line.strip()]

# data is a list
print(type(test_data))   # <class 'list'>
print(len(test_data))

with open("/Utilisateurs/umushtaq/emorec_work/unsloth_setup/inference_outputs_qwen7B_full_corr.json", "r", encoding="utf-8") as f:
    results = json.load(f)# for line in f if line.strip()

# data is a list
print(type(results))   # <class 'list'>
print(len(results))

## Post-processing

grounds = []

for item in test_data:

    grounds.append(ast.literal_eval(item['messages'][1]['content'][0]['text'])["emotions"])
    
    
predictions = []
bad_indexes = []

for idx, item in enumerate(results):
    try:
        preds = ast.literal_eval(item["prediction"])["emotions"]
        predictions.append(preds)
    except Exception as e:
        print(f"[ERROR] index={idx} | prediction={item.get('prediction')}")
        print(f"        exception={e}")
        bad_indexes.append(idx)


for i in sorted(bad_indexes, reverse=True):
    
    grounds.pop(i)
    
bad_indexes = []

for idx, (g, p) in enumerate(zip(grounds, predictions)):
    if len(g) != len(p):
        print(idx, "--", len(g), len(p))
        bad_indexes.append(idx)
        
for idx in bad_indexes:
    if len(grounds[idx]) > len(predictions[idx]):
        grounds[idx] = grounds[idx][:len(predictions[idx])]
    elif len(predictions[idx]) > len(grounds[idx]):
        predictions[idx] = predictions[idx][:len(grounds[idx])]
        

grounds_flattened = [item for sublist in grounds for item in sublist]
predictions_flattened = [item for sublist in predictions for item in sublist]

bad_indexes = []

for idx, (g, p) in enumerate(zip(grounds_flattened, predictions_flattened)):
    if len(g) != len(p):
        print(idx, "--", len(g), len(p))
        bad_indexes.append(idx)
        
for i in sorted(bad_indexes, reverse=True):
    
    grounds_flattened.pop(i)

for i in sorted(bad_indexes, reverse=True):
    
    predictions_flattened.pop(i)

ground_vals = [item for sublist in grounds_flattened for item in sublist]
prediction_vals = [item for sublist in predictions_flattened for item in sublist]

print("Accuracy:", accuracy_score(ground_vals, prediction_vals))
print("Weighted F1:", f1_score(ground_vals, prediction_vals, average="weighted"))
print("Weighted Precision:", precision_score(ground_vals, prediction_vals, average="weighted"))
print("Weighted Recall:", recall_score(ground_vals, prediction_vals, average="weighted"))

print("\nClassification Report:\n")
print(classification_report(ground_vals, prediction_vals, digits=4))

print("\nConfusion Matrix:\n")
print(confusion_matrix(ground_vals, prediction_vals))

