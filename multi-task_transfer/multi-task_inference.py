# ***************** Inference LLMs on Comics dataset *********************** #

import sys
import json
import torch

sys.path.append('../')

import pandas as pd

from PIL import Image
from pathlib import Path
from tqdm import tqdm
from llamafactory.chat import ChatModel
from llamafactory.extras.misc import torch_gc
from sklearn.metrics import classification_report
from transformers import AutoProcessor, MllamaForConditionalGeneration, AutoProcessor
from peft import PeftModel


BASE_MODEL = "unsloth/Llama-3.2-11B-Vision-Instruct"
processor = AutoProcessor.from_pretrained(BASE_MODEL)


args = dict(
    
  model_name_or_path=BASE_MODEL, # use bnb-4bit-quantized Llama-3-8B-Instruct model
  adapter_name_or_path="/Utilisateurs/umushtaq/emorec_work/lfac_setup/LlamaFactory/outputs/llama3_vision_lora_full",            # load the saved LoRA adapters  
  template="mllama",                     # same to the one in training
  
  finetuning_type="lora",                  # same to the one in training
  quantization_bit=4,                    # load 4-bit quantized model
  #device_map="auto"
)

model = ChatModel(args)


def run_inference(sample):
    
    #print(sample)
    # print("------------")
    # print(sample['messages'])
    # print("------------")
    # print(sample['messages'][0]['content']) # user instruction
    # print("------------")
    # print(sample['messages'][1]['content']) # assistant response
    #print("------------xxxx")
    
    message = [
        
        {
            "role": "user",
            "content": sample['messages'][0]['content']
        },  
        {
            "role": "assistant",
            "content": ''
        }
    ]
    
    image = sample["images"][0]
    
    
    
    with Image.open(image) as img:
        
        image = img.copy()   # force-load image into memory



    with torch.no_grad():
        
        generated_ids = model.chat(
            #messages=messages['content'], # type: ignore
            #sample['messages'],
            message,
            images=[image]
        )
        
    
    
    print(generated_ids[0].response_text)
    sys.exit()
    return generated_ids

outputs = []

def infer(data_list):
    
    for idx, sample in enumerate(tqdm(data_list, "Processing inferences ... ")): # type: ignore
        
        #data_sample = process_dataset(sample)
        #sample_output = {"id": idx, "prediction": generated_text}
        outputs.append(run_inference(sample))
        #outputs.append({"id": idx, "prediction": run_inference(sample)})
        #break
    
    return outputs

# DATA FILE


with open("/Utilisateurs/umushtaq/emorec_work/lfac_setup/files/llama3_vision_ds_test_lfc.jsonl", "r", encoding="utf-8") as f:
    test_data = [json.loads(line) for line in f if line.strip()]
    infer(test_data)


# Save all outputs to JSON
output_path = "inference_outputs_llama_full_lmf.json"

with open(output_path, "w", encoding="utf-8") as f:
    
    json.dump(outputs, f, ensure_ascii=False, indent=2)

print(f"\nSaved {len(outputs)} predictions to {output_path}")