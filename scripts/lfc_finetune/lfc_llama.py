# ***************** Fine-Tuning LLMs on EmoArt dataset *********************** #

# ********** Libraries and GPU *************

import os
import sys
import json
import torch
import subprocess

sys.path.append('../')

from pathlib import Path


try:    
    assert torch.cuda.is_available() is True
    
except AssertionError:
    
    print("Please set up a GPU before using LLaMA Factory...")


CURRENT_DIR = Path.cwd()
SCRIPTS_DIR = CURRENT_DIR.parent
MDLT_DIR = CURRENT_DIR.parent.parent

DATASET_DIR = MDLT_DIR / "datasets" / "json_datasets"

LLAMA_FACTORY_DIR = MDLT_DIR / "LlamaFactory"

BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
LOGGING_DIR = SCRIPTS_DIR / "lfc_log_dir"
OUTPUT_DIR = MDLT_DIR / "mdemoart_ft_models" / f"""emoartft_llama3.2_11B_merged"""


# ****************** DATASET FILES ******************


# # *** TRAIN/TEST DATASET NAME/FILENAME *** #

train_dataset_name = f"""emoart5k_llama_train.jsonl"""
test_dataset_name = f"""emoart5k_llama_test.jsonl"""


train_dataset_file = DATASET_DIR / train_dataset_name
test_dataset_file = DATASET_DIR / test_dataset_name



# # *** TRAIN ARGS FILE PATH *** #

if not os.path.exists(os.path.join(SCRIPTS_DIR, "lfc_model_args")):
    os.mkdir(os.path.join(SCRIPTS_DIR, "lfc_model_args"))

train_file = SCRIPTS_DIR / "lfc_model_args" / f"""{train_dataset_name.split(".")[0].split("train")[0]}{BASE_MODEL.split("/")[1]}.json"""

# *** UPDATE dataset_info.json file in LLaMA-Factory *** #


dataset_info_line =  {
  "file_name": f"{train_dataset_file}",
  "formatting": "sharegpt",
  "columns": {
    "messages": "messages",
    "images": "images"
  },
  "tags": {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant",
    "system_tag": "system"
  }
}

with open(os.path.join(LLAMA_FACTORY_DIR, "data/dataset_info.json"), "r") as jsonFile:
    data = json.load(jsonFile)

data["emoart5k"] = dataset_info_line

with open(os.path.join(LLAMA_FACTORY_DIR, "data/dataset_info.json"), "w") as jsonFile:
    json.dump(data, jsonFile)
    
# # #     # # ************************** TRAIN MODEL ******************************#

NB_EPOCHS = 3

args = dict(
    
  stage="sft",                           # do supervised fine-tuning
  do_train=True,

  model_name_or_path=BASE_MODEL,         # use bnb-4bit-quantized Llama-3-8B-Instruct model
  num_train_epochs=NB_EPOCHS,            # the epochs of training
  output_dir=str(OUTPUT_DIR),                 # the path to save LoRA adapters
  overwrite_output_dir=True,             # overrides existing output contents

  dataset="emoart5k",                      # dataset name
  template="qwen2_vl",                     # use llama3 prompt template
  #train_on_prompt=True,
  val_size=0.1,
  max_samples=10000,                       # use 500 examples in each dataset

  finetuning_type="lora",                # use LoRA adapters to save memory
  lora_target="all",                     # attach LoRA adapters to all linear layers
  per_device_train_batch_size=2,         # the batch size
  gradient_accumulation_steps=4,         # the gradient accumulation steps
  lr_scheduler_type="cosine",            # use cosine learning rate scheduler
  loraplus_lr_ratio=16.0,                # use LoRA+ algorithm with lambda=16.0
  #temperature=0.5,
  
  warmup_ratio=0.1,                      # use warmup scheduler    
  learning_rate=5e-5,                    # the learning rate
  max_grad_norm=1.0,                     # clip gradient norm to 1.0
  
  fp16=True,                             # use float16 mixed precision training
  quantization_bit=4,  
  #freeze_multi_modal_projector=True,     # use 4-bit QLoRA  
  #use_liger_kernel=True,
  #quantization_device_map="auto",
  
  logging_steps=10,                      # log every 10 steps
  save_steps=5000,                       # save checkpoint every 1000 steps    
  logging_dir=str(LOGGING_DIR),
  
  # use_unsloth=True,
  report_to="none"                       # discards wandb

)

json.dump(args, open(train_file, "w", encoding="utf-8"), indent=2)

p = subprocess.Popen(["llamafactory-cli", "train", train_file], cwd=LLAMA_FACTORY_DIR)
p.wait()

#sys.exit(0)
