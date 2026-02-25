import sys
import json
import tqdm
import torch
import random

from PIL import Image

from datasets import Dataset # type: ignore


from trl import SFTTrainer, SFTConfig # type: ignore
from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling
from transformers import AutoProcessor, BitsAndBytesConfig, MllamaForConditionalGeneration # FastLanguageModel for LLMs

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model_id = "/Utilisateurs/umushtaq/emorec_work/lfac_setup/LlamaFactory/saves/llamavision11B_sft_merged_full"

model = MllamaForConditionalGeneration.from_pretrained(

    "unsloth/Llama-3.2-11B-Vision-Instruct",
    torch_dtype=torch.bfloat16, # Use bfloat16 if possible for faster training. Switch to float16 if not supported on your GPU.
    device_map="auto",
)

#sys.exit(0)

processor = AutoProcessor.from_pretrained("unsloth/Llama-3.2-11B-Vision-Instruct")
collator = DataCollatorForVisionLanguageModeling(processor)

with open("/Utilisateurs/umushtaq/emorec_work/mdlt_er/datasets/json_datasets/mangavqa_train.jsonl", "r", encoding="utf-8") as f:
    train_data = [json.loads(line) for line in f if line.strip()]

with open("/Utilisateurs/umushtaq/emorec_work/mdlt_er/datasets/json_datasets/mangavqa_test.jsonl", "r", encoding="utf-8") as f:
    test_data = [json.loads(line) for line in f if line.strip()]

#train_data = random.sample(train_data, 10)
#test_data = random.sample(test_data, 3)
print("------------------------------")
print(type(train_data), type(test_data))
print(len(train_data), len(test_data))
#train_dataset = Dataset.from_list(train_data) # type: ignore
#print(type(train_dataset))
#print(train_dataset.features)
print("------------------------------")
train_data = [train_data[0]]
print(len(train_data))
print(type(train_data)) 
print(train_data)
print("------------------------------")

# Use a subset of the training data for quick testing. Remove this line for full training.
#test_data = random.sample(test_data, 3)

trainer = SFTTrainer(
    model = model,
    processing_class= processor,
    data_collator = collator, # Must use!
    train_dataset = test_data, # type: ignore
    args = SFTConfig(
        do_train=True,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        #max_steps = 30,
        num_train_epochs = 0.01, # Set this instead of max_steps for full training runs
        learning_rate = 2e-4,
        logging_steps = 25,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs/qwen7b",
        report_to = "none",     # For Weights and Biases
    ),
)

# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


