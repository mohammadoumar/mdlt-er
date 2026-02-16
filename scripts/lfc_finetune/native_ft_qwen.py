import sys
import json
import tqdm
import torch
import random

from PIL import Image


from trl import SFTTrainer, SFTConfig # type: ignore
from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, MllamaForConditionalGeneration # FastLanguageModel for LLMs

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(

    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16, # Use bfloat16 if possible for faster training. Switch to float16 if not supported on your GPU.
    device_map="auto",
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
#processor = AutoProcessor.from_pretrained("unsloth/Llama-3.2-11B-Vision-Instruct")
collator = DataCollatorForVisionLanguageModeling(processor)

with open("/Utilisateurs/umushtaq/emorec_work/mdlt_er/datasets/json_datasets/emoart5k_llama_train.jsonl", "r", encoding="utf-8") as f:
    train_data = [json.loads(line) for line in f if line.strip()]

with open("/Utilisateurs/umushtaq/emorec_work/mdlt_er/datasets/json_datasets/emoart5k_llama_test.jsonl", "r", encoding="utf-8") as f:
    test_data = [json.loads(line) for line in f if line.strip()]

train_data = random.sample(train_data, 10)
test_data = random.sample(test_data, 3)


trainer = SFTTrainer(
    model = model,
    data_collator = collator, # Must use!
    train_dataset = train_data, # type: ignore
    args = SFTConfig(
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
        #output_dir = "outputs/qwen7b",
        report_to = "none",     # For Weights and Biases

        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_length = 2048,
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


model.eval()

outputs = []


for idx, sample in enumerate(tqdm.tqdm(test_data, "Processing inferences ... ")): # type: ignore
    
    image = sample["images"][0]
            
    instruction = sample["messages"][0]["content"]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction}
            ]
        }
    ]

    input_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True
    )

    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            use_cache=True,
            temperature=1.5,
            min_p=0.1,
        )

    # Decode generated text (excluding prompt)
    prompt_len = inputs["input_ids"].shape[1]
    generated_text = processor.decode(
        generated_ids[0][prompt_len:],
        skip_special_tokens=True
    ).strip()

    outputs.append({
        "id": idx,
        "prediction": generated_text
    })

# Save all outputs to JSON
output_path = "inference_outputs_qwen7B_full_corr_cxc.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(outputs, f, ensure_ascii=False, indent=2)

print(f"\nSaved {len(outputs)} predictions to {output_path}")
