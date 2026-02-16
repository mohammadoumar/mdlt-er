import sys
import json
import tqdm
import torch

from PIL import Image

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, MllamaForConditionalGeneration # FastLanguageModel for LLMs
from trl import SFTTrainer, SFTConfig # type: ignore
#from unsloth.trainer import UnslothVisionDataCollator
from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #"unsloth/Llama-3.2-11B-Vision-Instruct",
    #"unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",
    #"/Utilisateurs/umushtaq/emorec_work/mdlt_er/lfc_merged_models/llama_3.2_11B_vision_merged",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    #torch_dtype=torch.bfloat16, # Use bfloat16 if possible for faster training. Switch to float16 if not supported on your GPU.
    #config=quant_config, # Use 4bit to reduce memory use. False for 16bit LoRA. # type: ignore
    #use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
    #rope_scaling={"type": "linear", "factor": 512/2048}
    device_map="auto",
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
#processor = AutoProcessor.from_pretrained("unsloth/Llama-3.2-11B-Vision-Instruct")
collator = DataCollatorForVisionLanguageModeling(processor)

# model = FastVisionModel.get_peft_model(
#     model,
#     finetune_vision_layers     = True, # False if not finetuning vision layers
#     finetune_language_layers   = True, # False if not finetuning language layers
#     finetune_attention_modules = True, # False if not finetuning attention layers
#     finetune_mlp_modules       = True, # False if not finetuning MLP layers

#     r = 16,           # The larger, the higher the accuracy, but might overfit
#     lora_alpha = 16,  # Recommended alpha == r at least
#     lora_dropout = 0,
#     bias = "none",
#     random_state = 3407,
#     use_rslora = False,  # We support rank stabilized LoRA
#     loftq_config = None, # And LoftQ
#     # target_modules = "all-linear", # Optional now! Can specify a list if needed
# )

#sys.exit(0) # Remove this to run training and inference!

with open("/Utilisateurs/umushtaq/emorec_work/mdlt_er/datasets/json_datasets/emoart5k_llama_train.jsonl", "r", encoding="utf-8") as f:
    train_data = [json.loads(line) for line in f if line.strip()]

# data is a list
#print(type(train_data))   # <class 'list'>
#print(len(train_data))

# with open("/Utilisateurs/umushtaq/emorec_work/unsloth_setup/llama3_vision_dataset_test_uns_noPIL_full.jsonl", "r", encoding="utf-8") as f:
#     test_data = [json.loads(line) for line in f if line.strip()]

# data is a list
#print(type(test_data))   # <class 'list'>
#print(len(test_data))

# def process_dataset(ds):
    
#     for item in tqdm.tqdm(ds):
        
#         image_path = item['messages'][0]['content'][0]['image']
#         #image = Image.open(image_path)

#         with Image.open(image_path) as img:
            
#             image = img.copy()   # force-load image into memory

#         item['messages'][0]['content'][0]['image'] = image
        
#     return ds

# #train_data = random.sample(train_data, 10)
# #test_data = random.sample(test_data, 3)

# train_dataset = process_dataset(train_data)
# #test_dataset = process_dataset(test_data)


# print("Train samples: " + str(len(train_dataset)))
#print("Test samples: " + str(len(test_dataset)))

#FastVisionModel.for_training(model) # Enable for training!

trainer = SFTTrainer(
    model = model,
    #tokenizer = tokenizer, # type: ignore
    data_collator = collator, # Must use!
    train_dataset = train_data, # type: ignore
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        #max_steps = 30,
        num_train_epochs = 0.1, # Set this instead of max_steps for full training runs
        learning_rate = 2e-4,
        logging_steps = 25,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs/qwen7b",
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


#sys.exit(0) # Remove this to run inference!

#FastVisionModel.for_inference(model) # Enable for inference!

model.eval()

outputs = []


for idx, sample in enumerate(tqdm.tqdm(test_dataset, "Processing inferences ... ")): # type: ignore
    
    # print(f"\nProcessing sample {idx}...")

    # Extract image and instruction
    image = sample["messages"][0]["content"][0]["image"]
    
    # with Image.open(image) as img:
            
    #         image = img.copy()   # force-load image into memory
            
    instruction = sample["messages"][0]["content"][1]["text"]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction}
            ]
        }
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    # Streamer (prints output live)
    # text_streamer = TextStreamer(
    #     tokenizer,
    #     skip_prompt=True,
    #     skip_special_tokens=True
    # )

    with torch.no_grad():
        
        generated_ids = model.generate(
            **inputs,
            #streamer=text_streamer,
            max_new_tokens=512,
            use_cache=True,
            temperature=1.5,
            min_p=0.1,
        )

    # Decode generated text (excluding prompt)
    prompt_len = inputs["input_ids"].shape[1]
    generated_text = tokenizer.decode(
        generated_ids[0][prompt_len:],
        skip_special_tokens=True
    ).strip()

    outputs.append({
        "id": idx,
        #"instruction": instruction,
        "prediction": generated_text
    })

# Save all outputs to JSON
output_path = "inference_outputs_qwen7B_full_corr.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(outputs, f, ensure_ascii=False, indent=2)

print(f"\nSaved {len(outputs)} predictions to {output_path}")
