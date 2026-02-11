import sys
import torch

from pathlib import Path
from peft import PeftModel
from transformers import AutoProcessor, MllamaForConditionalGeneration

## PATHS

LORA_DIR = Path.cwd().parent / "lfc_saved_lora" / "EC35_mm_Qwen2.5-VL-32B-Instruct-bnb-4bit"
OUTPUT_DIR = Path.cwd().parent / "lfc_merged_models"
MODEL_OUTPUT_DIR = OUTPUT_DIR / "qwen_2.5_32B_vision_merged"

# Model path for the fine-tuned Llama 3.2 Vision model

base_model_id = "unsloth/Qwen2.5-VL-32B-Instruct"   # SAME base used for LoRA

# Load the Llama 3.2 Vision model
processor = AutoProcessor.from_pretrained(base_model_id)

base_model = MllamaForConditionalGeneration.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, LORA_DIR).to("cuda")

# Merge the LoRA weights into the base model
merged_model = model.merge_and_unload() # type: ignore

# 4 — save merged model
merged_model.save_pretrained(MODEL_OUTPUT_DIR)

# save processor/tokenizer too
processor = AutoProcessor.from_pretrained(base_model_id)
processor.save_pretrained(MODEL_OUTPUT_DIR)

print("✅ Merge complete — standalone model saved.")

# new_model = MllamaForConditionalGeneration.from_pretrained(
#     "./merged_llama_vision",
#     torch_dtype=torch.bfloat16,
#     device_map="auto"
# )


# # Verify model info

# print("---------------------------------")

# print(f"Model loaded: {base_model.base_model.__class__.__name__}")
# print(f"Model architecture: {base_model.__class__.__name__}")

# print("---------------------------------")

# print(f"Model loaded: {model.get_base_model().__class__.__name__}")
# print(f"Model architecture: {model.__class__.__name__}")
# print(f"Device: {next(model.parameters()).device}")

# print("---------------------------------")

# print(f"Model loaded: {merged_model.base_model.__class__.__name__}")
# print(f"Model architecture: {merged_model.__class__.__name__}")
# print(f"Device: {next(merged_model.parameters()).device}")

# print("---------------------------------")

# print(f"Model loaded: {new_model.base_model.__class__.__name__}")
# print(f"Model architecture: {new_model.__class__.__name__}")

# print("---------------------------------")