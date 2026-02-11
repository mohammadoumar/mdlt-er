from transformers import AutoProcessor, MllamaForConditionalGeneration
import torch
from peft import PeftModel

# Model path for the fine-tuned Llama 3.2 Vision model

base_model_id = "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit"   # SAME base used for LoRA
lora_path = "/Utilisateurs/umushtaq/emorec_work/multimodal_er/EmoComics35/model_outputs/EC35_mm_Llama-3.2-11B-Vision-Instruct-bnb-4bit"  # Replace with your actual model path
output_path = "./merged_llama_vision"

# Load the Llama 3.2 Vision model
processor = AutoProcessor.from_pretrained(base_model_id)

base_model = MllamaForConditionalGeneration.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, lora_path)



# Verify model info
print(f"Model loaded: {model.get_base_model().__class__.__name__}")
print(f"Model architecture: {model.__class__.__name__}")
print(f"Device: {next(model.parameters()).device}")