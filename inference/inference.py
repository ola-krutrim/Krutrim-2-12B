import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "krutrim-ai-labs/Krutrim-2-instruct" 
device = "cuda" if torch.cuda.is_available else "cpu"
# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt = "Hello"

inputs = tokenizer(prompt, return_tensors='pt').to(device)
inputs.pop("token_type_ids", None)

# Generate response
outputs = model.generate(
    **inputs,
    max_length=5
)

response = tokenizer.decode(outputs[0])

print(response)