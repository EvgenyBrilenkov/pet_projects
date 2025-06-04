from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from huggingface_hub import login

login(token="HUGGING-FACE-TOKEN")

model_name = "mistralai/Mistral-7B-v0.1"

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=bnb_config
)

print("Model loaded")

queries = [
    "Объясни, что такое градиентный бустинг простыми словами",
    "Напиши код на Python для суммы двух чисел",
    "Как приготовить борщ? Пошагово.",
    "Кто такой Пелевин? Ответь в стиле его прозы"
]

f = open('./before_lora.txt', 'w')

for query in queries:
    prompt = f"""<s>[INST] <<SYS>>Ты — русскоязычный ассистент. Отвечай точно и профессионально.<</SYS>>
{query} [/INST]"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=1024)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Вопрос: {query}\nОтвет: {response}\n{'='*50}")
    f.write(f"Вопрос: {query}\nОтвет: {response}\n{'='*50}\n\n")