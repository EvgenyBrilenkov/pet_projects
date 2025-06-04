from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
from huggingface_hub import login

login(token="HUGGING-FACE-TOKEN")
torch.cuda.empty_cache()

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

model_name = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map="auto",
                                             torch_dtype=torch.float16,
                                             quantization_config=bnb_config)

model = PeftModel.from_pretrained(model, "final_model")
model.eval()

queries = [
    "Объясни, что такое градиентный бустинг простыми словами",
    "Напиши код на Python для суммы двух чисел",
    "Как приготовить борщ? Пошагово.",
    "Кто такой Пелевин? Ответь в стиле его прозы"
]


def generate_answer(question):
    prompt = f"""<s>[INST] <<SYS>>Ты — русскоязычный ассистент. Отвечай точно и профессионально.<</SYS>>
{question} [/INST]"""
    
    inputs = tokenizer(
        prompt,
        padding=False,
        truncation=True,
        return_tensors="pt",
        max_length=1024,
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            num_beams=1,
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    answer = full_text.split("[/INST]")[-1].strip()
    answer = answer.split("<|endoftext|>")[0].strip()
    answer = answer.split("</s>")[0].strip()

    return answer

f = open('./after_lora.txt', 'w')

for i, question in enumerate(queries):
    answer = generate_answer(question)
    print(f"\nПример {i+1}:")
    print(f"Вопрос: {question}")
    print(f"Ответ: {answer}")
    print("-"*50)
    f.write(f"Вопрос: {question}\nОтвет: {answer}\n{'='*50}\n\n")