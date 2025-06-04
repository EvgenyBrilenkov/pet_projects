from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch
from peft import LoraConfig, get_peft_model
from huggingface_hub import login

torch.cuda.empty_cache()
login(token="HUGGING-FACE-TOKEN")

dataset = load_from_disk("./ru_turbo_alpaca_filtered")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16)

model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)

lora_config = LoraConfig(
    r=32,
    lora_alpha=128,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="lora_only",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
print(f"Train size: {len(split_dataset['train'])}, Test size: {len(split_dataset['test'])}")

training_args = TrainingArguments(
    output_dir="./mistral-7b-ru-lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    label_names=["input_ids"],
    remove_unused_columns=False,
    num_train_epochs=3,
    eval_strategy="steps",
    eval_steps=200,
    save_steps=400,
    logging_steps=50,
    learning_rate=2e-5,
    fp16=False,
    bf16=True,
    optim="adamw_bnb_8bit",
    report_to="tensorboard",
    load_best_model_at_end=True
)

collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    data_collator=collator,
    tokenizer=tokenizer
)

print("Starting training...")

trainer.train()

model.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")