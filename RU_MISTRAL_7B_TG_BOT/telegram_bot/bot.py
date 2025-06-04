import logging
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, CommandHandler
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
from huggingface_hub import login

login(token="HUGGING-FACE-TOKEN")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class AIAssistant:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16)


        self.model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-v0.1",
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=bnb_config
        )
        self.model = PeftModel.from_pretrained(self.model, "./final_model")
        self.model.eval()

    def generate_response(self, user_message):
        prompt = f"""<s>[INST] <<SYS>>Ты — русскоязычный ассистент. Отвечай точно и профессионально.<</SYS>>
{user_message} [/INST]"""
    
        inputs = self.tokenizer(
            prompt,
            padding=False,
            truncation=True,
            return_tensors="pt",
            max_length=1024
        ).to(self.model.device)
    
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                repetition_penalty=1.2,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False,
                num_beams=1,
            )
    
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        answer = full_text.split("[/INST]")[-1].strip()
        answer = answer.split("<|endoftext|>")[0].strip()
        answer = answer.split("</s>")[0].strip()

        return answer

assistant = AIAssistant()

async def start(update: Update, context) -> None:
    """Обработчик команды /start."""
    await update.message.reply_text("Привет! Я ваш AI-ассистент. Задайте мне вопрос.")

async def handle_message(update: Update, context) -> None:
    """Ответ на сообщение пользователя."""
    user_text = update.message.text
    logger.info(f"User {update.message.from_user.id} asked: {user_text}")

    try:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

        response = assistant.generate_response(user_text)

        await update.message.reply_text(response)
    
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        await update.message.reply_text("Произошла ошибка. Попробуйте позже.")

def main():
    """Запуск бота."""
    application = Application.builder().token("BOT-TOKEN").build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT, handle_message))

    application.run_polling()

if __name__ == "__main__":
    main()
