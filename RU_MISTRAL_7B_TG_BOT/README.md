# 💡 Суть проекта  

Я попросил популярную нейросеть Deepseek придумать мне pet-проект. Я хотел дообучить какую-нибудь llm с помощью LoRA и использовать ее в качестве AI-ассистента в виде Телеграм бота. Deepseek должен был предложить llm, которую я буду обучать, чему я ее буду обучать, подобрать датасет для обучения, а также помочь с настройкой параметров и фиксом мелких ошибок. Я должен был направлять его, фильтровать его решения, писать основной код и исправлять основные ошибки. Таким образом, я использовал Deepseek не как программиста, а как генератора идей.

После дискусси была выбрана модель [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1).  
Целью было научить модель общаться на русском языке.  
Для обучения был выбран датасет [ru_turbo_alpaca](https://huggingface.co/datasets/IlyaGusev/ru_turbo_alpaca).  

### 📁 Структура проекта  
```
RU_MISTRAL_7B_TG_BOT/  
├── final_model/                       # Финальная дообученная модель  
├── model_training/  
│     ├── checking_dataset.ipynb       # Обработка датасета
│     ├── lora_training.py             # Дообучение модели с помощью LoRA 
│     ├── model_inference.py           # Тест модели после обучения
│     └── model_test_before_lora.py    # Тест модели до обучения
├── ru_turbo_alpaca_filtered/          # Датасет для обучения  
├── telegram_bot/  
│     └── bot.py                       # Телеграм-бот для обращения к модели  
├── after_lora.txt                     # Примеры ответов после обучения  
└── before_lora.txt                    # Примеры ответов до обучения  
```

### ⚒ Ход работы  

После того, как мы определились с задачей, необходимо было протестировать сырую модель на запросах, написанных на русском языке. Тест проводился в model_training/model_test_before_lora.py  
Результаты были записаны в файл before_lora.txt  
По результатам можно понять, что модель очень плохо понимает и обрабатывает запросы на русском языке.  

После этого в model_training/checking_dataset.ipynb был обработан и подготовлен датасет, сохраненный в ru_turbo_alpaca_filtered/  
Затем, после подбора параметров было проведено обучение в model_training/lora_training.py  
Финальная модель сохранена в final_model/  
Обучение проводилось на кластерном сервере и длилось около 12-ти часов. 

Результаты обучения были проверены на тех же вопросах, что и тест модели до обучения, в model_training/model_inference.py и записаны в after_lora.txt  
По результатам видно, что модель успешно понимает и обрабатывает русский язык, однако иногда допускает ошибки в склонениях, спряжениях и постановке неправильных окончаний в целом.  

Затем был создан Телеграм бот @mistral_ru_test_bot (сейчас он не работает, т.к. я не арендовываю сервер для него), в которого была интегрирована обученная модель. Бот откликается на команду /start и на любой запрос, который ему задают.  
В случае ошибки бот пишет "Произошла ошибка. Попробуйте позже." Также во время формирования ответа в статусе бота отображается "typing..."

На изображениях представлены результаты случайных запросов.

![image](https://github.com/user-attachments/assets/81ec06ff-740a-4cd9-8a72-fe7304038f89)
![image_2025-06-04_12-30-40](https://github.com/user-attachments/assets/262414d0-3023-4c08-9b7d-95d2af72352f)

Несмотря на то, что ассистент иногда галлюцинирует или делает орфографические ошибки, я считаю результат удачным для такого недолгого обучения. Модель поняла структуру построения предложений и значение большинства слов на русском языке.  

При желании использовать модель, вам необходимо скопировать мой репозиторий, установить зависимости, а также указать ваш Hugging Face токен в начале кода, который доступен для всех бесплатно.  
В случае создания бота, вам необходимо создать бота в Телеграм с помощью BotFather, скопировать мой репозиторий, установить зависимости и указать токен Hugging Face и токен созданного бота.  
