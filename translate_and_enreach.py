import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

XAI_API_KEY = os.environ.get('XAI_API_KEY')
XAI_API_URL = "https://api.x.ai/v1"
XAI_MODEL = "grok-4-1-fast-reasoning"
XAI_TEMP = 0.7
XAI_MAXTOKENS = 3000
MOVIES_PATH = Path('./json-data/movies')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s.%(funcName)s: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


def get_data_from_json_file(file: str) -> dict:
    excepted_keys = (
        'name', 'year', 'runtime',
        'actors', 'director',
    )
    with open(file) as f:
        data = json.load(f)
    if any(key not in data for key in excepted_keys):
        raise ValueError(f'All of {excepted_keys} should be in data')
    return data


def save_translated_json(file: str, data: dict):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_xai_response(data: dict):
    data = json.dumps(data)

    system_message = (
        "Ты — автоматический переводчик с английского на русский язык. "
        "Строго соблюдай инструкции. "
        "Пользователь передаст json-файл фильма на английском языке. "
        "Твоя задача: перевести значения ключей 'writers', 'director', "
        "'actors', 'storyline', 'categories', 'name'. "
        "Для ключа 'name' переводи название фильма так, "
        "как оно официально используется "
        "в русском прокате, если известно, "
        "иначе оставляй оригинальное название. "
        "Не дословно переводить слова в названии. "
        "Необходимо немного расширить и уточнить перевод, "
        "чтобы 'storyline' был около 500 знаков. "
        "Если ключа 'storyline' нет, добавь его и придумай описание сюжета. "
        "Не используй название фильма в 'storyline'. "
        "В ответе должна быть ТОЛЬКО валидная json-структура."
    )

    client = OpenAI(api_key=XAI_API_KEY, base_url=XAI_API_URL)
    response = client.chat.completions.create(
        model=XAI_MODEL,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": data}
        ],
        temperature=XAI_TEMP,
        max_tokens=XAI_MAXTOKENS
    )

    response = response.choices[0].message.content
    return json.loads(response)


def process_file(file: str):
    try:
        data = get_data_from_json_file(file)
        translated_data = get_xai_response(data)
        save_translated_json(file, translated_data)
        logger.info(f'Processed {file}')
    except Exception as e:
        logger.error(f'Error {e} while proccessing {file}')


def main():
    files = list(MOVIES_PATH.rglob('*.json'))
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_file, file) for file in files]
    for future in as_completed(futures):
        future.result()


if __name__ == '__main__':
    main()
