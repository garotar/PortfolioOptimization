import os
import requests
from typing import Optional

TOKEN = os.environ["INVEST_TOKEN"]


def download_zip_archive(instrument_id: str, year: int, token: str, save_dir: str = "./raw_data") -> Optional[str]:
    """
    Загружает .zip архив данных по API для указанного идентификатора (instrument_id) и года (year),
    сохраняя его в директорию save_dir.

    Args:
        instrument_id: Идентификатор инструмента (параметр instrumentId в запросе).
        year: Год, за который запрашиваются данные.
        token: API токен для авторизации.
        save_dir: Директория для сохранения архива. По умолчанию './raw_data'.

    Return:
        Путь к сохраненному файлу или None, если произошла ошибка.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    url = "https://invest-public-api.tinkoff.ru/history-data"
    params = {"instrumentId": instrument_id, "year": year}
    headers = {"Authorization": f"Bearer {token}"}
    
    print(f"Начало загрузки: {url} с параметрами {params}")
    try:
        response = requests.get(url, headers=headers, params=params, stream=True)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Ошибка запроса для {url} с параметрами {params}: {e}")
        return None

    filename = f"{instrument_id}_{year}.zip"
    file_path = os.path.join(save_dir, filename)
    try:
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Данные успешно сохранены в файл: {file_path}")
        return file_path
    except IOError as e:
        print(f"Ошибка при сохранении файла {file_path}: {e}")
        return None