import os
import time
from download import download_zip_archive
from to_dataframe import combine_csv_from_zips

TOKEN = os.environ["INVEST_TOKEN"]


def main():
    if not TOKEN:
        print("Ошибка: не найден API токен в переменной окружения INVEST_TOKEN.")
        return

    instrument_ids = [
        "a78b8349-a1dc-447d-9277-1d75826d089a",
        "926fdfbf-4b07-47c9-8928-f49858ca33f2",
        "8e2b0325-0292-4654-8a18-4f63ed3b0e09",
    ]
    years = [
        2023, 
        2024,
        ]

    for instrument_id in instrument_ids:
        for year in years:
            zip_filename = f"{instrument_id}_{year}.zip"
            file_path = os.path.join('./raw_data', zip_filename)

            if os.path.exists(file_path):
                print(f"Файл {zip_filename} уже существует, пропускаем скачивание.")
                print("При необходимости удалите файл и запустите скрипт еще раз.")
            else:
                zip_path = download_zip_archive(instrument_id, year, TOKEN)
                time.sleep(3)

    final_df = combine_csv_from_zips()
    if not final_df.empty:
        print("Данные успешно объединены!")
    else:
        print("Данные не объединены...")

if __name__ == "__main__":
    main()