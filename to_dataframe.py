import os
import zipfile
import pandas as pd


def combine_csv_from_zips(raw_data_dir: str = "./raw_data", output_file: str = "combined_data.csv") -> pd.DataFrame:
    """
    Обходит все .zip архивы в папке raw_data_dir, извлекает из них .csv файлы,
    объединяет данные в один датафрейм и сохраняет итоговый .csv файл в директории raw_data_dir.

    Args:
        raw_data_dir: Путь к директории с исходными .zip архивами.
        output_file: Имя итогового .csv файла.

    Return:
        Итоговый датафрейм, содержащий объединенные данные.
    """
    columns = ["UID", "UTC", "open", "close", "high", "low", "volume"]
    combined_data = []

    for file in os.listdir(raw_data_dir):
        if file.endswith(".zip"):
            zip_path = os.path.join(raw_data_dir, file)
            try:
                with zipfile.ZipFile(zip_path, "r") as archive:
                    for file_info in archive.infolist():
                        if file_info.filename.lower().endswith(".csv"):
                            try:
                                with archive.open(file_info) as csv_file:
                                    df = pd.read_csv(csv_file, sep=";", header=None, names=columns, index_col=False)
                                    combined_data.append(df)
                            except Exception as e:
                                print(f"Ошибка чтения .csv файла {file_info.filename} из {zip_path}: {e}")
            except Exception as e:
                print(f"Ошибка обработки .zip архива {zip_path}: {e}")

    if combined_data:
        final_df = pd.concat(combined_data)
        final_df["UTC"] = pd.to_datetime(final_df["UTC"], yearfirst=True, utc=True)
        final_df = final_df.sort_values(by=["UID", "UTC"]).reset_index(drop=True)
        output_path = os.path.join(raw_data_dir, output_file)
        final_df.to_csv(output_path, index=False, sep=";", encoding="utf-8")
        print(f"Объединенный .csv сохранен по пути: {output_path}")
        return final_df
    else:
        print("Не найдено .csv файлов для объединения.")
        return pd.DataFrame(columns=columns)