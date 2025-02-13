import concurrent.futures
import os
import zipfile

import kaggle

from data.constant import DATASET
from multilingual_speech_emotion_recognition.utils.utils import _get_logger

# Note: To Run This Script Please Download Kaggle API Key as kaggle.json and put it under ~/.kaggle/

logger = _get_logger(name=__name__)

datasets = {dataset.value.name: dataset.value.url for dataset in DATASET}


def download_and_extract(name: str, dataset: str):
    """Downloads and extracts a single dataset."""
    logger.info(f"Downloading {name} dataset...")

    kaggle.api.dataset_download_files(
        dataset, path="SER_Datasets", unzip=False
    )

    dataset_name = dataset.split("/")[-1]
    dataset_zip = f"SER_Datasets/{dataset_name}.zip"

    with zipfile.ZipFile(dataset_zip, "r") as zip_ref:
        zip_ref.extractall(f"SER_Datasets/{name}")
    os.remove(dataset_zip)

    logger.info(f"Finished downloading {name}")


def download_datasets():
    """Download all datasets in parallel."""
    os.makedirs("SER_Datasets", exist_ok=True)
    logger.info("Start downloading datasets")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(download_and_extract, name, dataset)
            for name, dataset in datasets.items()
        ]
        concurrent.futures.wait(futures)

    logger.info("End downloading datasets")


if __name__ == "__main__":
    download_datasets()
