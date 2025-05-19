import logging
import os
from typing import Dict, List

import pandas as pd


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def validate_inputs(dataset_path: str, selected_emotions: List[str]):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
    if not selected_emotions:
        raise ValueError("Selected emotions list cannot be empty")


def process_dataset(
    dataset_path: str,
    language_code: str,
    dataset_name: str,
    emotion_map: Dict[str, str],
    selected_emotions: List[str],
    file_processor,
) -> None:
    setup_logging()
    validate_inputs(dataset_path, selected_emotions)

    data = []
    try:
        data = file_processor(dataset_path, emotion_map, selected_emotions)
    except Exception as e:
        logging.error(f"Error processing {dataset_name} data: {str(e)}")
        raise

    df = pd.DataFrame(data, columns=["emotion", "audio_path"])
    dir = "./meta_csvs/"
    os.makedirs(dir, exist_ok=True)
    output_file = f"{dir}{language_code}_{dataset_name}.csv"
    df.to_csv(output_file, index=False)
    logging.info(
        f"Successfully processed {len(data)} files and saved to {output_file}"
    )
