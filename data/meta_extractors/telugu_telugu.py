import logging
import os

from data.constant import DATASET, EMOTION, SELECTED_EMOTIONS
from data.meta_extractors.dataset_processor import process_dataset

TELUGU_DATASET = DATASET.TELUGU_DATASET.value
EMOTION_MAP = {
    "angry": EMOTION.ANGER.value,
    "sad": EMOTION.SADNESS.value,
    "happy": EMOTION.HAPPINESS.value,
    "nuetral": EMOTION.NEUTRAL.value,  # Note: Original spelling maintained
    "suprised": EMOTION.SURPRISE.value,
}


def process_telugu_files(dataset_path, emotion_map, selected_emotions):
    data = []
    # Known problematic files to skip
    error_files = [ # sad directory
        "S45_SRI_C01_G2_D04_SPKF21_V1_SA4_MMM.wav",
        "S45_SRI_C03_G1_D03_SPKF21_V1_SA4_MMM.wav"
    ]

    try:
        for emotion_dir in os.listdir(dataset_path):
            dir_path = os.path.join(dataset_path, emotion_dir)
            if not os.path.isdir(dir_path):
                continue

            emotion = emotion_map.get(emotion_dir)
            if not emotion or emotion not in selected_emotions:
                continue
            files = os.listdir(dir_path)
            if emotion_dir == "sad":
                files = [f for f in files if f not in error_files]
            for file_name in files:
                file_path = os.path.join(dir_path, file_name)
                if file_path not in error_files:
                    data.append([emotion, file_path])

    except Exception as e:
        logging.error(f"Error processing Telugu files: {str(e)}")

    return data


if __name__ == "__main__":
    process_dataset(
        dataset_path=TELUGU_DATASET.path,
        language_code=TELUGU_DATASET.language,
        dataset_name=TELUGU_DATASET.name,
        emotion_map=EMOTION_MAP,
        selected_emotions=SELECTED_EMOTIONS,
        file_processor=process_telugu_files,
    )
