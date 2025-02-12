import logging
import os

from dataset_processor import process_dataset

from data.constant import DATASET, EMOTION, SELECTED_EMOTIONS

BANSPEMO = DATASET.BANSPEMO.value
EMOTION_MAP = {
    "01": EMOTION.ANGER.value,
    "02": EMOTION.DISGUST.value,
    "03": EMOTION.FEAR.value,
    "04": EMOTION.HAPPINESS.value,
    "05": EMOTION.SADNESS.value,
    "06": EMOTION.SURPRISE.value,
}


def process_bangla_files(dataset_path, emotion_map, selected_emotions):
    data = []
    try:
        for file_name in os.listdir(dataset_path):
            file_path = os.path.join(dataset_path, file_name)
            emo_code = file_name.split(".")[0].split("_")[-1]

            try:
                emotion = emotion_map[emo_code]
            except KeyError:
                logging.warning(f"Emotion code {emo_code} not found in emotion_map")
                continue

            if emotion in selected_emotions:
                data.append([emotion, file_path])

    except Exception as e:
        logging.error(f"Error processing Bangla files: {str(e)}")

    return data


if __name__ == "__main__":
    process_dataset(
        dataset_path=BANSPEMO.path,
        language_code=BANSPEMO.language,
        dataset_name=BANSPEMO.name,
        emotion_map=EMOTION_MAP,
        selected_emotions=SELECTED_EMOTIONS,
        file_processor=process_bangla_files,
    )
