import logging
import os

from meta.meta_extractors.dataset_processor import process_dataset
from kuralnet.utils.constant import DATASET, EMOTION, SELECTED_EMOTIONS

HungarianEmotionalSpeechCorpus = DATASET.HUNGARIANEMOTIONALSPEECHCORPUS.value
EMOTION_MAP = {
    "B": EMOTION.SADNESS.value,
    "D": EMOTION.ANGER.value,
    "F": EMOTION.FEAR.value,
    "I": EMOTION.EXCITEMENT.value,
    "L": EMOTION.DISGUST.value,
    "M": EMOTION.SURPRISE.value,
    "O": EMOTION.JOY.value,
    "S": EMOTION.NEUTRAL.value,
}


def process_hesd_files(dataset_path, emotion_map, selected_emotions):
    data = []
    for file in os.listdir(dataset_path):
        if file.endswith(".wav"):
            file_path = os.path.join(dataset_path, file)
            emo_abb = file.split("_")[0][-1]
            emotion = emotion_map.get(emo_abb)
            if not emotion:
                logging.warning(f"Emotion not found for {file}")
                continue
            if emotion in selected_emotions:
                data.append([emotion, file_path])
    return data


if __name__ == "__main__":
    process_dataset(
        dataset_path=HungarianEmotionalSpeechCorpus.path,
        language_code=HungarianEmotionalSpeechCorpus.language,
        dataset_name=HungarianEmotionalSpeechCorpus.name,
        emotion_map=EMOTION_MAP,
        selected_emotions=SELECTED_EMOTIONS,
        file_processor=process_hesd_files,
    )
