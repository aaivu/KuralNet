import logging
import os

from meta.meta_extractors.dataset_processor import process_dataset
from kuralnet.utils.constant import DATASET, EMOTION, SELECTED_EMOTIONS

AfrikaansEmotionalSpeechCorpus = DATASET.AFRIKAANSEMOTIONALSPEECHCORPUS.value
EMOTION_MAP = {
    "Anger": EMOTION.ANGER.value,
    "Anticipation": EMOTION.ANTICIPATION.value,
    "Trust": EMOTION.TRUST.value,
    "Sadness": EMOTION.SADNESS.value,
    "Surprise": EMOTION.SURPRISE.value,
    "Joy": EMOTION.JOY.value,
    "Disgust": EMOTION.DISGUST.value,
    "Fear": EMOTION.FEAR.value,
}


def process_afrikaans_emo_db_files(
    dataset_path, emotion_map, selected_emotions
):
    data = []
    for emotion in os.listdir(dataset_path):
        if emotion not in emotion_map:
            logging.warning(f"Emotion {emotion} not found in emotion_map")
            continue
        emo = emotion_map.get(emotion)
        if emo not in selected_emotions:
            continue
        for file in os.listdir(os.path.join(dataset_path, emotion)):
            data.append([emo, os.path.join(dataset_path, emotion, file)])

    return data


if __name__ == "__main__":
    process_dataset(
        dataset_path=AfrikaansEmotionalSpeechCorpus.path,
        language_code=AfrikaansEmotionalSpeechCorpus.language,
        dataset_name=AfrikaansEmotionalSpeechCorpus.name,
        emotion_map=EMOTION_MAP,
        selected_emotions=SELECTED_EMOTIONS,
        file_processor=process_afrikaans_emo_db_files,
    )
