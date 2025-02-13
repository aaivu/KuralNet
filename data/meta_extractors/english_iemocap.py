import logging
import os

from data.constant import DATASET, EMOTION, SELECTED_EMOTIONS
from data.meta_extractors.dataset_processor import process_dataset

IEMOCAP = DATASET.IEMOCAP.value
EMOTION_MAP = {
    "ang": EMOTION.ANGER.value,
    "hap": EMOTION.HAPPINESS.value,
    "sad": EMOTION.SADNESS.value,
    "neu": EMOTION.NEUTRAL.value,
    "fru": EMOTION.FRUSTRATION.value,
    "exc": EMOTION.EXCITEMENT.value,
    "fea": EMOTION.FEAR.value,
    "sur": EMOTION.SURPRISE.value,
    "dis": EMOTION.DISGUST.value,
    # "oth": EMOTION.OTHER.value,
    # "xxx": EMOTION.OTHER.value,
}

def process_iemocap_files(dataset_path, emotion_map, selected_emotions):
    data = []
    try:
        for session in os.listdir(dataset_path):
            session_path = os.path.join(dataset_path, session)
            for actor in os.listdir(session_path):
                actor_path = os.path.join(session_path, actor)
                for file_name in os.listdir(actor_path):
                    file_path = os.path.join(actor_path, file_name)
                    emotion = emotion_map.get(file_name.split("_")[2])
                    if emotion and emotion in selected_emotions:
                        data.append([emotion, file_path])
    except Exception as e:
        logging.error(f"Error processing IEMOCAP files: {str(e)}")

    return data

if __name__ == "__main__":
    process_dataset(
        dataset_path=IEMOCAP.path,
        language_code=IEMOCAP.language,
        dataset_name=IEMOCAP.name,
        emotion_map=EMOTION_MAP,
        selected_emotions=SELECTED_EMOTIONS,
        file_processor=process_iemocap_files,
    )