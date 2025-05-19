import logging
import os

from meta.meta_extractors.dataset_processor import process_dataset
from kuralnet.utils.constant import DATASET, EMOTION, SELECTED_EMOTIONS

from kuralnet.utils.utils import stratified_sampling

KazakhEmKazakhEmotionalTTS = DATASET.KAZAKHEMOTIONALTTS.value
EMOTION_MAP = {
    "happy": EMOTION.HAPPINESS.value,
    "neutral": EMOTION.NEUTRAL.value,
    "angry": EMOTION.ANGER.value,
    "sad": EMOTION.SADNESS.value,
    "fear": EMOTION.FEAR.value,
    "surprise": EMOTION.SURPRISE.value,
}


def process_emo_kaz_files(dataset_path, emotion_map, selected_emotions):
    data = []
    dirtories = ["1263201035", "399172782", "805570882"]
    for dir in dirtories:
        for subdir in ["eval", "train"]:
            files = os.listdir(os.path.join(dataset_path, dir, subdir))
            for file_name in files:
                if file_name.endswith(".wav"):
                    file_path = os.path.join(
                        dataset_path, dir, subdir, file_name
                    )
                    emotion = file_name.split("_")[1]
                    emotion = emotion_map.get(emotion)
                    if not emotion:
                        logging.warning(
                            f"Emotion {emotion} not found in emotion_map, file: {file_path}"
                        )
                        continue
                    if emotion in selected_emotions:
                        data.append([emotion, file_path])

    data = stratified_sampling(data, 3000)
    return data


if __name__ == "__main__":
    process_dataset(
        dataset_path=KazakhEmKazakhEmotionalTTS.path,
        language_code=KazakhEmKazakhEmotionalTTS.language,
        dataset_name=KazakhEmKazakhEmotionalTTS.name,
        emotion_map=EMOTION_MAP,
        selected_emotions=SELECTED_EMOTIONS,
        file_processor=process_emo_kaz_files,
    )
