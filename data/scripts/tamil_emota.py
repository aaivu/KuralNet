from dataset_processor import process_dataset
import os
import logging

def process_emota_files(dataset_path, emotion_map, selected_emotions):
    data = []
    for file_name in os.listdir(dataset_path):
        if file_name.endswith(".wav"):
            file_path = os.path.join(dataset_path, file_name)
            emo_abb = file_name.split('.')[0].split('_')[2].strip()
            emotion = emotion_map.get(emo_abb)
            if not emotion:
                logging.warning(f"Emotion {emo_abb} not found in emotion_map, file: {file_path}")
                continue
            if emotion in selected_emotions:
                data.append([emotion, file_path])
    return data

if __name__ == "__main__":
    selected_emotions = ['Fear', 'Sadness', 'Happiness', 'Anger', 'Neutral']
    dataset_path = "/kaggle/input/tamserdb/"
    emotion_map = {
        "ang": "Anger",
        "hap": "Happiness",
        "neu": "Neutral",
        "fea": "Fear",
        "sad": "Sadness"
    }

    process_dataset(
        dataset_path=dataset_path,
        language_code="ta",
        dataset_name="emota",
        emotion_map=emotion_map,
        selected_emotions=selected_emotions,
        file_processor=process_emota_files
    )

