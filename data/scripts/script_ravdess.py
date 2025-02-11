from dataset_processor import process_dataset
import os
import logging

def process_ravdess_files(dataset_path, emotion_map, selected_emotions):
    data = []
    for directory in os.listdir(dataset_path):
        directory_path = os.path.join(dataset_path, directory)
        if os.path.isdir(directory_path):
            for filename in os.listdir(directory_path):
                if filename.endswith(".wav"):
                    emo_abb = filename.split('-')[2]
                    emotion = emotion_map.get(emo_abb)
                    if not emotion:
                        logging.warning(f"Emotion not found for {filename}")
                        continue
                    if emotion in selected_emotions:
                        file_path = os.path.join(directory_path, filename)
                        data.append([emotion, file_path])
    return data

if __name__ == "__main__":
    selected_emotions = ['Fear', 'Sadness', 'Happiness', 'Anger', 'Neutral']
    dataset_path = "/content/drive/MyDrive/Dataset/Datasets/RAVDESS/archive (7)/audio_speech_actors_01-24/"
    emotion_map = {
        "01": "Neutral",
        "02": "Calm",
        "03": "Happy",
        "04": "Sad",
        "05": "Angry",
        "06": "Fearful",
        "07": "Disgust",
        "08": "Surprised"
    }

    process_dataset(
        dataset_path=dataset_path,
        language_code="en",
        dataset_name="ravdess",
        emotion_map=emotion_map,
        selected_emotions=selected_emotions,
        file_processor=process_ravdess_files
    )