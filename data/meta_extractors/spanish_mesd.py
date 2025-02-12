import logging
import os

from dataset_processor import process_dataset


def process_mesd_files(dataset_path, emotion_map, selected_emotions):
    data = []
    for file_name in os.listdir(dataset_path):
        emotion = file_name.split('_')[0]
        if emotion in selected_emotions:
            dir_path = os.path.join(dataset_path, file_name)
            data.append([emotion, dir_path])
    return data

if __name__ == "__main__":
    selected_emotions = ['Fear', 'Sadness', 'Happiness', 'Anger', 'Neutral']
    dataset_path = "/kaggle/input/mexican-emotional-speech-databasemesd/cy34mh68j9-5/Mexican Emotional Speech Database (MESD)/"
    
    process_dataset(
        dataset_path=dataset_path,
        language_code="es",
        dataset_name="mesd",
        emotion_map={},  # MESD doesn't need mapping
        selected_emotions=selected_emotions,
        file_processor=process_mesd_files
    )

