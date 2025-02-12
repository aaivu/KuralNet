import logging
import os

from dataset_processor import process_dataset


def process_bangla_files(dataset_path, emotion_map, selected_emotions):
    data = []
    try:
        for file_name in os.listdir(dataset_path):
            file_path = os.path.join(dataset_path, file_name)
            emo_code = file_name.split('.')[0].split('_')[-1]
            
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
    selected_emotions = ['Fear', 'Sadness', 'Happiness', 'Anger', 'Neutral']
    dataset_path = "/kaggle/input/bangla-lang-ser-dataset/BANSpEmo Dataset/"
    emotion_map = {
        "01": "Anger",
        "02": "Disgust",
        "03": "Fear",
        "04": "Happiness",
        "05": "Sadness",
        "06": "Surprise"
    }

    process_dataset(
        dataset_path=dataset_path,
        language_code="bn",
        dataset_name="bangla",
        emotion_map=emotion_map,
        selected_emotions=selected_emotions,
        file_processor=process_bangla_files
    )
