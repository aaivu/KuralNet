import logging
import os
from typing import Dict, List

from dataset_processor import process_dataset


def process_telugu_files(dataset_path: str, emotion_map: Dict[str, str], selected_emotions: List[str]) -> List[List[str]]:
    data = []
    # Known problematic files to skip
    error_files = {
        '/kaggle/input/telugu-emotion-speech/telugu/sad/S45_SRI_C01_G2_D04_SPKF21_V1_SA4_MMM.wav',
        '/kaggle/input/telugu-emotion-speech/telugu/sad/S45_SRI_C03_G1_D03_SPKF21_V1_SA4_MMM.wav'
    }
    
    try:
        for emotion_dir in os.listdir(dataset_path):
            dir_path = os.path.join(dataset_path, emotion_dir)
            if not os.path.isdir(dir_path):
                continue
                
            emotion = emotion_map.get(emotion_dir)
            if not emotion or emotion not in selected_emotions:
                continue
                
            for file_name in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file_name)
                if file_path not in error_files:
                    data.append([emotion, file_path])
                    
    except Exception as e:
        logging.error(f"Error processing Telugu files: {str(e)}")
        
    return data

if __name__ == "__main__":
    selected_emotions = ['Fear', 'Sadness', 'Happiness', 'Anger', 'Neutral']
    dataset_path = "/kaggle/input/telugu-emotion-speech/telugu/"
    emotion_map = {
        'angry': "Anger",
        'sad': "Sadness",
        'suprised': "Surprise",
        'happy': "Happiness",
        'nuetral': "Neutral"  # Note: Original spelling maintained
    }

    process_dataset(
        dataset_path=dataset_path,
        language_code="te",
        dataset_name="telugu",
        emotion_map=emotion_map,
        selected_emotions=selected_emotions,
        file_processor=process_telugu_files
    )
