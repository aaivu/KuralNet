from dataset_processor import process_dataset
from typing import List, Dict, Tuple
from pathlib import Path
import os
import logging

def process_cafe_files(dataset_path: str, 
                      emotion_map: Dict[str, str], 
                      selected_emotions: List[str]) -> List[List[str]]:
    data = []
    dataset_path = Path(dataset_path)
    
    try:
        dir_list = os.listdir(dataset_path)
        # Remove metadata files
        metadata_files = {'Version-ChangeLog.txt', 'License.txt', 'Readme.txt'}
        dir_list = [d for d in dir_list if d not in metadata_files]

        for directory in dir_list:
            dir_path = dataset_path / directory
            for subdir in os.listdir(dir_path):
                directory_path = dir_path / subdir
                if not directory_path.is_dir():
                    continue
                    
                for filename in os.listdir(directory_path):
                    try:
                        emo_abb = filename.split('-')[1]
                        emotion = emotion_map[emo_abb]
                        
                        if emotion in selected_emotions:
                            file_path = str(directory_path / filename)
                            data.append([emotion, file_path])
                    except (KeyError, IndexError) as e:
                        logging.warning(f"Error processing file {filename}: {str(e)}")
                        continue
                        
    except Exception as e:
        logging.error(f"Error processing CaFE files: {str(e)}")
        raise
        
    return data

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    selected_emotions = ['Fear', 'Sadness', 'Happiness', 'Anger', 'Neutral']
    dataset_path = "/kaggle/input/cafe-dataset/"
    emotion_map = {
        "C": "Anger",
        "D": "Disgust",
        "J": "Happiness",
        "N": "Neutral",
        "P": "Fear",
        "S": "Surprise",
        "T": "Sadness"
    }

    process_dataset(
        dataset_path=dataset_path,
        language_code="fr",
        dataset_name="cafe",
        emotion_map=emotion_map,
        selected_emotions=selected_emotions,
        file_processor=process_cafe_files
    )