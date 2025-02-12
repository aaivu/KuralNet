from dataset_processor import process_dataset
import os
import logging

def process_ased_files(dataset_path, emotion_map, selected_emotions):
    data = []
    try:
        dir_list = os.listdir(dataset_path)
        dir_list.remove('.gitattributes')
        dir_list.remove('README.md')

        for directory in dir_list:
            dir_path = os.path.join(dataset_path, directory)
            sub_dir = os.listdir(dir_path)
            emotion = directory[2:]  # Remove 'E_' prefix
            
            if emotion not in selected_emotions:
                continue
                
            for file_name in sub_dir:
                file_path = os.path.join(dir_path, file_name)
                data.append([emotion, file_path])
                
    except Exception as e:
        logging.error(f"Error processing ASED files: {str(e)}")
        
    return data

if __name__ == "__main__":
    selected_emotions = ['Fear', 'Sadness', 'Happiness', 'Anger', 'Neutral']
    dataset_path = "/kaggle/input/amharic-speech-emotional-dataset-ased/"
    emotion_map = {
        'Fear': 'Fear',
        'Sadness': 'Sadness',
        'Happiness': 'Happiness',
        'Anger': 'Anger',
        'Neutral': 'Neutral'
    }

    process_dataset(
        dataset_path=dataset_path,
        language_code="am",
        dataset_name="ased",
        emotion_map=emotion_map,
        selected_emotions=selected_emotions,
        file_processor=process_ased_files
    )
