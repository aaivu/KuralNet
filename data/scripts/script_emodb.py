from dataset_processor import process_dataset
import os
import logging

def process_emodb_files(dataset_path, emotion_map, selected_emotions):
    data = []
    try:
        for file_name in os.listdir(dataset_path):
            try:
                emo_code = file_name[-6:-5]  # Extract emotion code
                emotion = emotion_map.get(emo_code)
                
                if not emotion:
                    logging.warning(f"Unknown emotion code {emo_code} for file: {file_name}")
                    continue
                    
                if emotion in selected_emotions:
                    file_path = os.path.join(dataset_path, file_name)
                    data.append([emotion, file_path])
                    
            except Exception as e:
                logging.warning(f"Error processing file {file_name}: {str(e)}")
                continue
                
    except Exception as e:
        logging.error(f"Error processing EMoDB files: {str(e)}")
        
    return data

if __name__ == "__main__":
    selected_emotions = ['Fear', 'Sadness', 'Happiness', 'Anger', 'Neutral']
    dataset_path = "/kaggle/input/berlin-database-of-emotional-speech-emodb/wav/"
    emotion_map = {
        'W': 'Anger',
        'E': 'Disgust',
        'F': 'Happiness',
        'L': 'Boredom',
        'T': 'Sadness',
        'A': 'Fear',
        'N': 'Neutral'
    }

    process_dataset(
        dataset_path=dataset_path,
        language_code="de",
        dataset_name="emodb",
        emotion_map=emotion_map,
        selected_emotions=selected_emotions,
        file_processor=process_emodb_files
    )