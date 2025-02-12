from dataset_processor import process_dataset
import os
import logging

def process_esd_files(dataset_path, emotion_map, selected_emotions):
    data = []
    dir_list = os.listdir(dataset_path)
    if '.DS_Store' in dir_list:
        dir_list.remove('.DS_Store')
    dir_list.sort()
    dir_list = dir_list[:10]

    for directory in dir_list:
        dir_path = os.path.join(dataset_path, directory)
        sub_dir = os.listdir(dir_path)
        if '.DS_Store' in sub_dir:
            sub_dir.remove('.DS_Store')
        for dir in sub_dir:
            if dir.endswith(".txt"):
                continue
            directory_path = os.path.join(dir_path, dir)
            if os.path.isdir(directory_path):
                for filename in os.listdir(directory_path):
                    if filename.endswith(".wav"):
                        emo_abb = filename.split('_')[1].split('.')[0][-2:]
                        if (21 <= int(emo_abb) <= 50):
                            emotion = dir
                            file_path = os.path.join(directory_path, filename)
                            if emotion in selected_emotions:
                                data.append([emotion, file_path])
    return data

if __name__ == "__main__":
    selected_emotions = ['Fear', 'Sadness', 'Happiness', 'Anger', 'Neutral']
    dataset_path = "/content/drive/MyDrive/Dataset/Datasets/ESD/Emotion Speech Dataset/"

    process_dataset(
        dataset_path=dataset_path,
        language_code="zh",
        dataset_name="esd",
        emotion_map={},  # ESD doesn't need mapping
        selected_emotions=selected_emotions,
        file_processor=process_esd_files
    )