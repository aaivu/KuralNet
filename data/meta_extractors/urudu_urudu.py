import os

import pandas as pd

selected_emotions = ['Fear','Sadness','Happiness','Anger','Neutral']

urdu = "/kaggle/input/urdu-emotion-dataset/"
dir_list = os.listdir(urdu)

emotion_map={
    'Angry': "Anger",
    'Sad': "Sadness",
    'Happy': "Happiness",
    'Neutral': "Neutral"
}

data = []

for file_name in dir_list:
    dir_path = os.path.join(urdu, file_name)
    sub_dir = os.listdir(dir_path)
    for dire in sub_dir:
        file_path = os.path.join(dir_path, dire)
        try:
            emotion = emotion_map[file_name]
        except KeyError:
            print(f"Emotion {file_name} not found in emotion_map")
            continue
        if emotion in selected_emotions:
            data.append([emotion, file_path])

urdu_df = pd.DataFrame(data, columns=['emotion', 'audio_path'])
urdu.to_csv('ur_urudu.csv', index=False)
