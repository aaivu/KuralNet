import os
import zipfile

import kaggle

from multilingual_speech_emotion_recognition.utils.utils import _get_logger

# Note: To Run This Script Please Download Kaggle API Key as kaggle.json and put it under ~/.kaggle/


logger = _get_logger(name=__name__)

datasets = {
    "Amharic": "thanikansivatheepan/amharic-speech-emotional-dataset-ased",
    "Bangla": "thanikansivatheepan/bangla-lang-ser-dataset",
    "Cafe": "jubeerathan/cafe-dataset",
    "EMODB": "piyushagni5/berlin-database-of-emotional-speech-emodb",
    "EMOVO": "sourabhy/emovo-italian-ser-dataset",
    "ESD": "thanikansivatheepan/esd-dataset-fyp",
    "IEMOCAP1": "jubeerathan/iemocap-meta-data",
    "IEMOCAP2": "dejolilandry/iemocapfullrelease",
    "Kannada": "thanikansivatheepan/kannada-emo-speech-dataset",
    "MESD": "ashfaqsyed/mexican-emotional-speech-databasemesd",
    "RAVDESS": "uwrfkaggler/ravdess-emotional-speech-audio",
    "SUBESCO": "sushmit0109/subescobangla-speech-emotion-dataset",
    "EMOTA": "luxluxshan/tamserdb",
    "TELUGU": "jettysowmith/telugu-emotion-speech",
    "URDU": "kingabzpro/urdu-emotion-dataset",
}

os.makedirs("SER_Datasets", exist_ok=True)

logger.info("start downloading datasets")
for name, dataset in datasets.items():
    logger.info(f"downloading {name} dataset...")
    dataset_zip = f"SER_Datasets/{name}.zip"

    kaggle.api.dataset_download_files(dataset, path="SER_Datasets", unzip=False)

    with zipfile.ZipFile(dataset_zip, "r") as zip_ref:
        zip_ref.extractall(f"SER_Datasets/{name}")
    os.remove(dataset_zip)
logger.info("end downloading datasets")
