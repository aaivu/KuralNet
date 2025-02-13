import os

import pandas as pd

from multilingual_speech_emotion_recognition.preprocessing.feature_extractor.handcrafted_feature import (
    extract_chroma_stft, extract_melspectrogram, extract_mfcc, extract_rmse,
    extract_zcr)
from multilingual_speech_emotion_recognition.utils.utils import (_get_logger,
                                                                 load_audio,
                                                                 load_csv)

logger = _get_logger(__name__)


def extract_feature(extractor, meta_path, feature_type, sr=16000):
    logger.info(f"Feature extraction {feature_type} of {meta_path} started.")

    meta_csv = load_csv(meta_path)

    result_data = []

    for _, row in meta_csv.iterrows():
        emotion = row["emotion"]
        audio_path = row["audio_path"]

        try:
            audio, _sample_rate = load_audio(audio_path, sr)
            feat = extractor(audio, sr)
            result_data.append([emotion] + feat.tolist())
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")

    feature_names = [f"feature_{i}" for i in range(len(result_data[0]) - 1)]
    columns = ["emotion"] + feature_names

    result_df = pd.DataFrame(result_data, columns=columns)
    dir = os.path.join(meta_path.split("/")[0], "features")
    os.makedirs(dir, exist_ok=True)

    output_path = os.path.join(
        dir, f"{meta_path.split('/')[2].rsplit('.', 1)[0]}_{feature_type}.csv"
    )
    result_df.to_csv(output_path, index=False)
    logger.info(
        f"Feature extraction {feature_type} of {meta_path} completed. Saved to {output_path}"
    )


if __name__ == "__main__":

    extractors = [
        extract_chroma_stft,
        extract_melspectrogram,
        extract_mfcc,
        extract_rmse,
        extract_zcr,
    ]

    for extractor in extractors:
        feature_name = extractor.__name__.replace("extract_", "")
        extract_feature(extractor, "data/meta_csvs/am_ASED.csv", feature_name)
