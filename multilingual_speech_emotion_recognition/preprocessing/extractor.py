import os

import pandas as pd

from multilingual_speech_emotion_recognition.preprocessing.feature_extractor.handcrafted_feature import (
    extract_chroma_stft, extract_melspectrogram, extract_mfcc, extract_rmse,
    extract_zcr)
from multilingual_speech_emotion_recognition.preprocessing.feature_extractor.hubert import \
    HuBERTFeatureExtractor
from multilingual_speech_emotion_recognition.preprocessing.feature_extractor.wav2vec import \
    Wav2Vec2FeatureExtractor
from multilingual_speech_emotion_recognition.preprocessing.feature_extractor.wavlm import \
    WavLMModelFeatureExtractor
from multilingual_speech_emotion_recognition.preprocessing.feature_extractor.whispher import \
    WhisperFeatureExtractor
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


def extract_pretrained_feature(
    extractor_class, meta_path, feature_type, sr=16000
):
    logger.info(f"Feature extraction {feature_type} of {meta_path} started.")

    meta_csv = load_csv(meta_path)
    result_data = []

    extractor = extractor_class()

    for _, row in meta_csv.iterrows():
        emotion = row["emotion"]
        audio_path = row["audio_path"]

        try:
            audio, _ = load_audio(audio_path, sr)
            feat = extractor.extract_features(audio)  # Call method on instance
            result_data.append([emotion] + feat.tolist())
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")

    if not result_data:
        logger.warning(f"No features extracted for {meta_path}")
        return

    feature_names = [f"feature_{i}" for i in range(len(result_data[0]) - 1)]
    columns = ["emotion"] + feature_names

    dir = os.path.join(meta_path.split("/")[0], "features")
    os.makedirs(dir, exist_ok=True)

    output_path = os.path.join(
        dir, f"{meta_path.split('/')[2].rsplit('.', 1)[0]}_{feature_type}.csv"
    )
    pd.DataFrame(result_data, columns=columns).to_csv(output_path, index=False)

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

    pre_trained_extractors = [
        Wav2Vec2FeatureExtractor,
        WavLMModelFeatureExtractor,
        HuBERTFeatureExtractor,
        WhisperFeatureExtractor,
    ]

    csv_files = [f for f in os.listdir("data/meta_csvs") if f.endswith(".csv")]

    for csv_file in csv_files:
        csv_path = os.path.join("data/meta_csvs", csv_file)

        for extractor in extractors:
            feature_name = extractor.__name__.replace("extract_", "")
            extract_feature(extractor, csv_path, feature_name)

        for extractor_class in pre_trained_extractors:
            class_name = extractor_class.__name__
            extract_pretrained_feature(extractor_class, csv_path, class_name)
