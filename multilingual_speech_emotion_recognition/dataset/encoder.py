def emotion_encoder(emotion: str):
    EMOTION_MAPPING = {
        "neutral": 0,
        "happy": 1,
        "sad": 2,
        "angry": 3,
        "fearful": 4,
        "disgust": 5,
        "surprised": 6,
    }

    if emotion.lower() not in EMOTION_MAPPING:
        raise ValueError(
            f"Invalid emotion: {emotion}. Allowed values are {list(EMOTION_MAPPING.keys())}"
        )

    return EMOTION_MAPPING[emotion.lower()]


def gender_encoder(gender: str):
    GENDER_MAPPING = {"male": 0, "female": 1, "non-binary": 2, "other": 3}

    if gender.lower() not in GENDER_MAPPING:
        raise ValueError(
            f"Invalid gender: {gender}. Allowed values are {list(GENDER_MAPPING.keys())}"
        )

    return GENDER_MAPPING[gender.lower()]


def language_encoder(language: str):
    LANGUAGE_MAPPING = {
        "english": 0,
        "spanish": 1,
        "french": 2,
        "german": 3,
        "tamil": 4,
        "mandarin": 5,
        "hindi": 6,
    }

    if language.lower() not in LANGUAGE_MAPPING:
        raise ValueError(
            f"Invalid language: {language}. Allowed values are {list(LANGUAGE_MAPPING.keys())}"
        )

    return LANGUAGE_MAPPING[language.lower()]
