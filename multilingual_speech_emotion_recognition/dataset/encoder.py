from data.constant import EMOTION


def emotion_encoder(emotion: str):
    EMOTION_MAPPING = {
        EMOTION.ANGER.value: 0,
        EMOTION.FEAR.value: 1,
        EMOTION.HAPPINESS.value: 2,
        EMOTION.SADNESS.value: 3,
        EMOTION.NEUTRAL.value: 4,
    }

    if emotion not in EMOTION_MAPPING:
        raise ValueError(
            f"Invalid emotion: {emotion}. Allowed values are {list(EMOTION_MAPPING.keys())}"
        )

    return EMOTION_MAPPING[emotion]


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
