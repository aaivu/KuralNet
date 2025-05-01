def emotion_converter(value, EMOTION_MAPPING, mode="encode"):

    if mode == "encode":
        if value not in EMOTION_MAPPING:
            raise ValueError(
                f"Invalid emotion: {value}. Allowed values are {list(EMOTION_MAPPING.keys())}"
            )
        return EMOTION_MAPPING[value], EMOTION_MAPPING

    elif mode == "decode":
        if value not in EMOTION_MAPPING.values():
            raise ValueError(
                f"Invalid encoded value: {value}. Allowed values are {list(EMOTION_MAPPING.values())}"
            )
        return [k for k, v in EMOTION_MAPPING.items() if v == value][0]

    else:
        raise ValueError("Mode should be either 'encode' or 'decode'")


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
