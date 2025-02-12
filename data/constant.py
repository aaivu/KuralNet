from enum import Enum


class EMOTION(Enum):
    FEAR = "FEAR"
    SADNESS = "SADNESS"
    HAPPINESS = "HAPPINESS"
    ANGER = "ANGER"
    NEUTRAL = "NEUTRAL"
    DISGUST = "DISGUST"
    SURPRISE = "SURPRISE"
    CALM = "CALM"
    BOREDOM = "BOREDOM"
    SARCASTIC = "SARCASTIC"
    JOY = "JOY"


SELECTED_EMOTIONS = [
    EMOTION.ANGER.value,
    EMOTION.FEAR.value,
    EMOTION.HAPPINESS.value,
    EMOTION.SADNESS.value,
    EMOTION.NEUTRAL.value,
]


class LANGUAGE(Enum):
    ENGLISH = "en"
    GERMAN = "de"
    CHINESE = "zh"
    SPANISH = "es"
    ITALIAN = "it"
    URDU = "ur"
    TAMIL = "ta"
    TELUGU = "te"
    MALAYALAM = "ml"
    KANNADA = "kn"
    BENGALI = "bn"
    HINDI = "hi"
    FRENCH = "fr"
    AMHARIC = "am"


class Dataset:
    name: str
    language: str
    path: str

    def __init__(self, name, language, path):
        self.name = name
        self.language = language
        self.path = path


class DATASET(Enum):
    ASED = Dataset(
        name="ASED",
        language=LANGUAGE.AMHARIC.value,
        path="thanikansivatheepan/amharic-speech-emotional-dataset-ased",
    )
    BANSPEMO = Dataset(
        name="BANSpEmo",
        language=LANGUAGE.BENGALI.value,
        path="thanikansivatheepan/bangla-lang-ser-dataset/BANSpEmo Dataset/",
    )
    CAFE = Dataset(
        name="CaFE",
        language=LANGUAGE.FRENCH.value,
        path="jubeerathan/cafe-dataset",
    )
    EMODB = Dataset(
        name="EmoDB",
        language=LANGUAGE.GERMAN.value,
        path="piyushagni5/berlin-database-of-emotional-speech-emodb/wav/",
    )
    EMOTA = Dataset(
        name="EmoTa", language=LANGUAGE.TAMIL.value, path="luxluxshan/tamserdb"
    )
    EMOVO = Dataset(
        name="EMOVO",
        language=LANGUAGE.ITALIAN.value,
        path="sourabhy/emovo-italian-ser-dataset",
    )
    ESD_CHINESE = Dataset(
        name="ESD",
        language=LANGUAGE.CHINESE.value,
        path="thanikansivatheepan/esd-dataset-fyp",
    )
    HINDI_DATASET = Dataset(
        name="Hindi-Dataset",
        language=LANGUAGE.HINDI.value,
        path="vishlb/speech-emotion-recognition-hindi/data",
    )
    KANNADA_DATASET = Dataset(
        name="Kannada-Dataset",
        language=LANGUAGE.KANNADA.value,
        path="thanikansivatheepan/kannada-emo-speech-dataset",
    )
    MESD = Dataset(
        name="MESD",
        language=LANGUAGE.SPANISH.value,
        path="ashfaqsyed/mexican-emotional-speech-databasemesd"
        + "cy34mh68j9-5/Mexican Emotional Speech Database (MESD)/",
    )
    RAVDESS = Dataset(
        name="RAVDESS",
        language=LANGUAGE.ENGLISH.value,
        path="uwrfkaggler/ravdess-emotional-speech-audio",
    )
    SUBESCO = Dataset(
        name="SUBESCO",
        language=LANGUAGE.BENGALI.value,
        path="sushmit0109/subescobangla-speech-emotion-dataset/SUBESCO/",
    )
    TELUGU_DATASET = Dataset(
        name="Telugu-Dataset",
        language=LANGUAGE.TELUGU.value,
        path="jettysowmith/telugu-emotion-speech/telugu/",
    )
    URDU_DATASET = Dataset(
        name="Urdu-Dataset",
        language=LANGUAGE.URDU.value,
        path="kingabzpro/urdu-emotion-dataset",
    )
