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
        path="/kaggle/input/amharic-speech-emotional-dataset-ased/",
    )
    BANSPEMO = Dataset(
        name="BANSpEmo",
        language=LANGUAGE.BENGALI.value,
        path="/kaggle/input/bangla-lang-ser-dataset/BANSpEmo Dataset/",
    )
    CAFE = Dataset(
        name="CaFE", language=LANGUAGE.FRENCH.value, path="/kaggle/input/cafe-dataset/"
    )
    EMODB = Dataset(
        name="EmoDB",
        language=LANGUAGE.GERMAN.value,
        path="/kaggle/input/berlin-database-of-emotional-speech-emodb/wav/",
    )
    EMOTA = Dataset(
        name="EmoTa", language=LANGUAGE.TAMIL.value, path="/kaggle/input/tamserdb/"
    )
    EMOVO = Dataset(
        name="EMOVO",
        language=LANGUAGE.ITALIAN.value,
        path="/content/drive/MyDrive/FYP/Dataset/Datasets/EMOVO/EMOVO",
    )
    ESD_CHINESE = Dataset(
        name="ESD",
        language=LANGUAGE.CHINESE.value,
        path="/content/drive/MyDrive/Dataset/Datasets/ESD/Emotion Speech Dataset/",
    )
    HINDI_DATASET = Dataset(
        name="Hindi-Dataset",
        language=LANGUAGE.HINDI.value,
        path="/kaggle/input/speech-emotion-recognition-hindi",
    )
    KANNADA_DATASET = Dataset(
        name="Kannada-Dataset",
        language=LANGUAGE.KANNADA.value,
        path="/kaggle/input/kannada-emo-speech-dataset/",
    )
    MESD = Dataset(
        name="MESD",
        language=LANGUAGE.SPANISH.value,
        path="/kaggle/input/mexican-emotional-speech-databasemesd/" +
             "cy34mh68j9-5/Mexican Emotional Speech Database (MESD)/",
    )
    RAVDESS = Dataset(
        name="RAVDESS",
        language=LANGUAGE.ENGLISH.value,
        path="/content/drive/MyDrive/Dataset/Datasets/RAVDESS/archive (7)/audio_speech_actors_01-24/",
    )
    SUBESCO = Dataset(
        name="SUBESCO",
        language=LANGUAGE.BENGALI.value,
        path="/kaggle/input/subescobangla-speech-emotion-dataset/SUBESCO/",
    )
    TELUGU_DATASET = Dataset(
        name="Telugu-Dataset",
        language=LANGUAGE.TELUGU.value,
        path="/kaggle/input/telugu-emotion-speech/telugu/",
    )
    URDU_DATASET = Dataset(
        name="Urdu-Dataset",
        language=LANGUAGE.URDU.value,
        path="/kaggle/input/urdu-emotion-dataset/",
    )
