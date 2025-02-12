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

SELECTED_EMOTIONS = [EMOTION.ANGER.value, EMOTION.FEAR.value, EMOTION.HAPPINESS.value, EMOTION.SADNESS.value, EMOTION.NEUTRAL.value]


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
    path:str
    
    def __init__(self,name,language,path):
        self.name = name
        self.language = language 
        self.path = path


class DATASET(Enum):
   EMOTA = Dataset(name="EmoTa",language=LANGUAGE.TAMIL.value,path="/kaggle/input/tamserdb/")