from kuralnet.utils.constant import EMOTION  # Assuming EMOTION is an Enum
from typing import Dict


def emotion_encoder(emotion: str, emotion_map: Dict[str, int] = None) -> int:
    """
    Encodes an emotion string to an integer using a provided or default mapping.

    Args:
        emotion (str): The emotion string to encode.
        emotion_map (Dict[str, int], optional): A dictionary mapping emotion strings to integers.
            If None, a default mapping is used. Defaults to None.

    Returns:
        int: The integer representation of the emotion.

    Raises:
        KeyError: If the emotion is not found in the mapping.
    """
    if emotion_map is None:
        emotion_map = {
            EMOTION.HAPPINESS.value: 0,
            EMOTION.SADNESS.value: 1,
            EMOTION.FEAR.value: 2,
            EMOTION.ANGER.value: 3,
            EMOTION.NEUTRAL.value: 4,
        }
    return emotion_map[emotion]


def emotion_decoder(
    encoded_emotion: int, reverse_emotion_map: Dict[int, str] = None
) -> str:
    """
    Decodes an integer representation of an emotion back to its string value using a
    provided or default reverse mapping.

    Args:
        encoded_emotion (int): The integer to decode.
        reverse_emotion_map (Dict[int, str], optional): A dictionary mapping integers to emotion strings.
            If None, a default reverse mapping is used. Defaults to None.

    Returns:
        str: The string representation of the emotion.

    Raises:
        KeyError: If the encoded emotion is not found in the mapping.
    """
    if reverse_emotion_map is None:
        reverse_emotion_map = {
            0: EMOTION.HAPPINESS.value,
            1: EMOTION.SADNESS.value,
            2: EMOTION.FEAR.value,
            3: EMOTION.ANGER.value,
            4: EMOTION.NEUTRAL.value,
        }
    return reverse_emotion_map[encoded_emotion]
