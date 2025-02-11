import librosa 
import numpy as np


def pitch_shift(data, sampling_rate, pitch_factor=0.2):
    """
    Shift the pitch of an audio signal.

    Args:
        data (np.ndarray): Audio signal.
        sampling_rate (int): Sampling rate of the audio signal.
        pitch_shift_amount (int): Amount of pitch shift.

    Returns:
        np.ndarray: Pitch-shifted audio signal.
    """
    return librosa.effects.pitch_shift(data, n_steps=pitch_factor, sr=sampling_rate)

def speed_tuning(data, speed_factor):
    """
    Change the speed of an audio signal.

    Args:
        data (np.ndarray): Audio signal.
        speed_factor (float): Speed factor.

    Returns:
        np.ndarray: Speed-tuned audio signal.
    """
    return librosa.effects.time_stretch(data, speed_factor)

def add_noise(data, noise_factor=0.003):
    """
    Add noise to an audio signal.

    Args:
        data (np.ndarray): Audio signal.
        noise_factor (float): Noise factor.

    Returns:
        np.ndarray: Noisy audio signal.
    """
    noise = np.random.randn(len(data))
    noise_data = data + noise_factor * noise
    return noise_data


def add_pink_noise(data, noise_factor=0.003):
    """
    Add pink noise to an audio signal.

    Args:
        data (np.ndarray): Audio signal.
        noise_factor (float): Noise factor.

    Returns:
        np.ndarray: Noisy audio signal.
    """

    def pink_noise(length):
        """	Generate pink noise.
        Args:
            length (int): Length of the noise signal.
        Returns:
            np.ndarray: Pink noise signal.
        """
        uneven = length % 2
        X = np.random.randn(length // 2 + 1 + uneven) + 1j * np.random.randn(length // 2 + 1 + uneven)
        S = np.sqrt(np.arange(len(X)) + 1.)
        y = (np.fft.irfft(X / S)).real
        if uneven:
            y = y[:-1]
        return y
    
    noise = pink_noise(len(data))
    noise_data = data + noise_factor * noise
    return noise_data

def add_silence(data, sample_rate, silence_duration=0.3):
        """	Add silence to an audio signal.
        Args:
            data (np.ndarray): Audio signal.
            sample_rate (int): Sampling rate.
            silence_duration (float): Duration of the silence.
        Returns:
            np.ndarray: Audio signal with added silence.
        """

        silence = np.zeros(int(silence_duration * sample_rate))
        start = np.random.randint(0, len(data))
        return np.concatenate((data[:start], silence, data[start:]))

def combine_pitch_speed(data, sampling_rate, speed_factor, pitch_factor=0.2):
    """
    Shift the pitch and change the speed of an audio signal.

    Args:
        data (np.ndarray): Audio signal.
        sampling_rate (int): Sampling rate of the audio signal.
        pitch_factor (int): Amount of pitch shift.
        speed_factor (float): Speed factor.

    Returns:
        np.ndarray: Pitch-shifted and speed-tuned audio signal.
    """
    pitch_shifted = pitch_shift(data, sampling_rate, pitch_factor)
    return speed_tuning(pitch_shifted, speed_factor)

def combine_pitch_silence(data, sampling_rate, pitch_factor, silence_duration):
    """
    Shift the pitch and add silence to an audio signal.

    Args:
        data (np.ndarray): Audio signal.
        sampling_rate (int): Sampling rate of the audio signal.
        pitch_factor (int): Amount of pitch shift.
        silence_duration (float): Duration of the silence.

    Returns:
        np.ndarray: Pitch-shifted and silence-added audio signal.
    """
    pitch_shifted = pitch_shift(data, sampling_rate, pitch_factor)
    return add_silence(pitch_shifted, sampling_rate, silence_duration)

def combine_speed_silence(data, sampling_rate, speed_factor, silence_duration):
    """
    Change the speed and add silence to an audio signal.

    Args:
        data (np.ndarray): Audio signal.
        sampling_rate (int): Sampling rate of the audio signal.
        speed_factor (float): Speed factor.
        silence_duration (float): Duration of the silence.

    Returns:
        np.ndarray: Speed-tuned and silence-added audio signal.
    """
    speed_tuned = speed_tuning(data, speed_factor)
    return add_silence(speed_tuned, sampling_rate, silence_duration)

def augment_data(data, sampling_rate, aug:list, pitch_factor=0.2, slow_speed_factor=0.95, fast_speed_factor=1.05, noise_factor=0.003, silence_duration=0.3):
    """
    Augment an audio signal.

    Args:
        data (np.ndarray): Audio signal.
        sampling_rate (int): Sampling rate of the audio signal.
        aug (list): List of augmentation techniques which contains one or more of the following: "pitch_shift", "speed_tuning_fast", "speed_tuning_slow", "add_noise", "add_pink_noise", "add_silence", "combine_pitch_speed", "combine_pitch_silence", "combine_speed_silence".
        pitch_factor (float): Amount of pitch shift.
        speed_factor (float): Speed factor.
        noise_factor (float): Noise factor.
        silence_duration (float): Duration of the silence.

    Returns:
        list: Augmented audio signals.
    """
    augmented_data = []
    
    for augmentation in aug:
        if augmentation == "pitch_shift":
            augmented_data.append(pitch_shift(data, sampling_rate, pitch_factor))
        elif augmentation == "speed_tuning_fast":
            augmented_data.append(speed_tuning(data, slow_speed_factor))
        elif augmentation == "speed_tuning_slow":
            augmented_data.append(speed_tuning(data, fast_speed_factor))
        elif augmentation == "add_noise":
            augmented_data.append(add_noise(data, noise_factor))
        elif augmentation == "add_pink_noise":
            augmented_data.append(add_pink_noise(data, noise_factor))
        elif augmentation == "add_silence":
            augmented_data.append(add_silence(data, sampling_rate, silence_duration))
        elif augmentation == "combine_pitch_speed":
            augmented_data.append(combine_pitch_speed(data, sampling_rate, slow_speed_factor, pitch_factor ))
            augmented_data.append(combine_pitch_speed(data, sampling_rate, fast_speed_factor, pitch_factor ))
        elif augmentation == "combine_pitch_silence":
            augmented_data.append(combine_pitch_silence(data, sampling_rate, pitch_factor, silence_duration))
        elif augmentation == "combine_speed_silence":
            augmented_data.append(combine_speed_silence(data, sampling_rate, slow_speed_factor, silence_duration))
            augmented_data.append(combine_speed_silence(data, sampling_rate, fast_speed_factor, silence_duration))
        else:
            raise ValueError(f"Unknown augmentation technique: {augmentation}")
    
    return augmented_data
