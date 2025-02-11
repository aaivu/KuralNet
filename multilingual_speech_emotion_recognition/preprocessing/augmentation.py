import os
import librosa 
import numpy as np
import soundfile as sf
from typing import List, Union
from pathlib import Path
from data_preprocessing import load_audio

def validate_parameters(pitch_factor: float, speed_factor: float, noise_factor: float, silence_duration: float):
    """Validate augmentation parameters."""
    if not -2 <= pitch_factor <= 2:
        raise ValueError("Pitch factor should be between -2 and 2 semitones")
    if not 0.1 <= speed_factor <= 0.5:
        raise ValueError("Speed factor should be between 0.1 and 0.5")
    if not 0 <= noise_factor <= 0.1:
        raise ValueError("Noise factor should be between 0 and 0.1")
    if not 0.1 <= silence_duration <= 0.3:
        raise ValueError("Silence duration should be between 0.1 and 0.3 seconds")

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to prevent clipping."""
    return audio / (np.max(np.abs(audio)) + 1e-8)

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
        """ Generate pink noise.
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
    """ Add silence to an audio signal.
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

def augment_data(
    audio_path: Union[str, Path],
    output_folder: Union[str, Path],
    aug: List[str],
    pitch_factor: float = 0.2,
    slow_speed_factor: float = 0.95,
    fast_speed_factor: float = 1.05,
    noise_factor: float = 0.003,
    silence_duration: float = 0.3
) -> List[str]:
    """
    Augment an audio signal and save the augmented versions as audio files.

    Args:
        audio_path (str): Path to the input audio file
        output_folder (str): Path to save augmented audio files
        aug (list): List of augmentation techniques which contains one or more of the following: 
                   "pitch_shift", "speed_tuning_fast", "speed_tuning_slow", "add_noise", 
                   "add_pink_noise", "add_silence", "combine_pitch_speed", 
                   "combine_pitch_silence", "combine_speed_silence"
        pitch_factor (float): Amount of pitch shift
        slow_speed_factor (float): Speed factor for slowing down
        fast_speed_factor (float): Speed factor for speeding up
        noise_factor (float): Noise factor
        silence_duration (float): Duration of the silence

    Returns:
        list: Paths to the generated augmented audio files
    """
    try:
        # Convert paths to Path objects for better handling
        audio_path = Path(audio_path)
        output_folder = Path(output_folder)
        
        # Validate input parameters
        validate_parameters(pitch_factor, slow_speed_factor, noise_factor, silence_duration)
        validate_parameters(pitch_factor, fast_speed_factor, noise_factor, silence_duration)
        
        # Validate input file
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Create output directory
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Load audio with error handling
        try:
            data, sampling_rate = librosa.load(str(audio_path), sr=None)
        except Exception as e:
            raise RuntimeError(f"Error loading audio file: {e}")
        
        augmented_files = []
        file_name = audio_path.stem
        
        for augmentation in aug:
            try:
                if augmentation == "pitch_shift":
                    aug_data = pitch_shift(data, sampling_rate, pitch_factor)
                    out_path = output_folder / f"{file_name}_pitch_shift.wav"
                    
                elif augmentation == "speed_tuning_fast":
                    aug_data = speed_tuning(data, fast_speed_factor)
                    out_path = output_folder / f"{file_name}_speed_fast.wav"
                    
                elif augmentation == "speed_tuning_slow":
                    aug_data = speed_tuning(data, slow_speed_factor)
                    out_path = output_folder / f"{file_name}_speed_slow.wav"
                    
                elif augmentation == "add_noise":
                    aug_data = add_noise(data, noise_factor)
                    out_path = output_folder / f"{file_name}_noise.wav"
                    
                elif augmentation == "add_pink_noise":
                    aug_data = add_pink_noise(data, noise_factor)
                    out_path = output_folder / f"{file_name}_pink_noise.wav"
                    
                elif augmentation == "add_silence":
                    aug_data = add_silence(data, sampling_rate, silence_duration)
                    out_path = output_folder / f"{file_name}_silence.wav"
                    
                elif augmentation == "combine_pitch_speed":
                    # Fast version
                    aug_data = combine_pitch_speed(data, sampling_rate, fast_speed_factor, pitch_factor)
                    out_path = output_folder / f"{file_name}_pitch_speed_fast.wav"
                    
                    # Slow version
                    aug_data = combine_pitch_speed(data, sampling_rate, slow_speed_factor, pitch_factor)
                    out_path = output_folder / f"{file_name}_pitch_speed_slow.wav"
                    
                elif augmentation == "combine_pitch_silence":
                    aug_data = combine_pitch_silence(data, sampling_rate, pitch_factor, silence_duration)
                    out_path = output_folder / f"{file_name}_pitch_silence.wav"
                    
                elif augmentation == "combine_speed_silence":
                    # Fast version
                    aug_data = combine_speed_silence(data, sampling_rate, fast_speed_factor, silence_duration)
                    out_path = output_folder / f"{file_name}_speed_silence_fast.wav"
                    
                    # Slow version
                    aug_data = combine_speed_silence(data, sampling_rate, slow_speed_factor, silence_duration)
                    out_path = output_folder / f"{file_name}_speed_silence_slow.wav"
                    
                else:
                    raise ValueError(f"Unknown augmentation technique: {augmentation}")
                
                # Normalize and save the augmented audio
                aug_data = normalize_audio(aug_data)
                sf.write(str(out_path), aug_data, sampling_rate)
                augmented_files.append(str(out_path))
                
            except Exception as e:
                print(f"Warning: Failed to apply {augmentation} for audio in {audio_path}: {e}")
                continue
        
        if not augmented_files:
            raise RuntimeError("No augmentations were successfully applied")
        
        return augmented_files
    
    except Exception as e:
        raise RuntimeError(f"Augmentation failed: {e}")
