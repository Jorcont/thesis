''' Default pytorch Dataset class for Supervised Learning Task Sampling with audio data.'''

from typing_extensions import override
import torch
import torchaudio
from torch_audiomentations import Gain, AddColoredNoise, Compose

from datasets import load_dataset
import librosa

from data.dataset_classes.ImageDataset import ImageDataset

class AudioDataset(ImageDataset):
    @override
    def preprocess(self, X):
        ''' Preprocesses X to be a normalized, properly scaled input tensor.
        This version calls the full pipeline of audio preprocessing below.
        '''
        return self.process_audio_batch(X)

    def process_audio_batch(self, audio_batch, target_sr=32000, apply_augmentations=True):
        """
        Complete end-to-end audio processing pipeline.
        
        This function implements all processing steps for training an audio model:
        1. Resample audio
        2. Apply waveform augmentations (optional, typically for training)
        3. Create spectrograms via STFT
        4. Convert to Mel-scale
        5. Convert to dB scale
        6. Apply sample-wise z-standardization
        
        Args:
            audio_batch: List of audio dictionaries with 'array' and 'sampling_rate'
            target_sr: Target sample rate (default: 32000 Hz)
            apply_augmentations: Whether to apply waveform augmentations (default: True)
            
        Returns:
            Processed spectrogram batch ready for model input
        """
        # ---- STEP 1: RESAMPLE ----
        resampled_audios = []
        for audio_data in audio_batch:
            audio_array = audio_data["array"]
            original_sr = audio_data["sampling_rate"]

            # Resample to target rate
            resampled = self.resample_audio(audio_array, original_sr, target_sr)
            resampled_audios.append(resampled)

        del audio_batch
        #print(f"[DONE] Step 1: Resampled audio to {target_sr} Hz")

        # ---- STEP 2: APPLY WAVEFORM AUGMENTATIONS ----
        # (Only during training, not validation/test)
        if apply_augmentations:
            # Stack resampled audios and add channel dimension for torch_audiomentations
            waveform_batch = []
            for audio in resampled_audios:
                # Ensure it's a tensor
                if not isinstance(audio, torch.Tensor):
                    audio = torch.from_numpy(audio).float()
                waveform_batch.append(audio)
            
            waveform_batch = torch.stack(waveform_batch)  # (batch, time)
            waveform_batch = waveform_batch.unsqueeze(1)  # (batch, 1, time)
            
            waveform_batch = self.apply_waveform_augmentations(waveform_batch, sample_rate=target_sr)

            # Remove channel dimension for STFT
            waveform_batch = waveform_batch.squeeze(1)  # (batch, time)
            #print(f"[DONE] Step 2: Applied waveform augmentations - shape: {waveform_batch.shape}")
        else:
            # Stack resampled audios into batch and pad/truncate to same length
            max_length = max(len(audio) for audio in resampled_audios)
            padded_audios = []
            for audio in resampled_audios:
                if len(audio) < max_length:
                    audio = torch.nn.functional.pad(audio, (0, max_length - len(audio)))
                else:
                    audio = audio[:max_length]
                padded_audios.append(audio)

            waveform_batch = torch.stack(padded_audios)  # (batch, time)
            del resampled_audios
            #print(f"[DONE] Step 2: Skipped augmentations (eval mode) - shape: {waveform_batch.shape}")

        # ---- STEP 3: CREATE SPECTROGRAMS (STFT) ----
        spec_batch = self.create_spectrogram(
            waveform_batch,
            sample_rate=target_sr,
            n_fft=2048,
            hop_length=256,
            power=2.0
        )
        del waveform_batch
        #print(f"[DONE] Step 3: Created STFT spectrograms - shape: {spec_batch.shape}")

        # ---- STEP 4: MEL-SCALE CONVERSION ----
        mel_batch = self.convert_to_mel_scale(
            spec_batch,
            sample_rate=target_sr,
            n_mels=256,
            n_fft=2048
        )
        del spec_batch
        #print(f"[DONE] Step 4: Converted to Mel-scale - shape: {mel_batch.shape}")

        # ---- STEP 5: dB SCALE CONVERSION ----
        db_batch = self.convert_to_db_scale(mel_batch, top_db=80.0)
        del mel_batch
        #print(f"[DONE] Step 5: Converted to dB scale - shape: {db_batch.shape}")

        # ---- STEP 6: SAMPLE-WISE Z-STANDARDIZATION ----
        # Add channel dimension if needed
        if db_batch.dim() == 3:
            db_batch = db_batch.unsqueeze(1)  # (batch, 1, freq, time)

        normalized_batch = self.normalize_spectrogram_sample_wise(db_batch)
        #print(f"[DONE] Step 6: Applied sample-wise normalization - shape: {normalized_batch.shape}")

        return normalized_batch

    def resample_audio(self, audio_array, original_sr, target_sr=32000):
        """
        Resample audio from its original sample rate to a target sample rate.
        
        Resampling is crucial for standardizing all audio inputs to the same sample rate,
        which is a requirement for batch processing in neural networks.
        
        Default parameters used in GermanyAudioProtoP-Net:
            - target_sr: 32000 Hz (standard for environmental audio)
        
        Args:
            audio_array: Audio waveform as numpy array or tensor
            original_sr: Original sample rate of the audio
            target_sr: Target sample rate (default: 32000 Hz)
            
        Returns:
            Resampled audio array at target sample rate
        """
        # If already at target sample rate, no resampling needed
        if original_sr == target_sr:
            return audio_array
        
        # Convert tensor to numpy if needed
        if isinstance(audio_array, torch.Tensor):
            audio_np = audio_array.numpy()
        else:
            audio_np = audio_array
        
        # Use librosa's high-quality resampling
        resampled = librosa.resample(
            audio_np,
            orig_sr=original_sr,
            target_sr=target_sr,
            res_type='kaiser_best'  # High quality resampling
        )
        
        return torch.from_numpy(resampled).float()

    def create_spectrogram(self, waveform, sample_rate=32000, n_fft=2048, hop_length=256, power=2.0):
        """
        Convert audio waveform to spectrogram using Short-Time Fourier Transform (STFT).
        
        The spectrogram represents the frequency content of the audio over time,
        which is the input for vision-based neural networks (CNNs).
        
        Default parameters used in GermanyAudioProtoP-Net:
            - n_fft: 2048 (window size for FFT, defines frequency resolution)
            - hop_length: 256 (stride between frames, defines time resolution)
            - power: 2.0 (square the magnitude to get power spectrogram)
        
        Args:
            waveform: Audio waveform tensor of shape (time,) or (batch, time)
            sample_rate: Sample rate of the audio
            n_fft: FFT window size (higher = better frequency resolution, slower computation)
            hop_length: Number of samples between successive frames
            power: Power of the magnitude (2.0 = power spectrogram, 1.0 = magnitude spectrogram)
            
        Returns:
            Spectrogram tensor of shape (batch, freq_bins, time_frames)
        """
        # Add batch dimension if needed
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # (batch=1, time)
        
        # Create spectrogram transform
        spectrogram_transform = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=power,
            normalized=False
        )
        
        # Apply STFT
        # Output shape: (batch, n_fft//2 + 1, num_frames)
        #            = (batch, 513, num_frames)
        spectrogram = spectrogram_transform(waveform)
        
        return spectrogram

    def convert_to_mel_scale(self, spectrogram, sample_rate=32000, n_mels=256, n_fft=2048):
        """
        Convert linear frequency spectrogram to Mel-scale spectrogram.
        
        Mel-scale mimics how human ears perceive frequency (logarithmic, not linear).
        This typically improves model performance by reducing information in less
        perceptually important frequency regions.
        
        Default parameters used in GermanyAudioProtoP-Net:
            - n_mels: 256 (number of Mel frequency bands)
            - sample_rate: 32000 Hz
            - n_fft: 2048 (must match STFT n_fft)
        
        Args:
            spectrogram: Linear frequency spectrogram of shape (batch, freq_bins, time_frames)
            sample_rate: Sample rate of the audio
            n_mels: Number of Mel-frequency bands to generate
            n_fft: FFT window size (must match your STFT)
            
        Returns:
            Mel-scale spectrogram of shape (batch, n_mels, time_frames)
        """
        # Create Mel-scale transform
        # n_stft must be n_fft // 2 + 1 (for 2048, this is 1025)
        mel_transform = torchaudio.transforms.MelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            n_stft=1025,  # 2048 // 2 + 1 = 1025
        )
        
        # Convert to Mel-scale
        mel_spectrogram = mel_transform(spectrogram)
        
        return mel_spectrogram

    def convert_to_db_scale(self, spectrogram, top_db=80.0):
        """
        Convert power spectrogram to dB (decibel) scale using log compression.
        
        The dB scale compresses the dynamic range of the spectrogram, making it
        easier for the model to learn and reducing the impact of very loud vs quiet sounds.
        
        Formula: dB = 10 * log10(S) where S is the power spectrogram
        
        Args:
            spectrogram: Power spectrogram tensor of shape (batch, freq_bins, time_frames)
            top_db: Range for dB scaling (values below -top_db are clipped)
            
        Returns:
            dB-scaled spectrogram of shape (batch, freq_bins, time_frames)
        """
        # Add small epsilon to avoid log(0)
        epsilon = 1e-9
        
        # Convert power to dB
        db_spec = 10 * torch.log10(spectrogram + epsilon)
        
        # Normalize: subtract the max and clip to [-top_db, 0]
        db_spec = torch.clamp(db_spec, min=-top_db)
        db_spec = db_spec + top_db  # Shift so range is [0, top_db]
        
        return db_spec

    def apply_waveform_augmentations(self, waveform_batch, sample_rate=32000):
        """
        Apply waveform augmentations using torch_audiomentations.
        
        Default augmentations used in GermanyAudioProtoP-Net (excluding background noise
        and no-call mixer augmentations):
            1. Gain: Randomly change the volume (amplitude) of the audio
            2. AddColoredNoise: Add noise with specified frequency content
        
        Args:
            waveform_batch: Audio waveform tensor of shape (batch, 1, time)
            sample_rate: Sample rate of the audio
            
        Returns:
            Augmented waveform batch of the same shape
        """
        # Define augmentations with default parameters from the codebase
        augmentations = Compose(
            transforms=[
                # Gain augmentation: change volume randomly
                Gain(
                    min_gain_in_db=-18,      # Minimum volume decrease
                    max_gain_in_db=6,        # Maximum volume increase
                    p=0.2                    # 20% probability of application
                ),
                # AddColoredNoise augmentation: add frequency-colored noise
                AddColoredNoise(
                    min_f_decay=-2,          # Minimum decay factor
                    max_f_decay=2,           # Maximum decay factor
                    min_snr_in_db=3,         # Minimum signal-to-noise ratio
                    max_snr_in_db=30,        # Maximum signal-to-noise ratio
                    p=0.2                    # 20% probability of application
                ),
            ],
            p=1.0,                           # Always apply the composition
        )
        
        # Apply augmentations - returns ObjectDict with .samples attribute
        augmented = augmentations(waveform_batch, sample_rate=sample_rate)
        
        # Extract tensor from ObjectDict
        if hasattr(augmented, 'samples'):
            return augmented.samples
        else:
            return augmented

    def normalize_spectrogram_sample_wise(self, spectrogram_batch):
        """
        Apply sample-wise z-standardization to spectrograms.
        
        Unlike global normalization (which uses dataset-wide mean/std), sample-wise
        normalization computes mean and standard deviation for each sample independently.
        This makes the model more robust to variations in recording conditions.
        
        Formula: normalized = (spectrogram - sample_mean) / sample_std
        
        Args:
            spectrogram_batch: Spectrogram batch of shape (batch, channels, freq_bins, time_frames)
            
        Returns:
            Sample-wise normalized batch of the same shape
        """
        normalized_batch = spectrogram_batch.clone()
        
        # Process each sample in the batch
        for i in range(spectrogram_batch.shape[0]):
            sample = spectrogram_batch[i]  # (channels, freq_bins, time_frames)
            
            # Compute mean and std for this sample (across freq and time dimensions)
            sample_mean = sample.mean()
            sample_std = sample.std()
            
            # Avoid division by zero
            if sample_std < 1e-8:
                sample_std = 1e-8
            
            # Apply z-standardization
            normalized_batch[i] = (sample - sample_mean) / sample_std
        
        return normalized_batch
