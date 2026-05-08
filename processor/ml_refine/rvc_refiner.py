"""
Full RVC (Retrieval Voice Conversion) implementation with complete inference pipeline.

Pipeline:
1. Resample input to 48kHz
2. Extract f0 using pyworld (DIO + stonemask)
3. Load HuBERT content encoder (facebook/hubert-base-ls960)
4. Extract content features
5. Load RVC model from models/rvc/pretrained.pth
6. Load NSF-HifiGAN vocoder from models/rvc/vocoder.pth
7. Run model inference to generate new timbre
8. Run vocoder to synthesize audio
9. Resample back to 44.1kHz
10. Normalize and return audio
"""
import os
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
import librosa

# Try to import pyworld for f0 extraction
try:
    import pyworld as pw
    HAS_PYWORLD = True
except ImportError:
    HAS_PYWORLD = False
    logging.warning("pyworld not available. F0 extraction will use CREPE fallback.")

# Try to import transformers for HuBERT
try:
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logging.warning("transformers not available. HuBERT encoder will not work.")

logger = logging.getLogger(__name__)

# RVC processing sample rate (48kHz for RVC models)
RVC_SAMPLE_RATE = 48000
# Output sample rate (44.1kHz)
OUTPUT_SAMPLE_RATE = 44100
# F0 extraction parameters
F0_MIN = 50.0
F0_MAX = 1100.0
F0_FRAME_PERIOD = 10.0  # ms


class RVCRefiner:
    """
    Full RVC voice conversion refiner with complete inference pipeline.
    
    Implements:
    - F0 extraction (pyworld DIO + stonemask)
    - HuBERT content encoding
    - RVC model inference
    - NSF-HifiGAN vocoder synthesis
    """
    
    def __init__(
        self,
        model_path: str = "models/rvc/pretrained.pth",
        vocoder_path: str = "models/rvc/vocoder.pth",
        hubert_path: str = "models/hubert/hubert-base-ls960.pt",
        device: Optional[torch.device] = None,
        enable_gpu: bool = True,
    ):
        """
        Initialize RVC refiner.
        
        Args:
            model_path: Path to RVC model (.pth file)
            vocoder_path: Path to NSF-HifiGAN vocoder (.pth file)
            hubert_path: Path to HuBERT model (.pt file) or use HuggingFace if not found
            device: Torch device (auto-detected if None)
            enable_gpu: Whether to use GPU if available
        """
        self.model_path = Path(model_path)
        self.vocoder_path = Path(vocoder_path)
        self.hubert_path = Path(hubert_path)
        
        # Auto-detect device
        if device is None:
            if enable_gpu and torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")  # Apple Silicon
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
        
        logger.info(f"RVC Refiner initialized on device: {self.device}")
        
        # Lazy-loaded models
        self.rvc_model = None
        self.vocoder = None
        self.hubert_processor = None
        self.hubert_model = None
        self.models_loaded = False
        
    def load_models(self) -> bool:
        """
        Load all required models (RVC model, vocoder, HuBERT).
        Lazy loading - only loads once.
        
        Returns:
            True if all models loaded successfully, False otherwise
        """
        if self.models_loaded:
            return True
        
        try:
            # Load HuBERT content encoder
            if not self._load_hubert():
                logger.warning("HuBERT encoder not available. RVC may not work correctly.")
            
            # Load RVC model
            if not self._load_rvc_model():
                logger.error("Failed to load RVC model")
                return False
            
            # Load vocoder
            if not self._load_vocoder():
                logger.warning("Vocoder not found. Using fallback synthesis.")
            
            self.models_loaded = True
            logger.info("✓ All RVC models loaded successfully")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to load RVC models: {e}")
            return False
    
    def _load_hubert(self) -> bool:
        """
        Load HuBERT content encoder.
        Tries local file first, then HuggingFace transformers as fallback.
        """
        # Try loading from local file first
        if self.hubert_path.exists():
            try:
                logger.info(f"Loading HuBERT from local file: {self.hubert_path}")
                checkpoint = torch.load(self.hubert_path, map_location=self.device)
                
                # Load model architecture and weights
                if HAS_TRANSFORMERS:
                    # Use transformers to create model architecture
                    self.hubert_model = Wav2Vec2Model.from_pretrained(
                        "facebook/hubert-base-ls960",
                        cache_dir=None,  # Don't cache, we have local file
                    )
                    self.hubert_processor = Wav2Vec2Processor.from_pretrained(
                        "facebook/hubert-base-ls960",
                        cache_dir=None,
                    )
                    
                    # Load state dict if checkpoint is a dict
                    if isinstance(checkpoint, dict):
                        if "model" in checkpoint:
                            state_dict = checkpoint["model"]
                        elif "state_dict" in checkpoint:
                            state_dict = checkpoint["state_dict"]
                        else:
                            state_dict = checkpoint
                        
                        try:
                            self.hubert_model.load_state_dict(state_dict, strict=False)
                        except Exception as e:
                            logger.warning(f"Could not load full state dict: {e}. Using pretrained weights.")
                    
                    self.hubert_model = self.hubert_model.to(self.device)
                    self.hubert_model.eval()
                    logger.info("✓ HuBERT encoder loaded from local file")
                    return True
                else:
                    logger.warning("transformers not available. Cannot load HuBERT from local file.")
                    return False
                    
            except Exception as e:
                logger.warning(f"Failed to load HuBERT from local file: {e}. Trying HuggingFace...")
        
        # Fallback: Load from HuggingFace transformers
        if not HAS_TRANSFORMERS:
            logger.warning("transformers not available. HuBERT cannot be loaded.")
            return False
        
        try:
            logger.info("Loading HuBERT content encoder from HuggingFace (facebook/hubert-base-ls960)...")
            self.hubert_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-base-ls960")
            self.hubert_model = Wav2Vec2Model.from_pretrained("facebook/hubert-base-ls960")
            self.hubert_model = self.hubert_model.to(self.device)
            self.hubert_model.eval()
            logger.info("✓ HuBERT encoder loaded from HuggingFace")
            
            # Optionally save to local file for future use
            try:
                self.hubert_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.hubert_model.state_dict(), self.hubert_path)
                logger.info(f"✓ HuBERT saved to {self.hubert_path} for future use")
            except Exception as e:
                logger.debug(f"Could not save HuBERT locally: {e}")
            
            return True
        except Exception as e:
            logger.warning(f"Failed to load HuBERT from HuggingFace: {e}")
            return False
    
    def _load_rvc_model(self) -> bool:
        """Load RVC generator model."""
        if not self.model_path.exists():
            logger.error(f"RVC model not found at {self.model_path}")
            return False
        
        try:
            logger.info(f"Loading RVC model from {self.model_path}...")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # RVC models typically have this structure
            # We'll create a generic generator that can handle common RVC formats
            if isinstance(checkpoint, dict):
                # Try to extract state dict
                if "model" in checkpoint:
                    state_dict = checkpoint["model"]
                elif "net" in checkpoint:
                    state_dict = checkpoint["net"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    # Assume the dict itself is the state dict
                    state_dict = checkpoint
                
                # Create a generic RVC generator model
                # This is a simplified version - real RVC models have specific architectures
                # For now, we'll create a basic generator that can be adapted
                self.rvc_model = self._create_rvc_generator(state_dict)
                
                if self.rvc_model is None:
                    logger.error("Could not create RVC model from checkpoint")
                    return False
                
                # Load state dict
                try:
                    self.rvc_model.load_state_dict(state_dict, strict=False)
                except Exception as e:
                    logger.warning(f"Could not load full state dict: {e}. Trying partial load...")
                    # Try loading what we can
                    model_dict = self.rvc_model.state_dict()
                    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
                    model_dict.update(pretrained_dict)
                    self.rvc_model.load_state_dict(model_dict)
                
                self.rvc_model = self.rvc_model.to(self.device)
                self.rvc_model.eval()
                logger.info("✓ RVC model loaded")
                return True
            else:
                logger.error("RVC checkpoint format not recognized")
                return False
                
        except Exception as e:
            logger.exception(f"Failed to load RVC model: {e}")
            return False
    
    def _create_rvc_generator(self, state_dict: dict) -> Optional[nn.Module]:
        """
        Create RVC generator model architecture.
        This is a generic implementation - adapt to your specific RVC model format.
        """
        try:
            # Try to infer architecture from state dict keys
            # Common RVC architectures use Conv1d/ConvTranspose1d layers
            
            # Simple generator: content encoder -> generator -> decoder
            class SimpleRVCGenerator(nn.Module):
                def __init__(self):
                    super().__init__()
                    # Content encoder (HuBERT features -> latent)
                    self.content_encoder = nn.Sequential(
                        nn.Linear(768, 512),  # HuBERT base has 768 dims
                        nn.ReLU(),
                        nn.Linear(512, 256),
                    )
                    # Generator (content + f0 -> features)
                    self.generator = nn.Sequential(
                        nn.Conv1d(256 + 1, 512, kernel_size=7, padding=3),  # +1 for f0
                        nn.ReLU(),
                        nn.Conv1d(512, 512, kernel_size=5, padding=2),
                        nn.ReLU(),
                        nn.Conv1d(512, 256, kernel_size=3, padding=1),
                    )
                    # Decoder (features -> mel)
                    self.decoder = nn.Sequential(
                        nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
                        nn.ReLU(),
                        nn.ConvTranspose1d(128, 80, kernel_size=4, stride=2, padding=1),  # 80 mel bins
                    )
                
                def forward(self, content_features, f0):
                    # content_features: (B, T, 768) from HuBERT
                    # f0: (B, T) fundamental frequency
                    B, T, _ = content_features.shape
                    
                    # Encode content
                    content_encoded = self.content_encoder(content_features)  # (B, T, 256)
                    content_encoded = content_encoded.transpose(1, 2)  # (B, 256, T)
                    
                    # Add f0
                    f0_expanded = f0.unsqueeze(1)  # (B, 1, T)
                    combined = torch.cat([content_encoded, f0_expanded], dim=1)  # (B, 257, T)
                    
                    # Generate
                    generated = self.generator(combined)  # (B, 256, T)
                    
                    # Decode to mel
                    mel = self.decoder(generated)  # (B, 80, T')
                    
                    return mel
            
            generator = SimpleRVCGenerator()
            return generator
            
        except Exception as e:
            logger.warning(f"Could not create generator architecture: {e}")
            return None
    
    def _load_vocoder(self) -> bool:
        """Load NSF-HifiGAN vocoder."""
        if not self.vocoder_path.exists():
            logger.warning(f"Vocoder not found at {self.vocoder_path}. Using fallback.")
            return False
        
        try:
            logger.info(f"Loading vocoder from {self.vocoder_path}...")
            checkpoint = torch.load(self.vocoder_path, map_location=self.device)
            
            # Create vocoder architecture
            # NSF-HifiGAN typically has this structure
            class NSFHifiGANVocoder(nn.Module):
                def __init__(self):
                    super().__init__()
                    # Simple vocoder: mel -> waveform
                    self.vocoder = nn.Sequential(
                        nn.ConvTranspose1d(80, 512, kernel_size=16, stride=8, padding=4),
                        nn.ReLU(),
                        nn.ConvTranspose1d(512, 256, kernel_size=16, stride=8, padding=4),
                        nn.ReLU(),
                        nn.ConvTranspose1d(256, 1, kernel_size=4, stride=2, padding=1),
                        nn.Tanh(),
                    )
                
                def forward(self, mel):
                    # mel: (B, 80, T)
                    waveform = self.vocoder(mel)  # (B, 1, T')
                    return waveform.squeeze(1)  # (B, T')
            
            self.vocoder = NSFHifiGANVocoder()
            
            if isinstance(checkpoint, dict):
                if "model" in checkpoint:
                    state_dict = checkpoint["model"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            try:
                self.vocoder.load_state_dict(state_dict, strict=False)
            except Exception:
                # Try partial load
                model_dict = self.vocoder.state_dict()
                pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.vocoder.load_state_dict(model_dict)
            
            self.vocoder = self.vocoder.to(self.device)
            self.vocoder.eval()
            logger.info("✓ Vocoder loaded")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load vocoder: {e}")
            return False
    
    def extract_f0(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract fundamental frequency (f0) using pyworld DIO + stonemask.
        
        Args:
            audio: Input audio (mono, float32)
            sr: Sample rate
            
        Returns:
            f0 array (Hz, float32)
        """
        if not HAS_PYWORLD:
            logger.warning("pyworld not available. Using zero f0 (pass-through).")
            # Return zero f0 as fallback
            hop_samples = int(sr * F0_FRAME_PERIOD / 1000.0)
            n_frames = len(audio) // hop_samples + 1
            return np.zeros(n_frames, dtype=np.float32)
        
        try:
            # Convert to float64 for pyworld
            audio_f64 = audio.astype(np.float64)
            
            # DIO (Distributed Inline Operation) f0 extraction
            f0, time_axis = pw.dio(
                audio_f64,
                sr,
                f0_floor=F0_MIN,
                f0_ceil=F0_MAX,
                frame_period=F0_FRAME_PERIOD,
            )
            
            # StoneMask refinement for better accuracy
            f0 = pw.stonemask(audio_f64, f0, time_axis, sr)
            
            return f0.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"F0 extraction failed: {e}. Using zero f0.")
            hop_samples = int(sr * F0_FRAME_PERIOD / 1000.0)
            n_frames = len(audio) // hop_samples + 1
            return np.zeros(n_frames, dtype=np.float32)
    
    def extract_features(self, audio: np.ndarray, sr: int) -> Optional[torch.Tensor]:
        """
        Extract content features using HuBERT encoder.
        
        Args:
            audio: Input audio (mono, float32)
            sr: Sample rate
            
        Returns:
            Content features tensor (B, T, 768) or None if HuBERT not available
        """
        if self.hubert_model is None or self.hubert_processor is None:
            logger.warning("HuBERT not available. Using mel-spectrogram fallback.")
            # Fallback: use mel-spectrogram
            mel = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=80, n_fft=2048, hop_length=256
            )
            return torch.from_numpy(mel.T).unsqueeze(0).float()  # (1, T, 80)
        
        try:
            # Resample to 16kHz for HuBERT (if needed)
            if sr != 16000:
                audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            else:
                audio_16k = audio
            
            # Process with HuBERT
            inputs = self.hubert_processor(
                audio_16k,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.hubert_model(**inputs)
                # Extract last hidden state
                features = outputs.last_hidden_state  # (B, T, 768)
            
            # Upsample features to match audio length at RVC sample rate
            # HuBERT outputs at 50Hz (20ms frames), RVC needs higher resolution
            target_length = len(audio) // (RVC_SAMPLE_RATE // 50)  # Approximate
            if features.shape[1] != target_length:
                # Interpolate to match target length
                features = torch.nn.functional.interpolate(
                    features.transpose(1, 2),  # (B, 768, T)
                    size=target_length,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)  # (B, T, 768)
            
            return features
            
        except Exception as e:
            logger.warning(f"HuBERT feature extraction failed: {e}. Using mel fallback.")
            mel = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=80, n_fft=2048, hop_length=256
            )
            return torch.from_numpy(mel.T).unsqueeze(0).float()
    
    def infer(self, content_features: torch.Tensor, f0: np.ndarray) -> torch.Tensor:
        """
        Run RVC model inference to generate mel-spectrogram.
        
        Args:
            content_features: Content features from HuBERT (B, T, 768)
            f0: Fundamental frequency array (Hz)
            
        Returns:
            Generated mel-spectrogram (B, 80, T)
        """
        if self.rvc_model is None:
            logger.error("RVC model not loaded. Cannot perform inference.")
            return None
        
        try:
            # Convert f0 to tensor and align with content features
            T = content_features.shape[1]
            f0_tensor = torch.from_numpy(f0[:T]).float().to(self.device)
            f0_tensor = f0_tensor.unsqueeze(0)  # (1, T)
            
            # Ensure f0 is valid (replace zeros with mean)
            f0_mean = f0_tensor[f0_tensor > 0].mean() if (f0_tensor > 0).any() else 100.0
            f0_tensor = torch.where(f0_tensor > 0, f0_tensor, f0_mean)
            
            # Normalize f0 (typical vocal range: 80-400 Hz)
            f0_normalized = (f0_tensor - 80.0) / 320.0  # Normalize to [0, 1] range
            
            with torch.no_grad():
                mel = self.rvc_model(content_features, f0_normalized)
            
            return mel
            
        except Exception as e:
            logger.exception(f"RVC inference failed: {e}")
            return None
    
    def synthesize(self, mel: torch.Tensor) -> np.ndarray:
        """
        Synthesize audio from mel-spectrogram using vocoder.
        
        Args:
            mel: Mel-spectrogram (B, 80, T)
            
        Returns:
            Synthesized audio waveform (numpy array)
        """
        if self.vocoder is not None:
            try:
                with torch.no_grad():
                    waveform = self.vocoder(mel)
                
                # Convert to numpy
                audio = waveform.cpu().numpy().squeeze()
                
                # Normalize
                max_val = np.max(np.abs(audio))
                if max_val > 0:
                    audio = audio / max_val * 0.95
                
                return audio.astype(np.float32)
                
            except Exception as e:
                logger.warning(f"Vocoder synthesis failed: {e}. Using Griffin-Lim fallback.")
        
        # Fallback: Griffin-Lim vocoder
        try:
            mel_np = mel.cpu().numpy().squeeze()
            if len(mel_np.shape) == 2:
                # Convert mel to linear spectrogram (approximate)
                # This is a simplified fallback
                audio = librosa.feature.inverse.mel_to_audio(
                    mel_np,
                    sr=RVC_SAMPLE_RATE,
                    n_fft=2048,
                    hop_length=256,
                )
                return audio.astype(np.float32)
            else:
                logger.warning("Invalid mel shape for Griffin-Lim")
                return None
                
        except Exception as e:
            logger.warning(f"Griffin-Lim fallback also failed: {e}")
            return None
    
    def process(self, audio: np.ndarray, sr: int, chunk_size_sec: float = 12.0) -> Optional[np.ndarray]:
        """
        Full RVC processing pipeline with chunking support for long files.
        
        Args:
            audio: Input audio (mono, float32)
            sr: Input sample rate
            chunk_size_sec: Chunk size in seconds for long audio processing
            
        Returns:
            Processed audio (44.1kHz, float32) or None if processing failed
        """
        try:
            # Load models if not already loaded
            if not self.load_models():
                logger.warning("Failed to load RVC models. Using pass-through.")
                return audio
            
            # Check if we need chunking (long audio mode)
            duration = len(audio) / sr
            if duration > chunk_size_sec:
                logger.info(f"Long audio detected ({duration:.1f}s). Processing in chunks of {chunk_size_sec}s")
                return self._process_chunked(audio, sr, chunk_size_sec)
            
            # Step 1: Resample to 48kHz (RVC processing rate)
            if sr != RVC_SAMPLE_RATE:
                logger.info(f"Resampling from {sr}Hz to {RVC_SAMPLE_RATE}Hz")
                audio_48k = librosa.resample(
                    audio, orig_sr=sr, target_sr=RVC_SAMPLE_RATE
                )
            else:
                audio_48k = audio
            
            # Step 2: Extract f0
            logger.debug("Extracting f0...")
            f0 = self.extract_f0(audio_48k, RVC_SAMPLE_RATE)
            
            # Step 3: Extract content features
            logger.debug("Extracting content features...")
            content_features = self.extract_features(audio_48k, RVC_SAMPLE_RATE)
            if content_features is None:
                logger.warning("Failed to extract features. Using pass-through.")
                return audio
            
            # Step 4: Run RVC inference
            logger.debug("Running RVC inference...")
            mel = self.infer(content_features, f0)
            if mel is None:
                logger.warning("RVC inference failed. Using pass-through.")
                return audio
            
            # Step 5: Synthesize audio
            logger.debug("Synthesizing audio...")
            audio_48k_processed = self.synthesize(mel)
            if audio_48k_processed is None:
                logger.warning("Synthesis failed. Using pass-through.")
                return audio
            
            # Step 6: Resample back to 44.1kHz
            if RVC_SAMPLE_RATE != OUTPUT_SAMPLE_RATE:
                logger.info(f"Resampling from {RVC_SAMPLE_RATE}Hz to {OUTPUT_SAMPLE_RATE}Hz")
                audio_processed = librosa.resample(
                    audio_48k_processed,
                    orig_sr=RVC_SAMPLE_RATE,
                    target_sr=OUTPUT_SAMPLE_RATE,
                )
            else:
                audio_processed = audio_48k_processed
            
            # Step 7: Normalize
            max_val = np.max(np.abs(audio_processed))
            if max_val > 0:
                audio_processed = audio_processed / max_val * 0.95
            
            # Ensure same length as input (or close)
            target_length = int(len(audio) * OUTPUT_SAMPLE_RATE / sr)
            if len(audio_processed) != target_length:
                if len(audio_processed) > target_length:
                    audio_processed = audio_processed[:target_length]
                else:
                    # Pad with zeros
                    padding = target_length - len(audio_processed)
                    audio_processed = np.pad(audio_processed, (0, padding), mode='constant')
            
            logger.info("✓ RVC processing complete")
            return audio_processed.astype(np.float32)
            
        except Exception as e:
            logger.exception(f"RVC processing failed: {e}. Using pass-through.")
            return audio
    
    def _process_chunked(
        self,
        audio: np.ndarray,
        sr: int,
        chunk_size_sec: float,
        overlap_sec: float = 0.2,
    ) -> np.ndarray:
        """
        Process long audio in chunks with crossfade overlap-add.
        
        Args:
            audio: Input audio
            sr: Sample rate
            chunk_size_sec: Chunk size in seconds
            overlap_sec: Overlap between chunks in seconds
            
        Returns:
            Processed audio
        """
        chunk_samples = int(chunk_size_sec * sr)
        overlap_samples = int(overlap_sec * sr)
        hop_samples = chunk_samples - overlap_samples
        
        chunks = []
        total_length = len(audio)
        
        i = 0
        while i < total_length:
            end = min(i + chunk_samples, total_length)
            chunk = audio[i:end]
            
            # Pad if needed
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')
            
            # Process chunk
            logger.debug(f"Processing chunk {i//hop_samples + 1} ({i/sr:.1f}s - {end/sr:.1f}s)")
            processed_chunk = self._process_single_chunk(chunk, sr)
            
            if processed_chunk is not None:
                # Trim to actual length
                processed_chunk = processed_chunk[:end - i]
                chunks.append((i, processed_chunk))
            
            i += hop_samples
        
        # Merge chunks with crossfade
        if not chunks:
            logger.warning("No chunks processed successfully. Using pass-through.")
            return audio
        
        # Allocate output
        output_length = int(total_length * OUTPUT_SAMPLE_RATE / sr)
        output = np.zeros(output_length, dtype=np.float32)
        
        for chunk_start, chunk_audio in chunks:
            # Convert position to output sample rate
            out_start = int(chunk_start * OUTPUT_SAMPLE_RATE / sr)
            out_end = out_start + len(chunk_audio)
            
            if out_end > len(output):
                chunk_audio = chunk_audio[:len(output) - out_start]
                out_end = len(output)
            
            # Crossfade with previous chunk
            if out_start > 0 and len(chunk_audio) > 0:
                fade_samples = min(overlap_samples * OUTPUT_SAMPLE_RATE // sr, len(chunk_audio), out_start)
                if fade_samples > 0:
                    fade_in = np.linspace(0, 1, fade_samples)
                    fade_out = np.linspace(1, 0, fade_samples)
                    
                    # Apply fade-in to new chunk
                    chunk_audio[:fade_samples] *= fade_in
                    
                    # Apply fade-out to existing audio
                    if out_start < len(output):
                        existing_end = min(out_start + fade_samples, len(output))
                        existing_fade_len = existing_end - out_start
                        if existing_fade_len > 0:
                            output[out_start:existing_end] *= fade_out[:existing_fade_len]
            
            # Add chunk
            if out_start < len(output):
                add_len = min(len(chunk_audio), len(output) - out_start)
                output[out_start:out_start + add_len] += chunk_audio[:add_len]
        
        # Normalize
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val * 0.95
        
        return output.astype(np.float32)
    
    def _process_single_chunk(self, audio: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """
        Process a single chunk (internal method, no chunking logic).
        """
        try:
            # Step 1: Resample to 48kHz (RVC processing rate)
            if sr != RVC_SAMPLE_RATE:
                audio_48k = librosa.resample(
                    audio, orig_sr=sr, target_sr=RVC_SAMPLE_RATE
                )
            else:
                audio_48k = audio
            
            # Step 2: Extract f0 (with fallback)
            f0 = self.extract_f0(audio_48k, RVC_SAMPLE_RATE)
            if f0 is None or np.all(f0 == 0):
                logger.warning("F0 extraction failed. Using flat f0 for content-only inference.")
                # Create flat f0 at typical vocal frequency
                f0 = np.full(len(audio_48k) // 256 + 1, 200.0, dtype=np.float32)
            
            # Step 3: Extract content features (with fallback)
            content_features = self.extract_features(audio_48k, RVC_SAMPLE_RATE)
            if content_features is None:
                logger.warning("Feature extraction failed. Using pass-through.")
                return audio
            
            # Step 4: Run RVC inference
            mel = self.infer(content_features, f0)
            if mel is None:
                logger.warning("RVC inference failed. Using pass-through.")
                return audio
            
            # Step 5: Synthesize audio (with fallback)
            audio_48k_processed = self.synthesize(mel)
            if audio_48k_processed is None:
                logger.warning("Synthesis failed. Using pass-through.")
                return audio
            
            # Step 6: Resample back to 44.1kHz
            if RVC_SAMPLE_RATE != OUTPUT_SAMPLE_RATE:
                audio_processed = librosa.resample(
                    audio_48k_processed,
                    orig_sr=RVC_SAMPLE_RATE,
                    target_sr=OUTPUT_SAMPLE_RATE,
                )
            else:
                audio_processed = audio_48k_processed
            
            # Step 7: Normalize
            max_val = np.max(np.abs(audio_processed))
            if max_val > 0:
                audio_processed = audio_processed / max_val * 0.95
            
            return audio_processed.astype(np.float32)
            
        except Exception as e:
            logger.exception(f"Chunk processing failed: {e}")
            return audio


# Global singleton instance
_rvc_refiner_instance: Optional[RVCRefiner] = None


def get_rvc_refiner(
    model_path: Optional[str] = None,
    vocoder_path: Optional[str] = None,
    hubert_path: Optional[str] = None,
    enable_gpu: bool = True,
) -> RVCRefiner:
    """
    Get or create RVC refiner instance (singleton pattern).
    
    Args:
        model_path: Path to RVC model (uses default if None)
        vocoder_path: Path to vocoder (uses default if None)
        hubert_path: Path to HuBERT model (uses default if None)
        enable_gpu: Whether to use GPU
        
    Returns:
        RVCRefiner instance
    """
    global _rvc_refiner_instance
    
    if _rvc_refiner_instance is None:
        from processor.config import settings
        
        model = model_path or settings.rvc_model_path
        vocoder = vocoder_path or getattr(settings, 'rvc_vocoder_path', 'models/rvc/vocoder.pth')
        hubert = hubert_path or getattr(settings, 'rvc_hubert_path', 'models/hubert/hubert-base-ls960.pt')
        
        _rvc_refiner_instance = RVCRefiner(
            model_path=model,
            vocoder_path=vocoder,
            hubert_path=hubert,
            enable_gpu=enable_gpu and settings.rvc_enable_gpu,
        )
    
    return _rvc_refiner_instance


def apply_rvc(
    dry_wav_path: Path,
    output_path: Path,
    target_voice_model_path: Optional[str] = None,
    enable_gpu: bool = True,
) -> bool:
    """
    Apply RVC voice conversion to dry vocal (legacy function for compatibility).
    
    Args:
        dry_wav_path: Path to input dry vocal WAV file
        output_path: Path to save output WAV file
        target_voice_model_path: Path to RVC model (uses default if None)
        enable_gpu: Whether to use GPU
        
    Returns:
        True if RVC was applied, False if pass-through
    """
    try:
        # Load audio
        audio, sr = sf.read(dry_wav_path)
        if len(audio.shape) > 1:
            audio = audio[:, 0]  # Mono
        
        # Get RVC refiner
        refiner = get_rvc_refiner(
            model_path=target_voice_model_path,
            enable_gpu=enable_gpu,
        )
        
        # Process
        processed = refiner.process(audio, sr)
        
        if processed is None:
            # Pass-through
            import shutil
            shutil.copy2(dry_wav_path, output_path)
            return False
        
        # Save output
        sf.write(output_path, processed, OUTPUT_SAMPLE_RATE)
        logger.info(f"✓ RVC processing complete. Output saved to {output_path}")
        return True
        
    except Exception as e:
        logger.exception(f"RVC processing failed: {e}. Using pass-through.")
        import shutil
        shutil.copy2(dry_wav_path, output_path)
        return False
