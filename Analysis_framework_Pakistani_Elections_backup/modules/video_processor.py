# import sys
# import types
# import os
# import torch
# import numpy as np
# import pandas as pd
# import soundfile as sf
# import subprocess
# import tempfile
# from configparser import ConfigParser

# # =========================================================================
# # GLOBAL PATCHES (MUST RUN BEFORE LIBRARY IMPORTS)
# # =========================================================================

# # -------------------------------------------------------------------------
# # PATCH 1: CUSTOM AUDIO LOADER (Handles MP4 via FFmpeg)
# # -------------------------------------------------------------------------
# def custom_audio_load(filepath, **kwargs):
#     """
#     Robust audio loader that handles both Audio (WAV) and Video (MP4).
#     """
#     temp_wav_path = None
#     try:
#         # Try loading directly
#         data, sample_rate = sf.read(filepath)
#     except Exception:
#         # Use FFmpeg to convert MP4 -> WAV
#         try:
#             fd, temp_wav_path = tempfile.mkstemp(suffix='.wav')
#             os.close(fd)
            
#             command = [
#                 "ffmpeg", "-y", "-i", filepath, 
#                 "-vn", "-acodec", "pcm_s16le", 
#                 "-ar", "16000", "-ac", "1", 
#                 temp_wav_path
#             ]
            
#             subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#             data, sample_rate = sf.read(temp_wav_path)
            
#         except Exception as e:
#             raise RuntimeError(f"Failed to load audio: {e}")
#         finally:
#             if temp_wav_path and os.path.exists(temp_wav_path):
#                 try: os.remove(temp_wav_path)
#                 except: pass

#     # Convert to Tensor
#     tensor = torch.from_numpy(data).float()
#     if tensor.ndim == 1:
#         tensor = tensor.unsqueeze(0)
#     elif tensor.ndim == 2:
#         tensor = tensor.t()

#     return tensor, sample_rate

# # -------------------------------------------------------------------------
# # PATCH 2: CUSTOM AUDIO INFO (Fixes 'has no attribute info')
# # -------------------------------------------------------------------------
# class AudioMetaData:
#     def __init__(self, sample_rate, num_frames, num_channels, bits_per_sample, encoding):
#         self.sample_rate = sample_rate
#         self.num_frames = num_frames
#         self.num_channels = num_channels
#         self.bits_per_sample = bits_per_sample
#         self.encoding = encoding

# def custom_audio_info(filepath):
#     # Use soundfile to get info
#     try:
#         info = sf.info(filepath)
#         return AudioMetaData(
#             sample_rate=info.samplerate,
#             num_frames=info.frames,
#             num_channels=info.channels,
#             bits_per_sample=16, # approximation
#             encoding="PCM_S"
#         )
#     except:
#         # If soundfile fails (e.g. mp4), return dummy info so pipeline doesn't crash
#         # The actual loading will happen via ffmpeg later
#         return AudioMetaData(16000, 0, 1, 16, "PCM_S")

# # Apply Audio Patches
# import torchaudio
# torchaudio.load = custom_audio_load
# torchaudio.info = custom_audio_info # <--- NEW PATCH

# # -------------------------------------------------------------------------
# # PATCH 3: FORCE LEGACY MODEL LOADING
# # -------------------------------------------------------------------------
# _original_torch_load = torch.load
# def patched_load(*args, **kwargs):
#     if 'weights_only' not in kwargs:
#         kwargs['weights_only'] = False
#     return _original_torch_load(*args, **kwargs)

# torch.load = patched_load
# if hasattr(torch.serialization, 'load'):
#     torch.serialization.load = patched_load

# try:
#     if hasattr(_original_torch_load, "__kwdefaults__") and _original_torch_load.__kwdefaults__:
#         _original_torch_load.__kwdefaults__["weights_only"] = False
# except:
#     pass

# # -------------------------------------------------------------------------
# # PATCH 4: RESTORE DELETED ATTRIBUTES
# # -------------------------------------------------------------------------
# if not hasattr(torchaudio, "set_audio_backend"):
#     torchaudio.set_audio_backend = lambda backend: None
# if not hasattr(torchaudio, "get_audio_backend"):
#     torchaudio.get_audio_backend = lambda: "soundfile"
# if not hasattr(torchaudio, "list_audio_backends"):
#     torchaudio.list_audio_backends = lambda: ["soundfile"]

# if "torchaudio.backend" not in sys.modules:
#     backend_pkg = types.ModuleType("torchaudio.backend")
#     backend_pkg.__path__ = []
#     sys.modules["torchaudio.backend"] = backend_pkg
#     torchaudio.backend = backend_pkg
    
#     common_mod = types.ModuleType("torchaudio.backend.common")
#     sys.modules["torchaudio.backend.common"] = common_mod
#     backend_pkg.common = common_mod
#     common_mod.AudioMetaData = AudioMetaData # Use our class defined above

# # -------------------------------------------------------------------------
# # PATCH 5: WHITELIST PYANNOTE CLASSES
# # -------------------------------------------------------------------------
# try:
#     if hasattr(torch.serialization, "add_safe_globals"):
#         class Specifications: pass
#         class Problem: pass
#         class Resolution: pass
#         class Method: pass
#         from torch.torch_version import TorchVersion
#         torch.serialization.add_safe_globals([
#             TorchVersion, Specifications, Problem, Resolution, Method
#         ])
# except:
#     pass

# # -------------------------------------------------------------------------
# # PATCH 6: FIX NUMPY 2.0 (CRITICAL UPDATE)
# # -------------------------------------------------------------------------
# # Restore BOTH variations of NaN
# if not hasattr(np, "NaN"):
#     np.NaN = np.nan
# if not hasattr(np, "NAN"):  # <--- THIS WAS MISSING
#     np.NAN = np.nan

# # =========================================================================
# # MAIN IMPORTS
# # =========================================================================
# import stable_whisper as stable_ts
# from pyannote.audio import Pipeline

# # Read Config
# config = ConfigParser()
# config.read('config.ini')
# HF_TOKEN = config.get('API_KEYS', 'HF_TOKEN')

# class VideoProcessor:
#     def __init__(self, device=None):
#         if device is None:
#             self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         else:
#             self.device = device
        
#         print(f"Using device: {self.device}")
        
#         print("Loading transcription model (Whisper)...")
#         self.transcribe_model = stable_ts.load_model('large', device=self.device)
        
#         print("Loading diarization pipeline (pyannote.audio)...")
        
#         try:
#             self.diarization_pipeline = Pipeline.from_pretrained(
#                 "pyannote/speaker-diarization-3.1",
#                 use_auth_token=HF_TOKEN
#             ).to(torch.device(self.device))
#             print("Models loaded successfully.")
#         except Exception as e:
#             print(f"\nCRITICAL ERROR LOADING PYANNOTE: {e}")
#             print("Trying to apply emergency whitelist fix...")
#             try:
#                 from pyannote.audio.core.task import Specifications, Problem, Resolution
#                 torch.serialization.add_safe_globals([Specifications, Problem, Resolution])
#                 print("Whitelist applied. Retrying load...")
#                 self.diarization_pipeline = Pipeline.from_pretrained(
#                     "pyannote/speaker-diarization-3.1",
#                     use_auth_token=HF_TOKEN
#                 ).to(torch.device(self.device))
#                 print("Retry successful!")
#             except Exception as e2:
#                 print(f"Retry failed: {e2}")
#                 raise e

#     def process_video(self, video_path):
#         print(f"Starting processing for: {video_path}")
        
#         # 1. Transcription
#         print("Step 1/3: Transcribing audio...")
#         result = self.transcribe_model.transcribe(video_path, fp16=torch.cuda.is_available())
        
#         # 2. Diarization
#         print("Step 2/3: Performing speaker diarization...")
#         try:
#             # Try normal pipeline execution
#             diarization = self.diarization_pipeline(video_path)
#         except Exception as e:
#             print(f"Diarization Runtime Error: {e}")
#             print("Attempting manual audio load...")
#             # Fallback
#             waveform, sample_rate = custom_audio_load(video_path)
#             diarization = self.diarization_pipeline({"waveform": waveform, "sample_rate": sample_rate})

#         # 3. Aligning transcription with speakers
#         print("Step 3/3: Aligning transcription with speakers...")
        
#         speaker_ts = []
#         for turn, _, speaker in diarization.itertracks(yield_label=True):
#             speaker_ts.append([turn.start, turn.end, speaker])
        
#         spk_ts_df = pd.DataFrame(speaker_ts, columns=['start', 'end', 'speaker'])
        
#         result_dict = result.to_dict()
#         all_words = []
        
#         if 'segments' in result_dict:
#             for segment in result_dict['segments']:
#                 if 'words' in segment:
#                     all_words.extend(segment['words'])
        
#         words_df = pd.DataFrame(all_words)

#         if words_df.empty:
#             print("No speech detected.")
#             return []

#         words_df['speaker'] = words_df['start'].apply(
#             lambda x: self._get_speaker_for_timestamp(x, spk_ts_df)
#         )

#         utterances = self._group_words_into_utterances(words_df)
        
#         print("Processing complete.")
#         return utterances

#     @staticmethod
#     def _get_speaker_for_timestamp(timestamp, spk_ts_df):
#         for _, row in spk_ts_df.iterrows():
#             if row['start'] <= timestamp <= row['end']:
#                 return row['speaker']
#         return "UNKNOWN_SPEAKER"

#     @staticmethod
#     def _group_words_into_utterances(words_df):
#         if words_df.empty:
#             return []
            
#         utterances = []
#         current_utterance = {
#             'speaker': words_df.iloc[0]['speaker'],
#             'text': '',
#             'start': words_df.iloc[0]['start'],
#             'end': None
#         }
        
#         for i, row in words_df.iterrows():
#             if row['speaker'] == current_utterance['speaker']:
#                 current_utterance['text'] += row['word'] + ' '
#                 current_utterance['end'] = row['end']
#             else:
#                 current_utterance['text'] = current_utterance['text'].strip()
#                 utterances.append(current_utterance)
#                 current_utterance = {
#                     'speaker': row['speaker'],
#                     'text': row['word'] + ' ',
#                     'start': row['start'],
#                     'end': row['end']
#                 }

#         current_utterance['text'] = current_utterance['text'].strip()
#         utterances.append(current_utterance)
        
#         return utterances

import sys
import types
import os
import torch
import numpy as np
import pandas as pd
import soundfile as sf
import subprocess
import tempfile
from configparser import ConfigParser

# =========================================================================
# GLOBAL PATCHES (MUST RUN BEFORE LIBRARY IMPORTS)
# =========================================================================

# -------------------------------------------------------------------------
# PATCH 1: CUSTOM AUDIO LOADER (Handles MP4 via FFmpeg)
# -------------------------------------------------------------------------
def custom_audio_load(filepath, **kwargs):
    """
    Robust audio loader that handles both Audio (WAV) and Video (MP4).
    Uses FFmpeg to convert video audio to a temp WAV file.
    """
    temp_wav_path = None
    try:
        # Try loading directly (works for WAV)
        data, sample_rate = sf.read(filepath)
    except Exception:
        # Use FFmpeg to convert MP4 -> WAV
        try:
            fd, temp_wav_path = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
            
            # Convert to mono 16kHz WAV
            command = [
                "ffmpeg", "-y", "-i", filepath, 
                "-vn", "-acodec", "pcm_s16le", 
                "-ar", "16000", "-ac", "1", 
                temp_wav_path
            ]
            
            # Run silently
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            data, sample_rate = sf.read(temp_wav_path)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load audio from video: {e}")
        finally:
            if temp_wav_path and os.path.exists(temp_wav_path):
                try: os.remove(temp_wav_path)
                except: pass

    # Convert to Tensor
    tensor = torch.from_numpy(data).float()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 2:
        tensor = tensor.t()

    return tensor, sample_rate

# -------------------------------------------------------------------------
# PATCH 2: CUSTOM AUDIO INFO (Fixes 'has no attribute info')
# -------------------------------------------------------------------------
class AudioMetaData:
    def __init__(self, sample_rate, num_frames, num_channels, bits_per_sample, encoding):
        self.sample_rate = sample_rate
        self.num_frames = num_frames
        self.num_channels = num_channels
        self.bits_per_sample = bits_per_sample
        self.encoding = encoding

def custom_audio_info(filepath):
    try:
        info = sf.info(filepath)
        return AudioMetaData(
            sample_rate=info.samplerate,
            num_frames=info.frames,
            num_channels=info.channels,
            bits_per_sample=16,
            encoding="PCM_S"
        )
    except:
        return AudioMetaData(16000, 0, 1, 16, "PCM_S")

import torchaudio
torchaudio.load = custom_audio_load
torchaudio.info = custom_audio_info

# -------------------------------------------------------------------------
# PATCH 3: FORCE LEGACY MODEL LOADING (Security Bypass)
# -------------------------------------------------------------------------
_original_torch_load = torch.load
def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = patched_load
if hasattr(torch.serialization, 'load'):
    torch.serialization.load = patched_load

try:
    if hasattr(_original_torch_load, "__kwdefaults__") and _original_torch_load.__kwdefaults__:
        _original_torch_load.__kwdefaults__["weights_only"] = False
except:
    pass

# -------------------------------------------------------------------------
# PATCH 4: RESTORE DELETED ATTRIBUTES
# -------------------------------------------------------------------------
if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda backend: None
if not hasattr(torchaudio, "get_audio_backend"):
    torchaudio.get_audio_backend = lambda: "soundfile"
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

if "torchaudio.backend" not in sys.modules:
    backend_pkg = types.ModuleType("torchaudio.backend")
    backend_pkg.__path__ = []
    sys.modules["torchaudio.backend"] = backend_pkg
    torchaudio.backend = backend_pkg
    
    common_mod = types.ModuleType("torchaudio.backend.common")
    sys.modules["torchaudio.backend.common"] = common_mod
    backend_pkg.common = common_mod
    common_mod.AudioMetaData = AudioMetaData

# -------------------------------------------------------------------------
# PATCH 5: WHITELIST PYANNOTE CLASSES
# -------------------------------------------------------------------------
try:
    if hasattr(torch.serialization, "add_safe_globals"):
        class Specifications: pass
        class Problem: pass
        class Resolution: pass
        class Method: pass
        from torch.torch_version import TorchVersion
        torch.serialization.add_safe_globals([
            TorchVersion, Specifications, Problem, Resolution, Method
        ])
except:
    pass

# -------------------------------------------------------------------------
# PATCH 6: FIX NUMPY 2.0
# -------------------------------------------------------------------------
if not hasattr(np, "NaN"):
    np.NaN = np.nan
if not hasattr(np, "NAN"):
    np.NAN = np.nan

# =========================================================================
# MAIN IMPORTS
# =========================================================================
import stable_whisper as stable_ts
from pyannote.audio import Pipeline

# Read Config
config = ConfigParser()
config.read('config.ini')
HF_TOKEN = config.get('API_KEYS', 'HF_TOKEN')

class VideoProcessor:
    def __init__(self, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        print("Loading transcription model (Whisper)...")
        # Using 'large' model for best accuracy
        self.transcribe_model = stable_ts.load_model('large', device=self.device)
        
        print("Loading diarization pipeline (pyannote.audio)...")
        
        try:
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=HF_TOKEN
            ).to(torch.device(self.device))
            print("Models loaded successfully.")
        except Exception as e:
            print(f"\nCRITICAL ERROR LOADING PYANNOTE: {e}")
            print("Trying to apply emergency whitelist fix...")
            try:
                from pyannote.audio.core.task import Specifications, Problem, Resolution
                torch.serialization.add_safe_globals([Specifications, Problem, Resolution])
                print("Whitelist applied. Retrying load...")
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=HF_TOKEN
                ).to(torch.device(self.device))
                print("Retry successful!")
            except Exception as e2:
                print(f"Retry failed: {e2}")
                raise e

    def process_video(self, video_path):
        print(f"Starting processing for: {video_path}")
        
        # 1. Transcription
        print("Step 1/3: Transcribing audio (Forcing Hindi/Devanagari)...")
        # language='hi' forces Whisper to output Devanagari script (Hindi).
        # This is required because Perspective API handles Hindi well but fails on Urdu.
        result = self.transcribe_model.transcribe(
            video_path, 
            task='transcribe', 
            fp16=torch.cuda.is_available()
        )
        
        # 2. Diarization
        print("Step 2/3: Performing speaker diarization...")
        try:
            # Try normal pipeline execution
            diarization = self.diarization_pipeline(video_path)
        except Exception as e:
            print(f"Diarization Runtime Error: {e}")
            print("Attempting manual audio load...")
            # Fallback using custom loader
            waveform, sample_rate = custom_audio_load(video_path)
            diarization = self.diarization_pipeline({"waveform": waveform, "sample_rate": sample_rate})

        # 3. Aligning transcription with speakers
        print("Step 3/3: Aligning transcription with speakers...")
        
        speaker_ts = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_ts.append([turn.start, turn.end, speaker])
        
        spk_ts_df = pd.DataFrame(speaker_ts, columns=['start', 'end', 'speaker'])
        
        result_dict = result.to_dict()
        all_words = []
        
        if 'segments' in result_dict:
            for segment in result_dict['segments']:
                if 'words' in segment:
                    all_words.extend(segment['words'])
        
        words_df = pd.DataFrame(all_words)

        if words_df.empty:
            print("No speech detected.")
            return []

        words_df['speaker'] = words_df['start'].apply(
            lambda x: self._get_speaker_for_timestamp(x, spk_ts_df)
        )

        utterances = self._group_words_into_utterances(words_df)
        
        print("Processing complete.")
        return utterances

    @staticmethod
    def _get_speaker_for_timestamp(timestamp, spk_ts_df):
        for _, row in spk_ts_df.iterrows():
            if row['start'] <= timestamp <= row['end']:
                return row['speaker']
        return "UNKNOWN_SPEAKER"

    @staticmethod
    def _group_words_into_utterances(words_df):
        if words_df.empty:
            return []
            
        utterances = []
        current_utterance = {
            'speaker': words_df.iloc[0]['speaker'],
            'text': '',
            'start': words_df.iloc[0]['start'],
            'end': None
        }
        
        for i, row in words_df.iterrows():
            if row['speaker'] == current_utterance['speaker']:
                current_utterance['text'] += row['word'] + ' '
                current_utterance['end'] = row['end']
            else:
                current_utterance['text'] = current_utterance['text'].strip()
                utterances.append(current_utterance)
                current_utterance = {
                    'speaker': row['speaker'],
                    'text': row['word'] + ' ',
                    'start': row['start'],
                    'end': row['end']
                }

        current_utterance['text'] = current_utterance['text'].strip()
        utterances.append(current_utterance)
        
        return utterances