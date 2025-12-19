import stable_whisper as stable_ts
import torch
import torchaudio
import pandas as pd
import numpy as np  # <--- NEW IMPORT NEEDED
from configparser import ConfigParser

# --- CRITICAL PATCHES START ---
# 1. Patch Torchaudio (missing set_audio_backend in Torch 2.x)
if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda backend: None

# 2. Patch NumPy (missing np.NaN in NumPy 2.0+)
# pyannote.audio < 3.3.0 uses np.NaN, which was removed. We add it back.
if not hasattr(np, "NaN"):
    np.NaN = np.nan
# --- CRITICAL PATCHES END ---

# Now it is safe to import pyannote
from pyannote.audio import Pipeline

# Read Hugging Face token from config
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
        self.transcribe_model = stable_ts.load_model('base', device=self.device)
        
        print("Loading diarization pipeline (pyannote.audio)...")
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN
        ).to(torch.device(self.device))
        print("Models loaded successfully.")

    def process_video(self, video_path):
        print(f"Starting processing for: {video_path}")
        
        # 1. Transcription
        print("Step 1/3: Transcribing audio...")
        result = self.transcribe_model.transcribe(video_path, fp16=torch.cuda.is_available())
        
        # 2. Diarization
        print("Step 2/3: Performing speaker diarization...")
        diarization = self.diarization_pipeline(video_path)
        
        # 3. Aligning transcription with speakers
        print("Step 3/3: Aligning transcription with speakers...")
        
        speaker_ts = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_ts.append([turn.start, turn.end, speaker])
        
        spk_ts_df = pd.DataFrame(speaker_ts, columns=['start', 'end', 'speaker'])
        
        # Collect words from ALL segments
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

        # Assign speaker to each word
        words_df['speaker'] = words_df['start'].apply(
            lambda x: self._get_speaker_for_timestamp(x, spk_ts_df)
        )

        # Group words by speaker to form utterances
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