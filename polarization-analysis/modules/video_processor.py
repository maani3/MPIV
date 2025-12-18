import stable_ts
import torch
from pyannote.audio import Pipeline
import pandas as pd
from configparser import ConfigParser

# Read Hugging Face token from config
config = ConfigParser()
config.read('config.ini')
HF_TOKEN = config.get('API_KEYS', 'HF_TOKEN')

class VideoProcessor:
    """
    This class handles the transcription and speaker diarization of a video file.
    """
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
        """
        Transcribes and diarizes a video file, returning a structured list of utterances.
        """
        print(f"Starting processing for: {video_path}")
        
        # 1. Transcription
        print("Step 1/3: Transcribing audio...")
        result = self.transcribe_model.transcribe(video_path, fp16=torch.cuda.is_available())
        
        # 2. Diarization
        print("Step 2/3: Performing speaker diarization...")
        diarization = self.diarization_pipeline(video_path)
        
        # 3. Aligning transcription with diarization
        print("Step 3/3: Aligning transcription with speakers...")
        speaker_ts = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_ts.append([turn.start, turn.end, speaker])
        
        spk_ts_df = pd.DataFrame(speaker_ts, columns=['start', 'end', 'speaker'])
        
        # Get word-level timestamps from Whisper
        word_timestamps = result.to_dict()['segments'][0]['words']
        words_df = pd.DataFrame(word_timestamps)

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
        """Helper to find which speaker was active at a given timestamp."""
        for _, row in spk_ts_df.iterrows():
            if row['start'] <= timestamp <= row['end']:
                return row['speaker']
        return "UNKNOWN_SPEAKER"

    @staticmethod
    def _group_words_into_utterances(words_df):
        """Groups consecutive words from the same speaker into a single utterance."""
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
            else:
                current_utterance['end'] = words_df.iloc[i-1]['end']
                current_utterance['text'] = current_utterance['text'].strip()
                utterances.append(current_utterance)
                current_utterance = {
                    'speaker': row['speaker'],
                    'text': row['word'] + ' ',
                    'start': row['start'],
                    'end': None
                }

        # Add the last utterance
        current_utterance['end'] = words_df.iloc[-1]['end']
        current_utterance['text'] = current_utterance['text'].strip()
        utterances.append(current_utterance)
        
        return utterances