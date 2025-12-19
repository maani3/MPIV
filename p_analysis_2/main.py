import os
from configparser import ConfigParser
import pandas as pd
from tqdm import tqdm

from modules.video_processor import VideoProcessor
from modules.perspective_analyzer import PerspectiveAnalyzer
from modules.llm_annotator import LLMAnnotator

def main():
    # 1. Load Configuration
    config = ConfigParser()
    config.read('config.ini')
    
    # --- UPDATED SECTION START ---
    # We now read the JSON path, NOT the API Key
    try:
        PERSPECTIVE_JSON = config.get('API_KEYS', 'PERSPECTIVE_JSON_PATH')
    except Exception as e:
        print(f"Configuration Error: Could not find 'PERSPECTIVE_JSON_PATH' in config.ini. Details: {e}")
        return

    try:
        GEMINI_KEY = config.get('API_KEYS', 'GEMINI_API_KEY')
    except Exception as e:
        print("Configuration Error: Could not find 'GEMINI_API_KEY' in config.ini")
        return
    # --- UPDATED SECTION END ---

    # 2. Initialize Tools
    # Pass the JSON path to the analyzer
    perspective_analyzer = PerspectiveAnalyzer(service_account_json=PERSPECTIVE_JSON)
    
    # Initialize other tools
    try:
        video_processor = VideoProcessor()
        llm_annotator = LLMAnnotator(api_key=GEMINI_KEY)
    except Exception as e:
        print(f"Error initializing modules: {e}")
        return

    # 3. Find Videos to Process
    input_dir = 'data/input_videos'
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure input directory exists
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"Created folder '{input_dir}'. Please put video files inside it.")
        return

    video_files = [f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.mov', '.avi'))]

    if not video_files:
        print(f"No video files found in {input_dir}. Please add some videos and try again.")
        return

    # 4. Process Each Video
    for video_file in video_files:
        video_path = os.path.join(input_dir, video_file)
        print(f"\n{'='*20} Processing Video: {video_file} {'='*20}")

        # Get diarized utterances from the video
        utterances = video_processor.process_video(video_path)
        
        if not utterances:
            print(f"Could not extract any utterances from {video_file}. Skipping.")
            continue
            
        # Create a single string of the full transcript to use as context for the LLM
        full_transcript_context = " ".join([u['text'] for u in utterances])

        results = []
        
        # 5. Analyze each utterance
        print("Analyzing each utterance with Perspective API and LLM...")
        for utterance in tqdm(utterances, desc="Analyzing Utterances"):
            # Get Perspective API scores
            perspective_scores = perspective_analyzer.get_scores(utterance['text'])

            # Get LLM annotations
            llm_annotations = llm_annotator.annotate_utterance(full_transcript_context, utterance)

            # Combine all data
            combined_result = {
                'start_time': utterance['start'],
                'end_time': utterance['end'],
                'speaker': utterance['speaker'],
                'dialogue': utterance['text'],
                'llm_source': llm_annotations.get('source'),
                'llm_target': llm_annotations.get('target'),
                'llm_category': llm_annotations.get('category'),
                'llm_justification': llm_annotations.get('justification'),
                'perspective_toxicity': perspective_scores.get('TOXICITY') if perspective_scores else None,
                'perspective_insult': perspective_scores.get('INSULT') if perspective_scores else None,
                'perspective_identity_attack': perspective_scores.get('IDENTITY_ATTACK') if perspective_scores else None,
                'perspective_threat': perspective_scores.get('THREAT') if perspective_scores else None,
            }
            results.append(combined_result)

        # 6. Save results to CSV
        output_filename = os.path.splitext(video_file)[0] + '_annotated.csv'
        output_path = os.path.join(output_dir, output_filename)
        
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"\nAnalysis complete! Results saved to: {output_path}")

if __name__ == '__main__':
    main()