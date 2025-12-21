# import os
# from configparser import ConfigParser
# import pandas as pd
# from tqdm import tqdm

# from modules.video_processor import VideoProcessor
# from modules.perspective_analyzer import PerspectiveAnalyzer
# from modules.llm_annotator import LLMAnnotator

# def main():
#     # 1. Load Configuration
#     config = ConfigParser()
#     config.read('config.ini')
    
#     # --- UPDATED SECTION START ---
#     # We now read the JSON path, NOT the API Key
#     try:
#         PERSPECTIVE_JSON = config.get('API_KEYS', 'PERSPECTIVE_JSON_PATH')
#     except Exception as e:
#         print(f"Configuration Error: Could not find 'PERSPECTIVE_JSON_PATH' in config.ini. Details: {e}")
#         return

#     try:
#         GEMINI_KEY = config.get('API_KEYS', 'GEMINI_API_KEY')
#     except Exception as e:
#         print("Configuration Error: Could not find 'GEMINI_API_KEY' in config.ini")
#         return
#     # --- UPDATED SECTION END ---

#     # 2. Initialize Tools
#     # Pass the JSON path to the analyzer
#     perspective_analyzer = PerspectiveAnalyzer(service_account_json=PERSPECTIVE_JSON)
    
#     # Initialize other tools
#     try:
#         video_processor = VideoProcessor()
#         llm_annotator = LLMAnnotator(api_key=GEMINI_KEY)
#     except Exception as e:
#         print(f"Error initializing modules: {e}")
#         return

#     # 3. Find Videos to Process
#     input_dir = 'data/input_videos'
#     output_dir = 'output'
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Ensure input directory exists
#     if not os.path.exists(input_dir):
#         os.makedirs(input_dir)
#         print(f"Created folder '{input_dir}'. Please put video files inside it.")
#         return

#     video_files = [f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.mov', '.avi'))]

#     if not video_files:
#         print(f"No video files found in {input_dir}. Please add some videos and try again.")
#         return

#     # 4. Process Each Video
#     for video_file in video_files:
#         video_path = os.path.join(input_dir, video_file)
#         print(f"\n{'='*20} Processing Video: {video_file} {'='*20}")

#         # Get diarized utterances from the video
#         utterances = video_processor.process_video(video_path)
        
#         if not utterances:
#             print(f"Could not extract any utterances from {video_file}. Skipping.")
#             continue
            
#         # Create a single string of the full transcript to use as context for the LLM
#         full_transcript_context = " ".join([u['text'] for u in utterances])

#         results = []
        
#         # 5. Analyze each utterance
#         print("Analyzing each utterance with Perspective API and LLM...")
#         for utterance in tqdm(utterances, desc="Analyzing Utterances"):
#             # Get Perspective API scores
#             perspective_scores = perspective_analyzer.get_scores(utterance['text'])

#             # Get LLM annotations
#             llm_annotations = llm_annotator.annotate_utterance(full_transcript_context, utterance)

#             # Combine all data
#             combined_result = {
#                 'start_time': utterance['start'],
#                 'end_time': utterance['end'],
#                 'speaker': utterance['speaker'],
#                 'dialogue': utterance['text'],
#                 'llm_source': llm_annotations.get('source'),
#                 'llm_target': llm_annotations.get('target'),
#                 'llm_category': llm_annotations.get('category'),
#                 'llm_justification': llm_annotations.get('justification'),
#                 'perspective_toxicity': perspective_scores.get('TOXICITY') if perspective_scores else None,
#                 'perspective_insult': perspective_scores.get('INSULT') if perspective_scores else None,
#                 'perspective_identity_attack': perspective_scores.get('IDENTITY_ATTACK') if perspective_scores else None,
#                 'perspective_threat': perspective_scores.get('THREAT') if perspective_scores else None,
#             }
#             results.append(combined_result)

#         # 6. Save results to CSV
#         output_filename = os.path.splitext(video_file)[0] + '_annotated.csv'
#         output_path = os.path.join(output_dir, output_filename)
        
#         df = pd.DataFrame(results)
#         df.to_csv(output_path, index=False)
#         print(f"\nAnalysis complete! Results saved to: {output_path}")

# if __name__ == '__main__':
#     main()

import os
import pandas as pd
import json
import time
from configparser import ConfigParser
from tqdm import tqdm

# Import modules
from modules.video_processor import VideoProcessor
from modules.perspective_analyzer import PerspectiveAnalyzer
from modules.llm_annotator import LLMAnnotator

def main():
    # 1. Load Configuration
    config = ConfigParser()
    config.read('config.ini')
    
    try:
        PERSPECTIVE_JSON = config.get('API_KEYS', 'PERSPECTIVE_JSON_PATH')
        GEMINI_KEY = config.get('API_KEYS', 'GEMINI_API_KEY')
    except Exception as e:
        print(f"Config Error: {e}")
        return

    # 2. Initialize Tools
    # Note: VideoProcessor loads the heavy models. 
    # Ideally, we only load it if we actually need to process a video, 
    # but initializing it once here is fine for simplicity.
    video_processor = VideoProcessor()
    perspective_analyzer = PerspectiveAnalyzer(service_account_json=PERSPECTIVE_JSON)
    llm_annotator = LLMAnnotator(api_key=GEMINI_KEY)

    # 3. Setup Folders
    input_dir = 'data/input_videos'
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    video_files = [f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.mov', '.avi'))]
    if not video_files:
        print("No videos found.")
        return

    # 4. Processing Loop
    for video_file in video_files:
        video_path = os.path.join(input_dir, video_file)
        video_name_no_ext = os.path.splitext(video_file)[0]
        
        # Define the Cache Filename
        transcript_cache_path = os.path.join(output_dir, f"{video_name_no_ext}_raw_transcript.json")
        
        print(f"\n{'='*20} Processing: {video_file} {'='*20}")

        # PHASE 1: TRANSCRIPTION & DIARIZATION (WITH CACHING)
        raw_utterances = []
        
        # CHECK CACHE
        if os.path.exists(transcript_cache_path):
            print(f"✅ Found cached transcript: {transcript_cache_path}")
            print("Skipping Transcription/Diarization step...")
            try:
                with open(transcript_cache_path, 'r', encoding='utf-8') as f:
                    raw_utterances = json.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}. Re-running processor.")
        
        # IF NO CACHE OR LOAD FAILED
        if not raw_utterances:
            print("Running Transcription & Diarization (This may take time)...")
            raw_utterances = video_processor.process_video(video_path)
            
            if raw_utterances:
                # SAVE TO CACHE IMMEDIATELY
                print(f"Saving transcript to cache: {transcript_cache_path}")
                with open(transcript_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(raw_utterances, f, ensure_ascii=False, indent=2)
        
        if not raw_utterances:
            print("No audio detected or processing failed. Skipping.")
            continue

        # PHASE 2: LLM PASS 1 (Correction & Identification)
        print(f"\n[LLM Phase 1] Translating & Identifying Entities for {len(raw_utterances)} utterances...")
        
        # Check if we have a trace for Phase 1 to resume (Optional, but helpful)
        trace_path_1 = os.path.join(output_dir, f"{video_name_no_ext}_trace_pass1.json")
        structured_data = []
        
        if os.path.exists(trace_path_1):
             print(f"Found Phase 1 Trace. Resuming from: {trace_path_1}")
             with open(trace_path_1, 'r', encoding='utf-8') as f:
                 structured_data = json.load(f)
        else:
            structured_data = llm_annotator.correct_and_structure_transcript(raw_utterances)
            # Save Phase 1 Trace
            with open(trace_path_1, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, ensure_ascii=False, indent=2)
        
        # Merge LLM results back with raw info
        combined_data = []
        for i, raw in enumerate(raw_utterances):
            # Find matching LLM result
            llm_res = next((item for item in structured_data if item.get("id") == i), {})
            
            merged = {
                "id": i,
                "start": raw['start'],
                "end": raw['end'],
                "speaker": raw['speaker'],
                "original_text": raw['text'],
                "hindi_text": llm_res.get("corrected_text", raw['text']), # Using 'corrected_text' from LLM which is now Hindi
                "source_entity": llm_res.get("source_entity", "Unknown"),
                "source_party": llm_res.get("source_background", "None"),
                "target_entity": llm_res.get("target_entity", "None"),
                "target_party": llm_res.get("target_background", "None"),
                "check_toxicity": llm_res.get("check_toxicity", True),
                "perspective_scores": {} 
            }
            combined_data.append(merged)

        # PHASE 3: PERSPECTIVE API
        print("\n[Perspective API] Checking flagged utterances...")
        
        check_count = 0
        for item in tqdm(combined_data, desc="Toxicity Scan"):
            if item["check_toxicity"]:
                check_count += 1
                # Ensure we send the HINDI text to Perspective
                scores = perspective_analyzer.get_scores(item["hindi_text"])
                if scores:
                    item["perspective_scores"] = scores
                
                # Rate Limit Sleep
                time.sleep(1.2)
        
        print(f"Sent {check_count} utterances to Perspective API.")

        # PHASE 4: LLM PASS 2 (Final Classification)
        print("\n[LLM Phase 2] Finalizing Annotations & Categories...")
        
        # We can also check for Phase 2 trace if you want to skip re-running this on crashes
        # But usually we want to re-run this if we tweaked the Perspective Logic
        final_data = llm_annotator.finalize_annotations(combined_data)
        
        # Save Pass 2 Trace
        trace_path_2 = os.path.join(output_dir, f"{video_name_no_ext}_trace_pass2.json")
        with open(trace_path_2, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)

        # PHASE 5: SAVE RESULTS (Multi-Label Expansion)
        csv_rows = []
        if not final_data:
            final_data = []

        for item in combined_data:
            llm_res = next((f for f in final_data if f.get("id") == item["id"]), {})
            annotations = llm_res.get("annotations", [])
            
            base_row = {
                "start_time": item.get("start"),
                "end_time": item.get("end"),
                "speaker_label": item.get("speaker"),
                "original_text": item.get("original_text"),
                "hindi_translation": item.get("hindi_text"),
                "source_entity": item.get("source_entity"),
                "source_party": item.get("source_party"),
                "target_entity": item.get("target_entity"),
                "target_party": item.get("target_party"),
                "toxicity": item.get("perspective_scores", {}).get("TOXICITY", 0.0),
                "insult": item.get("perspective_scores", {}).get("INSULT", 0.0),
                "threat": item.get("perspective_scores", {}).get("THREAT", 0.0)
            }

            if not annotations:
                row = base_row.copy()
                row["polarization_type"] = "None"
                row["polarization_code"] = "Neutral"
                row["justification"] = "No polarization detected."
                csv_rows.append(row)
            else:
                for ann in annotations:
                    row = base_row.copy()
                    row["polarization_type"] = ann.get("type", "Unknown")
                    row["polarization_code"] = ann.get("code", "Unknown")
                    row["justification"] = ann.get("justification", "")
                    csv_rows.append(row)

        output_filename = os.path.join(output_dir, f"{video_name_no_ext}_final_analysis.csv")
        
        df = pd.DataFrame(csv_rows)
        cols = [
            "start_time", "end_time", "speaker_label", 
            "polarization_type", "polarization_code", "justification",
            "original_text", "hindi_translation", 
            "source_entity", "source_party", "target_entity", "target_party",
            "toxicity", "insult", "threat"
        ]
        # Keep existing columns
        existing_cols = [c for c in cols if c in df.columns]
        df = df[existing_cols]
        
        df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"\nSUCCESS! Results saved to: {output_filename}")

if __name__ == '__main__':
    main()