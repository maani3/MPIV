import google.generativeai as genai
import json
from codebook.definitions import CODEBOOK_DEFINITIONS

class LLMAnnotator:
    """
    A class to use a Generative LLM (Gemini Pro) for polarization annotation.
    """
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro-latest')

    def generate_prompt(self, chunk_context, utterance_to_analyze):
        """Constructs the detailed prompt for the LLM."""
        
        prompt = f"""
        **Task: Analyze a single dialogue from a video transcript for political polarization.**

        **Instructions:**
        1.  Read the "Full Context" to understand the conversation's flow.
        2.  Focus *only* on the "Dialogue to Analyze".
        3.  Using the "Codebook Definitions", classify the dialogue into ONE category.
        4.  Identify the 'source' (the speaker) and the 'target' (who the dialogue is directed at).
        5.  Provide a brief justification for your choice.
        6.  Your final output MUST be a single, valid JSON object and nothing else.

        ---

        **Codebook Definitions:**
        {CODEBOOK_DEFINITIONS}

        ---

        **Full Context:**
        "{chunk_context}"

        ---

        **Dialogue to Analyze:**
        - Speaker: {utterance_to_analyze['speaker']}
        - Dialogue: "{utterance_to_analyze['text']}"

        ---

        **Required JSON Output Format:**
        {{
            "source": "The speaker of the dialogue",
            "target": "The person, group, or entity being targeted",
            "category": "The single best category from the codebook",
            "justification": "A brief, one-sentence explanation for your category choice."
        }}
        """
        return prompt

    def annotate_utterance(self, full_transcript, utterance):
        """
        Sends the utterance and context to the LLM for annotation.
        """
        prompt = self.generate_prompt(full_transcript, utterance)
        
        try:
            response = self.model.generate_content(prompt)
            # The API response may be wrapped in markdown JSON formatting ```json ... ```
            cleaned_response = response.text.strip().replace('```json', '').replace('```', '').strip()
            return json.loads(cleaned_response)
        except Exception as e:
            print(f"LLM Annotation Error: {e}")
            print(f"Failed to parse response: {response.text if 'response' in locals() else 'No response'}")
            return {
                "source": "ERROR",
                "target": "ERROR",
                "category": "ERROR",
                "justification": str(e)
            }