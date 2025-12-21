# import google.generativeai as genai
# import time
# import json
# import re

# class LLMAnnotator:
#     def __init__(self, api_key):
#         genai.configure(api_key=api_key)
#         # Using 1.5-flash for speed/cost. You can switch to 'gemini-1.5-pro' if you need deeper reasoning.
#         self.model = genai.GenerativeModel('gemini-2.5-flash')

#     def _clean_json_response(self, text):
#         try:
#             return json.loads(text)
#         except:
#             if "```" in text:
#                 text = re.split(r"```(?:json)?", text)[1].split("```")[0]
#             try:
#                 return json.loads(text)
#             except:
#                 return []

#     def correct_and_structure_transcript(self, raw_utterances):
#         """
#         PASS 1: 
#         1. Translate Urdu/Hinglish -> Proper Hindi (Devanagari) for Perspective API.
#         2. Identify Political Entities (Source/Target).
#         """
#         simplified_input = [
#             {"id": i, "speaker": u["speaker"], "text": u["text"]} 
#             for i, u in enumerate(raw_utterances)
#         ]

#         prompt = f"""
#         You are an expert South Asian political translator and analyst.
        
#         INPUT: A video transcript of a Pakistani political discussion.
#         CONTEXT: The speakers speak Urdu. The text might be in Urdu script or rough Hindi.
        
#         TASKS:
#         1. **TRANSLATE TO HINDI**: Convert the text to proper, formal Hindi (Devanagari script). 
#            - Goal: This text will be analyzed by an AI for toxicity, so ensure insults/threats are preserved accurately in Hindi.
#         2. **IDENTIFY ENTITIES**:
#            - "source_entity": Name/Role of speaker.
#            - "source_party": Political affiliation (PTI, PMLN, PPP, Army, etc.).
#            - "target_entity": Name/Group being criticized.
#            - "target_party": Target's affiliation.
#         3. **FLAG TOXICITY**: Set "check_toxicity": true if the text contains insults, threats, sarcasm, or aggressive criticism.

#         OUTPUT FORMAT (JSON List):
#         [
#           {{
#             "id": 0,
#             "hindi_text": "...", 
#             "source_entity": "...",
#             "source_party": "...",
#             "target_entity": "...",
#             "target_party": "...",
#             "check_toxicity": true/false
#           }},
#           ...
#         ]
        
#         RAW TRANSCRIPT:
#         {json.dumps(simplified_input, ensure_ascii=False)}
#         """

#         try:
#             response = self.model.generate_content(prompt)
#             time.sleep(5)
#             return self._clean_json_response(response.text)
#         except Exception as e:
#             print(f"LLM Pass 1 Error: {e}")
#             # Fallback
#             return [{"id": i, "hindi_text": u["text"], "check_toxicity": True} for i, u in enumerate(raw_utterances)]

#     def finalize_annotations(self, enriched_data):
#         """
#         PASS 2: Apply the Polarization Framework.
#         """
#         prompt = f"""
#         You are a political scientist analyzing polarization.
        
#         INPUT: Hindi Utterances with Toxicity Scores (Perspective API).
        
#         FRAMEWORK DEFINITIONS:
#         1. **Personality Polarization**: Attacks on a specific leader's character.
#            - Categories: Strawman, Offensive Language, Absolutism, Invalidation, Vilification/Defamation, Threats, Moral Superiority.
#         2. **Party Polarization**: Attacks on a group/party.
#            - Categories: Strawman, Offensive Language, Absolutism, Invalidation, Otherization (Us vs Them), Threats, Moral Superiority.
#         3. **Issue Polarization**: Disagreement on policies/facts.
#            - Categories: Strawman, Absolutism, Invalidation, Otherization, Fact Denial.
        
#         TASK:
#         For each utterance, determine which DOMAIN (Personality, Party, or Issue) it belongs to, and which CATEGORY.
#         If it doesn't fit (e.g., neutral/chit-chat), use Domain="None", Category="Neutral".
        
#         GUIDELINES:
#         - High 'INSULT' score (>0.5) usually implies "Offensive Language" or "Vilification".
#         - High 'THREAT' score (>0.5) implies "Threats".
#         - If "target_entity" is a person -> Likely Personality Polarization.
#         - If "target_entity" is a group -> Likely Party Polarization.

#         OUTPUT FORMAT (JSON List):
#         [
#           {{
#              "id": 0,
#              "polarization_domain": "Personality" | "Party" | "Issue" | "None",
#              "polarization_category": "Vilification" | "Strawman" | ... | "Neutral",
#              "justification": "English explanation..."
#           }},
#           ...
#         ]
        
#         INPUT DATA:
#         {json.dumps(enriched_data, ensure_ascii=False)}
#         """

#         try:
#             response = self.model.generate_content(prompt)
#             time.sleep(5)
#             return self._clean_json_response(response.text)
#         except Exception as e:
#             print(f"LLM Pass 2 Error: {e}")
#             return []

import google.generativeai as genai
import time
import json
import re

class LLMAnnotator:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        # Using 1.5-flash for speed/cost. You can switch to 'gemini-1.5-pro' if you need deeper reasoning.
        self.model = genai.GenerativeModel('gemini-2.5-pro')

    def _clean_json_response(self, text):
        try:
            return json.loads(text)
        except:
            if "```" in text:
                text = re.split(r"```(?:json)?", text)[1].split("```")[0]
            try:
                return json.loads(text)
            except:
                return []

    def correct_and_structure_transcript(self, raw_utterances):
        """
        PASS 1: 
        1. Translate Urdu/Hinglish -> Proper Hindi (Devanagari) for Perspective API.
        2. Identify Political Entities (Source/Target).
        """
        simplified_input = [
            {"id": i, "speaker": u["speaker"], "text": u["text"]} 
            for i, u in enumerate(raw_utterances)
        ]

        prompt = f"""
        You are an expert South Asian political translator and analyst.
        
        INPUT: A video transcript of a Pakistani political discussion.
        CONTEXT: The speakers speak Urdu. The text might be in Urdu script or rough Hindi.
        
        TASKS:
        1. **TRANSLATE TO HINDI**: Convert the text to proper, formal Hindi (Devanagari script). 
           - Goal: This text will be analyzed by an AI for toxicity, so ensure insults/threats are preserved accurately in Hindi.
        2. **IDENTIFY ENTITIES**:
           - "source_entity": Name/Role of speaker.
           - "source_party": Political affiliation (PTI, PMLN, PPP, Army, etc.).
           - "target_entity": Name/Group being criticized.
           - "target_party": Target's affiliation.
        3. **FLAG TOXICITY**: Set "check_toxicity": true if the text contains insults, threats, sarcasm, or aggressive criticism.

        OUTPUT FORMAT (JSON List):
        [
          {{
            "id": 0,
            "hindi_text": "...", 
            "source_entity": "...",
            "source_party": "...",
            "target_entity": "...",
            "target_party": "...",
            "check_toxicity": true/false
          }},
          ...
        ]
        
        RAW TRANSCRIPT:
        {json.dumps(simplified_input, ensure_ascii=False)}
        """

        try:
            response = self.model.generate_content(prompt)
            time.sleep(5)
            return self._clean_json_response(response.text)
        except Exception as e:
            print(f"LLM Pass 1 Error: {e}")
            # Fallback
            return [{"id": i, "hindi_text": u["text"], "check_toxicity": True} for i, u in enumerate(raw_utterances)]
        
    def finalize_annotations(self, enriched_data):
        """
        PASS 2: Multi-Label Polarization Annotation based on the New Codebook.
        """
        prompt = f"""
        You are an expert political analyst coding transcripts according to a strict Codebook.
        
        INPUT: List of utterances (Hindi/Urdu translation) with Toxicity Scores.
        
        ### CODEBOOK RULES (Strict Adherence Required):
        1. **Unit of Analysis**: Analyze the text segment independently.
        2. **Multi-Label Framework**: A single text may contain MULTIPLE polarization instances. 
           - Example: A text can be BOTH "Party Polarization x Strawman" AND "Personality Polarization x Insult".
           - If multiple apply, list ALL of them.
        3. **Primary Target Rule**: 
           - **Personality Polarization**: Target is a named INDIVIDUAL (e.g., Imran Khan, Nawaz Sharif).
           - **Party Polarization**: Target is a PARTY or COLLECTIVE (e.g., PTI, PMLN, The Army, The Establishment, The Media).
           - **Issue Polarization**: Target is an EVENT, POLICY, or LAW (e.g., May 9th, Election Laws).
        4. **Polarization Codes** (How it is expressed):
           - **Strawman**: Misrepresenting an argument or deflecting to unrelated topics.
           - **Absolutism**: "All/Never/Sole cause", moral finality without nuance.
           - **Vilification/Defamation**: Ridicule, degrading terms (e.g., "Boot licker", "Traitor", "Clown").
           - **Intimidation**: Threats or dominance-asserting language.
           - **Political Fear Mongering**: Existential threats (e.g., "Country will be destroyed").
           - **Moral Superiority**: Claims of exceptional moral/religious righteousness.
           - **Fact-Denial**: Rejecting verifiable evidence.
           - **Invalidation**: Dismissing viewpoints without substance.
           - **Otherisation**: Us vs. Them framing.
           - **Historical Grievances**: Invoking past events to justify hostility.

        ### OUTPUT FORMAT:
        Return a JSON List. If a text has NO polarization, set "annotations": [].
        If it has polarization, "annotations" must be a list of objects.
        
        Example Structure:
        [
          {{
            "id": 0,
            "annotations": [
              {{
                "type": "Party Polarization",
                "code": "Strawman",
                "justification": "Speaker deflects the question about MPOs by attacking the other party's history."
              }},
              {{
                "type": "Personality Polarization",
                "code": "Vilification",
                "justification": "Speaker calls the individual a 'traitor'."
              }}
            ]
          }},
          ...
        ]

        INPUT DATA:
        {json.dumps(enriched_data, ensure_ascii=False)}
        """

        try:
            response = self.model.generate_content(prompt)
            time.sleep(5)
            return self._clean_json_response(response.text)
        except Exception as e:
            print(f"LLM Pass 2 Error: {e}")
            return []