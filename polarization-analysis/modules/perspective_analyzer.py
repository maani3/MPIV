from googleapiclient import discovery
import time

class PerspectiveAnalyzer:
    """
    A class to interact with the Google Perspective API.
    """
    def __init__(self, api_key):
        self.client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            static_discovery=False,
        )

    def get_scores(self, text):
        """
        Analyzes a text string and returns scores for various attributes.
        """
        if not text:
            return None

        analyze_request = {
            'comment': {'text': text},
            'requestedAttributes': {
                'TOXICITY': {},
                'SEVERE_TOXICITY': {},
                'IDENTITY_ATTACK': {},
                'INSULT': {},
                'THREAT': {}
            }
        }
        
        try:
            response = self.client.comments().analyze(body=analyze_request).execute()
            scores = {
                attr: response['attributeScores'][attr]['summaryScore']['value']
                for attr in response['attributeScores']
            }
            return scores
        except Exception as e:
            print(f"Perspective API Error: {e}")
            # Add a small delay to avoid hitting rate limits if there are many errors
            time.sleep(2)
            return {
                'TOXICITY': 'ERROR',
                'SEVERE_TOXICITY': 'ERROR',
                'IDENTITY_ATTACK': 'ERROR',
                'INSULT': 'ERROR',
                'THREAT': 'ERROR'
            }