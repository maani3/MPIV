from googleapiclient import discovery
from google.oauth2 import service_account
import time
import os

class PerspectiveAnalyzer:
    def __init__(self, service_account_json):
        if not os.path.exists(service_account_json):
            print(f"ERROR: Service account file not found at: {service_account_json}")
            self.client = None
            return

        try:
            credentials = service_account.Credentials.from_service_account_file(service_account_json)
            self.client = discovery.build(
                "commentanalyzer",
                "v1alpha1",
                credentials=credentials,
                static_discovery=False,
            )
            print("Perspective API initialized successfully (Hindi Mode).")
            
        except Exception as e:
            print(f"Failed to initialize Perspective API: {e}")
            self.client = None

    def get_scores(self, text):
        if not text or self.client is None:
            return None

        # Request all attributes (Supported in Hindi)
        analyze_request = {
            'comment': {'text': text},
            'languages': ['hi'], # Explicitly set to Hindi
            'requestedAttributes': {
                'TOXICITY': {},
                'SEVERE_TOXICITY': {},
                'INSULT': {},
                'IDENTITY_ATTACK': {},
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
            # print(f"Perspective API Error: {e}") # Uncomment for debug
            return None