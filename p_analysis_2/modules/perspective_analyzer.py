from googleapiclient import discovery
from google.oauth2 import service_account
import time
import os

class PerspectiveAnalyzer:
    def __init__(self, service_account_json):
        # Check if file exists first
        if not os.path.exists(service_account_json):
            print(f"ERROR: Service account file not found at: {service_account_json}")
            self.client = None
            return

        try:
            # This loads the "private_key" and "client_email" from your JSON
            credentials = service_account.Credentials.from_service_account_file(service_account_json)
            
            # Builds the service using the loaded credentials
            self.client = discovery.build(
                "commentanalyzer",
                "v1alpha1",
                credentials=credentials,
                static_discovery=False,
            )
            print("Perspective API initialized successfully.")
            
        except Exception as e:
            print(f"Failed to initialize Perspective API: {e}")
            self.client = None

    def get_scores(self, text):
        if not text or self.client is None:
            return None

        analyze_request = {
            'comment': {'text': text},
            'requestedAttributes': {
                'TOXICITY': {},
                'INSULT': {},
                'THREAT': {},
                'IDENTITY_ATTACK': {}
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
            # Rate limit or API error handling
            print(f"Perspective API Error: {e}")
            time.sleep(1) 
            return None