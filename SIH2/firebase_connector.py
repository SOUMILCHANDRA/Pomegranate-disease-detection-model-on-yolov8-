import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import datetime
import os

class FirebaseConnector:
    def __init__(self, key_path="e:/SIH2/serviceAccountKey.json"):
        if not os.path.exists(key_path):
            print(f"[Firebase] Error: Key file not found at {key_path}")
            self.db = None
            return

        try:
            # Check if already initialized to avoid "App already exists" error
            if not firebase_admin._apps:
                cred = credentials.Certificate(key_path)
                firebase_admin.initialize_app(cred)
            
            self.db = firestore.client()
            print("[Firebase] Connected Successfully.")
        except Exception as e:
            print(f"[Firebase] Connection Failed: {e}")
            self.db = None

    def push_report(self, report_data):
        """
        Pushes the report dictionary to Firestore 'scan_results' collection.
        """
        if not self.db:
            print("[Firebase] Database not initialized. Skipping upload.")
            return

        try:
            # Add Timestamp
            report_data["timestamp"] = datetime.datetime.now()
            
            # Push to 'scan_results' collection
            # .add() returns (update_time, doc_ref)
            update_time, doc_ref = self.db.collection("scan_results").add(report_data)
            
            print(f"[Firebase] Report Uploaded! Doc ID: {doc_ref.id}")
        except Exception as e:
            print(f"[Firebase] Upload Failed: {e}")
