import firebase_admin
from firebase_admin import credentials

cred = credentials.Certificate("deepsearch-f96b5-firebase-adminsdk-0rfb2-b86a06dfbb.json")
firebase_admin.initialize_app(cred)