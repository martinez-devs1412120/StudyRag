"""Firebase Gmail auth helper (mock Gmail + optional Google OAuth placeholder)."""
import os
import json
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

def _firebase_available():
    return bool(os.getenv("FIREBASE_PROJECT_ID") or os.getenv("FIREBASE_CREDENTIALS") or os.getenv("FIREBASE_CREDENTIALS_JSON"))

def get_user():
    return st.session_state.get("user")

def is_signed_in():
    return st.session_state.get("user") is not None

def sign_in_mock(email: str):
    email = email.strip().lower()
    if not email.endswith("@gmail.com"):
        return False, "Use a Gmail address (@gmail.com)"
    st.session_state["user"] = {"email": email, "name": email.split("@")[0], "avatar": f"https://i.pravatar.cc/100?u={email}", "provider": "gmail-mock"}
    return True, f"Signed in as {email} (history will save to Firebase if configured, else local)"

def sign_out():
    st.session_state.pop("user", None)
    st.query_params.clear()

def init_firebase():
    """Init firebase_admin if credentials present. Returns firestore client or None."""
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
        if firebase_admin._apps:
            return firestore.client()
        # Try JSON string env
        cred_json = os.getenv("FIREBASE_CREDENTIALS_JSON")
        cred_path = os.getenv("FIREBASE_CREDENTIALS") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if cred_json:
            info = json.loads(cred_json)
            cred = credentials.Certificate(info)
            firebase_admin.initialize_app(cred, {"projectId": os.getenv("FIREBASE_PROJECT_ID")})
            return firestore.client()
        elif cred_path and os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
            return firestore.client()
        # Try default without creds (for emulator/local)
        return None
    except Exception:
        return None
