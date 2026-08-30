"""Optional Gmail auth — Supabase Google OAuth with mock fallback for local demo."""
import os
import hmac
import time
import hashlib
import logging
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# OAuth state tokens are valid for this long after issuance (login-CSRF window).
OAUTH_STATE_TTL = 900


def _oauth_secret() -> bytes:
    secret = os.getenv("STUDYRAG_SECRET") or os.getenv("GOOGLE_CLIENT_SECRET") or ""
    return secret.encode()


def make_oauth_state():
    """Stateless signed state for the OAuth redirect. None if no secret is configured."""
    secret = _oauth_secret()
    if not secret:
        return None
    ts = str(int(time.time()))
    sig = hmac.new(secret, ts.encode(), hashlib.sha256).hexdigest()[:32]
    return f"{ts}.{sig}"


def verify_oauth_state(state) -> bool:
    """Validate the signed OAuth state (HMAC + freshness). Fails closed on tampering."""
    secret = _oauth_secret()
    if not secret:
        # Without a secret there is no working OAuth flow anyway (the token
        # exchange needs GOOGLE_CLIENT_SECRET), so nothing to enforce.
        return True
    if not state or "." not in state:
        return False
    ts, sig = state.split(".", 1)
    if not ts.isdigit():
        return False
    expected = hmac.new(secret, ts.encode(), hashlib.sha256).hexdigest()[:32]
    if not hmac.compare_digest(sig, expected):
        return False
    return time.time() - int(ts) <= OAUTH_STATE_TTL


def verify_google_id_token(token):
    """Verify a Google-issued OIDC id_token: signature, audience, issuer, expiry.

    Returns the email claim, or None on ANY failure. Fails closed — the token
    payload is never trusted without a signature check.
    """
    if not token:
        return None
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    if not client_id:
        logger.error("GOOGLE_CLIENT_ID not set; cannot verify id_token")
        return None
    try:
        from google.oauth2 import id_token as google_id_token
        from google.auth.transport import requests as google_requests

        info = google_id_token.verify_oauth2_token(token, google_requests.Request(), client_id)
    except Exception:
        logger.exception("Google id_token verification failed")
        return None
    if info.get("iss") not in ("accounts.google.com", "https://accounts.google.com"):
        return None
    if not info.get("email_verified"):
        return None
    return info.get("email") or None

def get_supabase():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        return None
    try:
        from supabase import create_client
        return create_client(url, key)
    except Exception:
        return None

def get_user():
    return st.session_state.get("user")

def is_signed_in():
    return st.session_state.get("user") is not None

def sign_in_mock(email: str):
    email = email.strip().lower()
    if not email.endswith("@gmail.com"):
        return False, "Use a Gmail address (@gmail.com)"
    st.session_state["user"] = {"email": email, "name": email.split("@")[0], "provider": "mock"}
    return True, "Signed in (demo)"

def sign_in_google():
    sb = get_supabase()
    if sb is None:
        return None, "Supabase not configured — use Gmail demo below"
    try:
        # Supabase will redirect to Google then back to app URL
        redirect = os.getenv("APP_URL", "http://localhost:8501")
        res = sb.auth.sign_in_with_oauth({"provider": "google", "options": {"redirect_to": redirect}})
        # supabase-py returns url
        url = getattr(res, "url", None) or (res.get("url") if isinstance(res, dict) else None)
        if url:
            st.link_button("Continue to Google", url)
        return url, None
    except Exception as e:
        return None, str(e)

def handle_oauth_callback():
    # Supabase redirects with ?code= — supabase handles exchange client-side; try to get session
    params = st.query_params
    code = params.get("code")
    if code:
        sb = get_supabase()
        if sb:
            try:
                sb.auth.exchange_code_for_session({"auth_code": code})
                user = sb.auth.get_user()
                if user and user.user:
                    st.session_state["user"] = {"email": user.user.email, "name": user.user.email, "avatar": user.user.user_metadata.get("avatar_url", ""), "provider": "google"}
                    st.query_params.clear()
                    return True
            except Exception:
                pass
    # Also check existing session
    sb = get_supabase()
    if sb:
        try:
            sess = sb.auth.get_session()
            if sess and sess.user:
                st.session_state["user"] = {"email": sess.user.email, "name": sess.user.email, "provider": "google"}
                return True
        except Exception:
            pass
    return False

def sign_out():
    sb = get_supabase()
    if sb:
        try:
            sb.auth.sign_out()
        except Exception:
            pass
    st.session_state.pop("user", None)
    st.query_params.clear()
