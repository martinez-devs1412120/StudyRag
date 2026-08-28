"""Optional Gmail auth — Supabase Google OAuth with mock fallback for local demo."""
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

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
    st.session_state["user"] = {"email": email, "name": email.split("@")[0], "avatar": f"https://i.pravatar.cc/100?u={email}", "provider": "mock"}
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
