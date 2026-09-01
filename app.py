"""StudyRAG — Study assistant dashboard"""
import streamlit as st
from pathlib import Path
from src.rag.pipeline import RAGPipeline
from src.rag.ingestion import extract_text, clean_text
from src.rag.chunking import chunk_with_metadata
from src.rag.auth import get_user, is_signed_in, sign_in_mock, sign_in_google, handle_oauth_callback, sign_out
from src.rag.history import save_record, load_history
from src.rag.auth import make_oauth_state, verify_oauth_state, verify_google_id_token
import re
import os, json, html, time, logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def esc(s) -> str:
    """HTML-escape anything interpolated into unsafe_allow_html markdown."""
    return html.escape(str(s), quote=True)

def _is_mock_user(user) -> bool:
    """Demo Gmail logins are spoofable by design — keep their history local-only."""
    return (user or {}).get("provider") in ("mock", "gmail-mock")

def _admin_emails() -> set:
    raw = os.getenv("ADMIN_EMAILS", "")
    return {e.strip().lower() for e in raw.split(",") if e.strip()}

def _can_manage(user) -> bool:
    """Corpus management is admin-only: a VERIFIED Google login whose email is
    allowlisted in ADMIN_EMAILS. Demo/mock logins never qualify — otherwise
    anyone could mock-sign-in as an admin's email and wipe the store."""
    if not user or _is_mock_user(user):
        return False
    if user.get("provider") != "google-verified":
        return False
    return user.get("email", "").lower() in _admin_emails()

def _allow_question() -> bool:
    """Per-session sliding-window rate limit (12 questions / minute)."""
    now = time.time()
    window = [t for t in st.session_state.get("q_times", []) if now - t < 60]
    if len(window) >= 12:
        st.session_state.q_times = window
        return False
    window.append(now)
    st.session_state.q_times = window
    return True

MAX_UPLOAD_BYTES = 200 * 1024 * 1024

def _safe_upload_name(raw_name: str):
    """Return a flattened, whitelist-checked filename, or None if unsafe."""
    name = Path(raw_name).name
    if (not name or name in {".", ".."} or "/" in raw_name or "\\" in raw_name
            or "\x00" in raw_name or name.startswith(".")):
        return None
    return name

st.set_page_config(page_title="StudyRAG", page_icon="◼", layout="wide")

# --- TOKENS: compact + hierarchy ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root { --bg:#0E0E0E; --panel:#171717; --card:#1C1C1C; --card2:#1E1E1E; --line:#242424; --line2:#2A2A2A; --t:#F5F5F5; --t2:#9A9A9A; --t3:#6B6B6B; --accent:#FFFFFF; --hover:#1A1A1A; --active:#222222; --sidebar-w:240px; }
html, body, [class*="css"] {font-family:'Inter',system-ui,sans-serif;}
.stApp {background: var(--bg);}

/* --- Sidebar --- */
section[data-testid="stSidebar"] {background: var(--panel); border-right:1px solid var(--line); min-width: var(--sidebar-w); max-width: var(--sidebar-w);}
section[data-testid="stSidebar"] .block-container {padding: 14px 10px; height: 100vh; overflow-y: auto; display: flex; flex-direction: column;}
section[data-testid="stSidebar"] [data-testid="stMarkdown"] {margin: 0;}

/* Sidebar collapsed state */
section[data-testid="stSidebar"][aria-expanded="false"] {min-width: 54px; max-width: 54px;}
section[data-testid="stSidebar"][aria-expanded="false"] .sb-user-info,
section[data-testid="stSidebar"][aria-expanded="false"] .sb-section-title,
section[data-testid="stSidebar"][aria-expanded="false"] .sb-history-item span.history-text,
section[data-testid="stSidebar"][aria-expanded="false"] .sb-history-item span.history-time,
section[data-testid="stSidebar"][aria-expanded="false"] .sb-footer-text,
section[data-testid="stSidebar"][aria-expanded="false"] .sb-auth-card,
section[data-testid="stSidebar"][aria-expanded="false"] .sb-divider + .sb-section-title {display: none;}
section[data-testid="stSidebar"][aria-expanded="false"] .sb-user {justify-content: center; padding: 8px 2px;}
section[data-testid="stSidebar"][aria-expanded="false"] .sb-user svg {width: 26px; height: 26px;}
section[data-testid="stSidebar"][aria-expanded="false"] button[kind="secondary"] {text-align: center; padding: 6px 2px; font-size: 0; min-height: 32px;}
section[data-testid="stSidebar"][aria-expanded="false"] button[kind="secondary"]::before {content: attr(aria-label); font-size: 10px;}

/* Sidebar user badge */
.sb-user {display:flex; align-items:center; gap:10px; padding:10px 8px; border-radius:8px; margin-bottom:4px; transition: background 0.15s;}
.sb-user:hover {background: var(--hover);}
.sb-user svg {width:32px; height:32px; border-radius:50%; background:#2A2A2A; padding:4px; flex-shrink:0;}
.sb-user-info {display:flex; flex-direction:column; gap:1px; min-width:0;}
.sb-user-name {font-size:13px; font-weight:600; color:var(--t); line-height:1; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.sb-user-role {font-size:9px; font-weight:600; color:var(--t3); letter-spacing:0.1em; text-transform:uppercase; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}

/* Nav buttons */
.sb-nav-item {display:flex; align-items:center; gap:9px; padding:8px 10px; border-radius:6px; cursor:pointer; transition: background 0.15s; margin:1px 0;}
.sb-nav-item:hover {background: var(--hover);}
.sb-nav-item.active {background: var(--active); border-left:2px solid var(--accent); padding-left:8px;}
.sb-nav-item span.icon {font-size:14px; width:18px; text-align:center; color:var(--t2);}
.sb-nav-item.active span.icon {color:var(--accent);}
.sb-nav-item span.label {font-size:11px; font-weight:600; letter-spacing:0.06em; color:var(--t2); text-transform:uppercase;}
.sb-nav-item.active span.label {color:var(--t);}

/* Sidebar divider */
.sb-divider {height:1px; background:var(--line); margin:10px 0;}

/* Sidebar section header */
.sb-section-title {font-size:9px; font-weight:700; letter-spacing:0.14em; color:var(--t3); text-transform:uppercase; padding:4px 8px; margin-bottom:4px;}

/* History items */
.sb-history-item {display:flex; align-items:center; gap:8px; padding:7px 8px; border-radius:6px; cursor:pointer; transition: background 0.15s; margin:1px 0;}
.sb-history-item:hover {background: var(--hover);}
.sb-history-item span.history-icon {font-size:11px; color:var(--t3); width:14px; text-align:center; flex-shrink:0;}
.sb-history-item span.history-text {font-size:11px; color:var(--t2); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; flex:1;}
.sb-history-item span.history-time {font-size:9px; color:var(--t3); white-space:nowrap;}

/* Sidebar footer */
.sb-footer {margin-top:auto; padding:8px; border-top:1px solid var(--line);}
.sb-footer-text {font-size:9px; color:var(--t3); letter-spacing:0.08em;}

/* Auth form */
.sb-auth-card {background: var(--card); border:1px solid var(--line); border-radius:8px; padding:14px 12px; margin:8px 0;}
.sb-auth-title {font-size:12px; font-weight:700; color:var(--t); margin-bottom:6px; letter-spacing:0.02em;}
.sb-auth-sub {font-size:10px; color:var(--t3); line-height:1.5; margin-bottom:10px;}

/* Main content — keep 980px centered when sidebar open */
.block-container {padding-top: 0.9rem; padding-bottom: 0.6rem; max-width: 980px !important; width:100% !important; transition: max-width 0.22s ease, width 0.22s ease; margin-left:auto !important; margin-right:auto !important;}
section[data-testid="stMain"] {margin-left: 0 !important; transition: margin-left 0.22s ease;}
/* When sidebar collapsed, stretch to nearly full viewport */
section[data-testid="stSidebar"][aria-expanded="false"] ~ section[data-testid="stMain"] .block-container,
section[data-testid="stSidebar"][aria-expanded="false"] ~ div .block-container,
[data-testid="stSidebar"][aria-expanded="false"] ~ [data-testid="stMain"] .block-container,
.stApp:has([data-testid="stSidebar"][aria-expanded="false"]) [data-testid="stMain"] .block-container,
.stApp:has([data-testid="stSidebar"][aria-expanded="false"]) .block-container,
body:has([data-testid="stSidebar"][aria-expanded="false"]) .block-container {max-width: 1280px !important; width: 96% !important; padding-left: 1rem !important; padding-right: 1rem !important;}
section[data-testid="stSidebar"][aria-expanded="false"] ~ section[data-testid="stMain"],
.stApp:has([data-testid="stSidebar"][aria-expanded="false"]) section[data-testid="stMain"] {margin-left: 0 !important; width: 100% !important;}
/* Hide collapsed sidebar strip completely for true full-width */
section[data-testid="stSidebar"][aria-expanded="false"] {min-width: 0 !important; max-width: 0 !important; width: 0 !important; overflow: hidden !important; border: none !important; padding: 0 !important;}
section[data-testid="stSidebar"][aria-expanded="false"] > div {display:none !important;}
h1, h2, h3 {letter-spacing:-0.03em; color:var(--t);}
.bar-label {font-size:9px; letter-spacing:0.16em; color:var(--t2); font-weight:700; text-transform:uppercase;}
.welcome-sub {font-size:9px; letter-spacing:0.18em; color:var(--t2); font-weight:700; text-transform:uppercase; line-height:1;}
.welcome-title {font-size:18px; font-weight:500; color:var(--t); margin:3px 0 0 0; letter-spacing:-0.02em; line-height:1.1;}
.big-number {font-size:32px; font-weight:700; color:var(--t); line-height:1; letter-spacing:-0.03em;}
.metric-card {background: var(--card); border:1px solid var(--line); border-radius:6px; padding:12px 10px;}
.track {height:3px; background: var(--line); border-radius:999px; overflow:hidden;}
.fill {height:100%; background: var(--accent);}
.chat-user {background:#FFFFFF; color:#0E0E0E; border-radius:12px 12px 2px 12px; padding:9px 11px; max-width:72%; font-size:13px; line-height:1.45;}
.chat-ai {background: var(--card); border:1px solid var(--line); color:var(--t); border-radius:12px 12px 12px 2px; padding:10px 12px; max-width:76%; font-size:13px; line-height:1.5;}
.source-card {background: var(--panel); border:1px solid var(--line); border-left:2px solid #E5E5E5; border-radius:6px; padding:8px 10px; margin:5px 0;}
div[data-testid="stChatInput"] {background: var(--card); border:1px solid var(--line);}
hr {margin: 10px 0; border-color: var(--line);}

/* Primary button — white bg with dark text for visibility (Ingest) */
button[kind="primary"] {
    background: #FFFFFF !important;
    color: #0E0E0E !important;
    border: 1px solid #FFFFFF !important;
}
button[kind="primary"] p {
    color: #0E0E0E !important;
}
button[kind="primary"]:hover {
    background: #E5E5E5 !important;
    border-color: #E5E5E5 !important;
    color: #0E0E0E !important;
}

/* Sidebar nav buttons */
div[data-testid="stSidebar"] button[kind="secondary"] {
    background: transparent;
    border: 1px solid transparent;
    text-align: left;
    padding: 8px 10px;
    border-radius: 6px;
    transition: background 0.15s;
    color: var(--t2);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.06em;
}
div[data-testid="stSidebar"] button[kind="secondary"]:hover {
    background: var(--hover);
    border-color: var(--line2);
}
div[data-testid="stSidebar"] button[kind="secondary"]:disabled {
    opacity: 0.4;
}

/* Sign-in form inputs */
div[data-testid="stSidebar"] input[type="text"],
div[data-testid="stSidebar"] input[type="email"],
div[data-testid="stSidebar"] input[type="password"] {
    background: var(--bg);
    border: 1px solid var(--line2);
    border-radius: 6px;
    color: var(--t);
    font-size: 12px;
    padding: 8px 10px;
}
div[data-testid="stSidebar"] input:focus {
    border-color: var(--t3);
}

/* --- Responsive: mobile / small screens --- */
@media (max-width: 768px) {
    section[data-testid="stSidebar"] {
        min-width: 100vw !important;
        max-width: 100vw !important;
        position: fixed;
        top: 0;
        left: 0;
        z-index: 999;
        height: 100vh;
    }
    section[data-testid="stSidebar"] .block-container {
        padding: 20px 16px;
    }
    .block-container {
        padding: 12px 10px !important;
        max-width: 100% !important;
    }
    .chat-user {max-width: 85%; font-size: 12px;}
    .chat-ai {max-width: 90%; font-size: 12px;}
    h1 {font-size: 16px !important;}
}

@media (max-width: 480px) {
    .block-container {
        padding: 8px 6px !important;
        max-width: 100% !important;
    }
    .chat-user {max-width: 90%; font-size: 11px; padding: 7px 9px;}
    .chat-ai {max-width: 95%; font-size: 11px; padding: 8px 10px;}
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_pipeline():
    return RAGPipeline()
pipeline = get_pipeline()

# handle OAuth return (?code=) before UI
try:
    handle_oauth_callback()
except Exception:
    pass
# Firebase verified Gmail via ?verified_email&id_token (from JS popup).
# Fail closed: the session email comes ONLY from a Firebase-verified id_token.
# A bare ?verified_email=... without a verifiable token signs nobody in.
if "verified_email" in st.query_params:
    email = None
    token = st.query_params.get("id_token")
    if token:
        try:
            from src.rag.auth_firebase import verify_firebase_email
            email = verify_firebase_email(token)
        except Exception:
            logger.exception("Firebase verification raised")
            email = None
    st.query_params.clear()
    if email and email.lower().endswith("@gmail.com"):
        st.session_state["user"] = {"email": email.lower(), "name": email.split("@")[0], "provider": "google-verified"}

if "page" not in st.session_state:
    st.session_state.page = "DASHBOARD"
if "messages" not in st.session_state:
    st.session_state.messages = []

def strip_citations(text):
    """Remove inline [Source: ...] / [source: ...] patterns from answer text."""
    return re.sub(r'\[Source:.*?Chunk:.*?\]', '', text).strip()

def list_docs():
    d = Path("data/documents")
    if not d.exists(): return []
    return sorted([p for p in d.iterdir() if p.suffix.lower() in {".pdf",".pptx"}], key=lambda x: x.name.lower())

docs = list_docs()
stats = pipeline.stats()

# ---------- SIDEBAR ----------
with st.sidebar:
    # User badge
    user = get_user()
    PERSON_SVG = '<svg viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="18" cy="18" r="18" fill="#2A2A2A"/><circle cx="18" cy="14" r="6" fill="#555"/><ellipse cx="18" cy="30" rx="11" ry="9" fill="#555"/></svg>'
    if is_signed_in():
        st.markdown(f"""
        <div class="sb-user">
            {PERSON_SVG}
            <div class="sb-user-info">
                <span class="sb-user-name">{esc(user.get('name','User'))}</span>
                <span class="sb-user-role">{esc(user.get('email',''))}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Sign out", key="signout", use_container_width=True):
            sign_out(); st.rerun()
        try:
            hist = load_history(user["email"], limit=5, local_only=_is_mock_user(user))
            if hist:
                st.caption(f"{len(hist)} saved chats")
        except Exception:
            pass
    else:
        st.markdown(f"""
        <div class="sb-user">
            {PERSON_SVG}
            <div class="sb-user-info">
                <span class="sb-user-name">Guest</span>
                <span class="sb-user-role">Sign in to save history</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="sb-auth-card">
            <div class="sb-auth-title">Save history</div>
            <div class="sb-auth-sub">Anonymous works. Demo Gmail saves locally; Google (verified) saves to Firestore and prevents spoofing.</div>
        </div>
        """, unsafe_allow_html=True)
        gmail = st.text_input("Email", placeholder="you@gmail.com", label_visibility="collapsed", key="gmail_demo")
        if st.button("Sign in (demo Gmail — not verified)", use_container_width=True, key="btn_signin", help="Demo: anyone can type any Gmail. Use Google below for real verification."):
            ok, msg = sign_in_mock(gmail)
            if ok: st.success(msg + " — ⚠️ demo, spoofable"); st.rerun()
            else: st.error(msg)
        # Real Google - server-side OAuth (works in iframe, no popup block)
        google_id = os.getenv("GOOGLE_CLIENT_ID")
        if not google_id:
            st.caption("Verified Google needs `GOOGLE_CLIENT_ID` + `GOOGLE_CLIENT_SECRET` in Render → Environment. Get from console.cloud.google.com → APIs & Services → Credentials → OAuth client (Web) → add redirect `https://studyrag-4xvz.onrender.com` (+ `http://localhost:8501` for local). Demo Gmail above works now.")
        else:
            import urllib.parse
            redirect_uri = os.getenv("APP_URL", "https://studyrag-4xvz.onrender.com")
            params = {
                "client_id": google_id,
                "redirect_uri": redirect_uri,
                "response_type": "code",
                "scope": "openid email profile",
                "access_type": "offline",
                "prompt": "consent",
            }
            state = make_oauth_state()
            if state:
                params["state"] = state
            auth_url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params)
            st.link_button("Sign in with Google (verified)", auth_url, use_container_width=True, help="Verifies Gmail ownership — prevents spoofing, saves to Firestore")
            # handle OAuth callback ?code=
            if "code" in st.query_params:
                code = st.query_params["code"]
                try:
                    # Login-CSRF guard: state must be one we signed and fresh
                    if not verify_oauth_state(st.query_params.get("state")):
                        raise ValueError("invalid OAuth state")
                    import requests
                    token_res = requests.post("https://oauth2.googleapis.com/token", data={
                        "code": code,
                        "client_id": google_id,
                        "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
                        "redirect_uri": redirect_uri,
                        "grant_type": "authorization_code",
                    }, timeout=10).json()
                    if token_res.get("error"):
                        raise ValueError(token_res.get("error"))
                    email = verify_google_id_token(token_res.get("id_token"))
                    if not email or not email.lower().endswith("@gmail.com"):
                        raise ValueError("no verified Gmail in token")
                    st.session_state["user"] = {"email": email.lower(), "name": email.split("@")[0], "provider": "google-verified"}
                    st.query_params.clear()
                    st.rerun()
                except Exception:
                    logger.exception("Google sign-in failed")
                    st.query_params.clear()
                    st.error("Google sign-in failed. Please try again.")

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # Navigation
    navs = [("DASHBOARD","Dashboard"),("DATABASE","Database"),("STATISTICS","Statistics"),("SETTINGS","Settings"),("MEMBERS","Members")]
    for key, label in navs:
        disabled = key == "MEMBERS"
        if disabled:
            st.button(label, key=f"nav_{key}", use_container_width=True, disabled=True, help="Coming soon")
        else:
            if st.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state.page = key
                st.rerun()

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # Chat history
    st.markdown('<div class="sb-section-title">Recent Chats</div>', unsafe_allow_html=True)
    if is_signed_in():
        user = get_user()
        saved = load_history(user["email"], limit=5, local_only=_is_mock_user(user))
        if saved:
            for rec in saved[-5:]:
                q = rec["question"][:28] + ("..." if len(rec["question"]) > 28 else "")
                ts = rec.get("ts","")[:10] if rec.get("ts") else ""
                if st.button(f"{q}", key=f"hist_{q}_{ts}", use_container_width=True, help=rec["question"]):
                    st.session_state.messages.append({"role":"user","content":rec["question"]})
                    st.session_state.messages.append({"role":"assistant","content":rec["answer"],"sources":rec.get("sources",[])})
                    st.session_state.page = "DASHBOARD"; st.rerun()
        else:
            st.markdown("""<div class="sb-history-item" style="opacity:0.5;"><span class="history-icon">-</span><span class="history-text">No saved chats</span></div>""", unsafe_allow_html=True)
        if st.button("Clear history", key="clear_saved"):
            from src.rag.history import clear_history as ch
            ch(user["email"], local_only=_is_mock_user(user)); st.success("Cleared"); st.rerun()
    else:
        if st.session_state.messages:
            user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
            for msg in user_msgs[:5]:
                text = msg["content"][:28] + ("..." if len(msg["content"]) > 28 else "")
                st.markdown(f"""<div class="sb-history-item"><span class="history-icon">-</span><span class="history-text">{esc(text)}</span><span class="history-time">session</span></div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="sb-history-item" style="opacity:0.5;"><span class="history-icon">-</span><span class="history-text">No chats yet</span></div>""", unsafe_allow_html=True)

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # Footer
    st.markdown(f"""
    <div class="sb-footer">
        <span class="sb-footer-text">{len(docs)} files / {st.session_state.page.lower()}</span>
    </div>
    """, unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<div class="welcome-title">Welcome</div>', unsafe_allow_html=True)
st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

# ---------- DASHBOARD PAGE ----------
if st.session_state.page in ("DASHBOARD","MEMBERS"):
    # Question guide
    st.markdown('<div class="bar-label">How it works</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:10px;">
        <div class="metric-card">
            <div style="font-size:18px; font-weight:700; color:var(--t); line-height:1;">01</div>
            <div style="font-size:11px; font-weight:600; color:var(--t2); margin-top:6px;">Upload materials</div>
            <div style="font-size:10px; color:var(--t3); margin-top:3px; line-height:1.4;">Add PDFs or PPTX files in the Database section</div>
        </div>
        <div class="metric-card">
            <div style="font-size:18px; font-weight:700; color:var(--t); line-height:1;">02</div>
            <div style="font-size:11px; font-weight:600; color:var(--t2); margin-top:6px;">Ask questions</div>
            <div style="font-size:10px; color:var(--t3); margin-top:3px; line-height:1.4;">Type any question about your uploaded materials below</div>
        </div>
        <div class="metric-card">
            <div style="font-size:18px; font-weight:700; color:var(--t); line-height:1;">03</div>
            <div style="font-size:11px; font-weight:600; color:var(--t2); margin-top:6px;">Get answers</div>
            <div style="font-size:10px; color:var(--t3); margin-top:3px; line-height:1.4;">Receive grounded responses with source citations</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:16px"></div><hr>', unsafe_allow_html=True)

    # Chat
    st.markdown('<div class="bar-label">Ask a question</div>', unsafe_allow_html=True)
    if not st.session_state.messages:
        st.caption("Type a question below or try one of these to get started.")
        ec1, ec2, ec3 = st.columns(3)
        for c, q in zip([ec1,ec2,ec3], ["Summarize these notes", "What are the key concepts?", "Explain the main topics"]):
            if c.button(q, use_container_width=True, key=f"ex_{q[:8]}"):
                st.session_state.messages.append({"role":"user","content":q})
                if not _allow_question():
                    st.session_state.messages.append({"role":"assistant","content":"⏳ Too many questions in a minute — please wait a moment.","sources":[]})
                else:
                    try:
                        r=pipeline.query(q)
                        st.session_state.messages.append({"role":"assistant","content":r["answer"],"sources":r["sources"]})
                        if is_signed_in():
                            try: save_record(get_user()["email"], q, r["answer"], r["sources"], local_only=_is_mock_user(get_user()))
                            except Exception: pass
                    except Exception:
                        logger.exception("query failed")
                        st.session_state.messages.append({"role":"assistant","content":"Something went wrong answering that. Please try again.","sources":[]})
                st.rerun()
    else:
        st.caption(f"{len([m for m in st.session_state.messages if m['role']=='user'])} questions")

    for m in st.session_state.messages:
        if m["role"]=="user":
            st.markdown(f'<div style="display:flex; justify-content:flex-end; margin:6px 0;"><div class="chat-user">{esc(m["content"])}</div></div>', unsafe_allow_html=True)
        else:
            clean_answer = strip_citations(m["content"])
            st.markdown(f'<div style="display:flex; justify-content:flex-start; margin:6px 0;"><div class="chat-ai">{esc(clean_answer)}</div></div>', unsafe_allow_html=True)
            if m.get("sources"):
                with st.expander(f"{len(m['sources'])} sources", expanded=False):
                    for s in m["sources"]:
                        st.markdown(f"<div class='source-card'><b style='font-size:12px;'>{esc(s['source'])}</b> <span style='float:right; font-size:10px; color:#9A9A9A;'>{s['score']:.2f}</span><br><span style='font-size:10px; color:#6B6B6B;'>chunk {esc(s['chunk_id'])}</span></div>", unsafe_allow_html=True)

    prompt = st.chat_input("Ask from your materials…", max_chars=400)
    if prompt:
        st.session_state.messages.append({"role":"user","content":prompt})
        if not _allow_question():
            st.session_state.messages.append({"role":"assistant","content":"⏳ Too many questions in a minute — please wait a moment.","sources":[]})
        else:
            with st.spinner("Searching…"):
                try:
                    r=pipeline.query(prompt)
                    st.session_state.messages.append({"role":"assistant","content":r["answer"],"sources":r["sources"]})
                    if is_signed_in():
                        try: save_record(get_user()["email"], prompt, r["answer"], r["sources"], local_only=_is_mock_user(get_user()))
                        except Exception: pass
                except Exception:
                    logger.exception("query failed")
                    st.session_state.messages.append({"role":"assistant","content":"Something went wrong answering that. Please try again.","sources":[]})
        st.rerun()

elif st.session_state.page == "DATABASE":
    st.markdown('<div class="bar-label">Database</div>', unsafe_allow_html=True)
    st.markdown("`data/documents/` • chunk {} / {} • TF-IDF".format(pipeline.cfg["CHUNK_SIZE"], pipeline.cfg["CHUNK_OVERLAP"]))
    # Upload/delete/clear are destructive or poison the shared corpus — admins only
    can_manage = _can_manage(get_user())
    if not can_manage:
        if is_signed_in():
            st.caption("🔒 Admins only. Add your verified Gmail to `ADMIN_EMAILS` in Render → Environment to manage documents.")
        else:
            st.caption("🔒 Sign in with verified Google (sidebar) to manage documents. Admins are listed in `ADMIN_EMAILS`.")
    if docs:
        for p in docs:
            a,b,c = st.columns([5,1,1])
            a.markdown(f"<div class='card' style='padding:10px;'><b style='font-size:13px;'>{esc(p.name)}</b><br><span style='font-size:11px; color:#9A9A9A;'>{esc(p.suffix[1:].upper())} • {p.stat().st_size/1024:.1f} KB</span></div>", unsafe_allow_html=True)
            if b.button("Del", key=f"del_{p.name}", disabled=not can_manage):
                p.unlink(); st.rerun()
            if c.button("Re-index", key=f"re_{p.name}", disabled=not can_manage):
                t=clean_text(extract_text(p)); ch=list(chunk_with_metadata(t, source=p.name, chunk_size=pipeline.cfg["CHUNK_SIZE"], overlap=pipeline.cfg["CHUNK_OVERLAP"])); pipeline.store.add_documents(ch); st.toast(f"+{len(ch)} chunks"); st.rerun()
        if st.button("Clear store", disabled=not can_manage):
            pipeline.clear(); st.success("Cleared"); st.rerun()
    else:
        st.markdown("""
        <style>
        @keyframes float {0%,100%{transform:translateY(0px)} 50%{transform:translateY(-6px)}}
        @keyframes pulse-border {0%{border-color:#2A2A2A} 50%{border-color:#3A3A3A} 100%{border-color:#2A2A2A}}
        .drag-box {background:#141414; border:2px dashed #2A2A2A; border-radius:12px; padding:22px; text-align:center; animation: pulse-border 2.2s infinite;}
        .drag-icon {font-size:28px; animation: float 1.8s ease-in-out infinite; display:inline-block;}
        </style>
        <div class="drag-box">
            <div class="drag-icon" style="width:28px; height:28px; margin:0 auto; background:#2A2A2A; border-radius:4px; display:flex; align-items:center; justify-content:center;"><svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M6 2 H10 L13 5 V13 H6 Z" fill="#3A3A3A" stroke="#555" stroke-width="0.8"/><path d="M10 2 V5 H13" fill="none" stroke="#555" stroke-width="0.8"/></svg></div>
            <div style="font-size:13px; font-weight:600; color:#F5F5F5; margin-top:8px;">Drop PDFs or PPTX here</div>
            <div style="font-size:11px; color:#9A9A9A; margin-top:4px;">or click <b style="color:#F5F5F5;">Browse files</b> below • Limit 200MB per file</div>
            <div style="font-size:10px; color:#6B6B6B; margin-top:10px; letter-spacing:0.08em;">QUICK GUIDE: 1) Drop file → 2) Click Ingest → 3) Ask in Dashboard</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    up = st.file_uploader("Drag files here or choose files", type=["pdf","pptx"], accept_multiple_files=True, label_visibility="collapsed", help="PDF or PPTX, up to 200MB each")
    if up and can_manage and st.button("Ingest — make searchable", type="primary", use_container_width=True):
        d=Path("data/documents"); d.mkdir(parents=True, exist_ok=True)
        tot=0
        for f in up:
            name = _safe_upload_name(f.name)
            if not name:
                st.warning(f"Skipped unsafe filename: {f.name[:40]}")
                continue
            if f.size > MAX_UPLOAD_BYTES:
                st.warning(f"Skipped {name}: over 200MB")
                continue
            out=d/name; out.write_bytes(f.getbuffer())
            try:
                t=clean_text(extract_text(out)); ch=list(chunk_with_metadata(t, source=name, chunk_size=pipeline.cfg["CHUNK_SIZE"], overlap=pipeline.cfg["CHUNK_OVERLAP"])); pipeline.store.add_documents(ch); tot+=len(ch)
            except Exception:
                logger.exception("Ingest failed for %s", name)
                st.warning(f"Could not read {name} — skipped")
    elif up and not can_manage:
        st.caption("Only admins (ADMIN_EMAILS) can ingest.")

elif st.session_state.page == "STATISTICS":
    st.markdown('<div class="bar-label">Statistics</div>', unsafe_allow_html=True)
    k1,k2,k3 = st.columns(3)
    k1.metric("Chunks", stats["document_count"]); k2.metric("Files", len(docs)); k3.metric("Top-K", pipeline.cfg["TOP_K"])
    st.caption("TF-IDF / cosine similarity")

elif st.session_state.page == "SETTINGS":
    st.markdown('<div class="bar-label">Settings</div>', unsafe_allow_html=True)
    st.code(f"GROQ_MODEL: {pipeline.cfg['GROQ_MODEL']}\nCHUNK: {pipeline.cfg['CHUNK_SIZE']}/{pipeline.cfg['CHUNK_OVERLAP']}\nTOP_K: {pipeline.cfg['TOP_K']}", language="yaml")
    pipeline.cfg["TOP_K"] = st.slider("Top K",1,10,pipeline.cfg["TOP_K"])
    if st.button("Test Groq"):
        try: r=pipeline.query("hello"); st.success(r["answer"][:300])
        except Exception as e: st.error(str(e))
