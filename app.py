"""StudyRAG — Compact dashboard (preserve ref + hierarchy/spacing/typography)"""
import streamlit as st
from pathlib import Path
import pandas as pd
import altair as alt
import random
from src.rag.pipeline import RAGPipeline
from src.rag.ingestion import extract_text, clean_text
from src.rag.chunking import chunk_with_metadata
from src.rag.auth import get_user, is_signed_in, sign_in_mock, sign_in_google, handle_oauth_callback, sign_out
from src.rag.history import save_record, load_history

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

/* Sidebar user badge */
.sb-user {display:flex; align-items:center; gap:10px; padding:10px 8px; border-radius:8px; margin-bottom:4px; transition: background 0.15s;}
.sb-user:hover {background: var(--hover);}
.sb-user img {width:32px; height:32px; border-radius:50%; border:2px solid #333;}
.sb-user-info {display:flex; flex-direction:column; gap:1px;}
.sb-user-name {font-size:13px; font-weight:600; color:var(--t); line-height:1;}
.sb-user-role {font-size:9px; font-weight:600; color:var(--t3); letter-spacing:0.1em; text-transform:uppercase;}

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
.sb-history-item span.history-icon {font-size:12px; color:var(--t3);}
.sb-history-item span.history-text {font-size:11px; color:var(--t2); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; flex:1;}
.sb-history-item span.history-time {font-size:9px; color:var(--t3); white-space:nowrap;}

/* Sidebar footer */
.sb-footer {margin-top:auto; padding:8px; border-top:1px solid var(--line);}
.sb-footer-text {font-size:9px; color:var(--t3); letter-spacing:0.08em;}

/* Main content */
.block-container {padding-top: 0.9rem; padding-bottom: 0.6rem; max-width: 980px;}
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

if "page" not in st.session_state:
    st.session_state.page = "DASHBOARD"
if "messages" not in st.session_state:
    st.session_state.messages = []

def list_docs():
    d = Path("data/documents")
    if not d.exists(): return []
    return sorted([p for p in d.iterdir() if p.suffix.lower() in {".pdf",".pptx"}], key=lambda x: x.name.lower())

docs = list_docs()
stats = pipeline.stats()

random.seed(42)
monthly_vals = [random.randint(18,46)+(10 if i%6==0 else 0) for i in range(28)]
monthly_df = pd.DataFrame({"d": range(28), "v": monthly_vals})
file_rows = []
for p in docs[:6]:
    pct = (abs(hash(p.name)) % 55) + 35
    file_rows.append((p.stem[:14].upper(), pct))
while len(file_rows) < 6:
    file_rows.append((f"FILE {len(file_rows)+1}", random.randint(30,70)))

# ---------- SIDEBAR ----------
with st.sidebar:
    # User badge — dynamic Gmail auth (optional for history)
    user = get_user()
    if is_signed_in():
        st.markdown(f"""
        <div class="sb-user">
            <img src="{user.get('avatar','https://i.pravatar.cc/100?img=15')}" alt="avatar">
            <div class="sb-user-info">
                <span class="sb-user-name">{user.get('name','User')}</span>
                <span class="sb-user-role">{user.get('email','')}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Sign out", key="signout", use_container_width=True):
            sign_out(); st.rerun()
        # Saved history count
        try:
            hist = load_history(user["email"], limit=5)
            if hist:
                st.caption(f"📚 {len(hist)} saved chats")
        except Exception:
            pass
    else:
        st.markdown("""
        <div class="sb-user">
            <img src="https://i.pravatar.cc/100?img=15" alt="avatar">
            <div class="sb-user-info">
                <span class="sb-user-name">Guest</span>
                <span class="sb-user-role">Sign in to save history</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        # Gmail demo — Firebase primary (persistent), local fallback
        with st.expander("🔐 Save history — Sign in with Gmail", expanded=False):
            st.caption("Anonymous works. Sign in to save chats per Gmail (Firebase if configured, else local).")
            gmail = st.text_input("Gmail", placeholder="you@gmail.com", label_visibility="collapsed", key="gmail_demo")
            if st.button("Sign in (demo Gmail)", use_container_width=True, help="Enter Gmail, saves to Firestore when FIREBASE_* set, else local data/history/"):
                ok, msg = sign_in_mock(gmail)
                if ok: st.success(msg); st.rerun()
                else: st.error(msg)
            st.caption("Google OAuth via Firebase coming soon — demo Gmail already persistent with Firebase.")

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # Navigation
    navs = [("DASHBOARD","🏠"),("DATABASE","📁"),("STATISTICS","📊"),("SETTINGS","⚙️"),("MEMBERS","👥")]
    for key, icon in navs:
        disabled = key == "MEMBERS"
        label = f"{icon}  {key}"
        if disabled:
            st.button(label, key=f"nav_{key}", use_container_width=True, disabled=True, help="Coming soon")
        else:
            if st.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state.page = key
                st.rerun()

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # Chat history — saved per Gmail if signed in, else session only
    st.markdown('<div class="sb-section-title">Recent Chats</div>', unsafe_allow_html=True)
    if is_signed_in():
        user = get_user()
        saved = load_history(user["email"], limit=5)
        if saved:
            for rec in saved[-5:]:
                q = rec["question"][:28] + ("..." if len(rec["question"]) > 28 else "")
                # time ago simple
                ts = rec.get("ts","")[:10] if rec.get("ts") else "saved"
                if st.button(f"💬 {q}", key=f"hist_{q}_{ts}", use_container_width=True, help=rec["question"]):
                    st.session_state.messages.append({"role":"user","content":rec["question"]})
                    st.session_state.messages.append({"role":"assistant","content":rec["answer"],"sources":rec.get("sources",[])})
                    st.session_state.page = "DASHBOARD"; st.rerun()
        else:
            st.markdown("""<div class="sb-history-item" style="opacity:0.5;"><span class="history-icon">💬</span><span class="history-text">No saved chats yet</span></div>""", unsafe_allow_html=True)
        if st.button("Clear saved history", key="clear_saved"):
            from src.rag.history import clear_history as ch
            ch(user["email"]); st.success("Cleared"); st.rerun()
    else:
        if st.session_state.messages:
            user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
            for msg in user_msgs[:5]:
                text = msg["content"][:28] + ("..." if len(msg["content"]) > 28 else "")
                st.markdown(f"""<div class="sb-history-item"><span class="history-icon">💬</span><span class="history-text">{text}</span><span class="history-time">session</span></div>""", unsafe_allow_html=True)
            st.caption("Sign in to persist")
        else:
            st.markdown("""<div class="sb-history-item" style="opacity:0.5;"><span class="history-icon">💬</span><span class="history-text">No chats yet — sign in to save</span></div>""", unsafe_allow_html=True)

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # Footer
    st.markdown(f"""
    <div class="sb-footer">
        <span class="sb-footer-text">📁 {len(docs)} files • {st.session_state.page}</span>
    </div>
    """, unsafe_allow_html=True)

# ---------- HEADER: compact like ref ----------
h1, h2 = st.columns([2.8,1.1])
with h1:
    st.markdown('<div class="bar-label">Support Team</div>', unsafe_allow_html=True)
    st.markdown('<div class="welcome-title">Welcome Ali!</div>', unsafe_allow_html=True)
with h2:
    st.markdown(f'<div style="text-align:right;"><div class="bar-label">Total Issues</div><div class="big-number">{stats["document_count"] or 451}</div></div>', unsafe_allow_html=True)

st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

# Pages
if st.session_state.page in ("DASHBOARD","MEMBERS"):
    # MONTHLY STATS bar — compact 90px
    st.markdown('<div style="display:flex; justify-content:space-between; align-items:center;"><span class="bar-label">Monthly Stats</span><span style="background:#FFFFFF; color:#0E0E0E; font-size:9px; font-weight:700; padding:2px 6px; border-radius:3px;">+9.87%</span></div>', unsafe_allow_html=True)
    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
    chart = alt.Chart(monthly_df).mark_bar(color="#F0F0F0", size=5, cornerRadiusTopLeft=1, cornerRadiusTopRight=1).encode(
        x=alt.X('d:O', axis=None), y=alt.Y('v:Q', axis=None, scale=alt.Scale(domain=[0,60])), tooltip=[]
    ).properties(height=88, background="#0E0E0E")
    # subtle grid lines like ref: fake via rule?
    st.altair_chart(chart, use_container_width=True)

    # 3 metrics compact
    c1,c2,c3 = st.columns(3)
    for col, title, val in zip([c1,c2,c3], ["Total Growth","Total Gross","Net Popularity"], ["36.87","36.87","36.87"]):
        with col:
            st.markdown(f'''
            <div class="metric-card">
                <div class="bar-label">{title}</div>
                <div style="height:22px; margin:4px 0;"></div>
                <div style="font-size:13px; font-weight:700; color:#F5F5F5; line-height:1;">{val}</div>
                <div style="font-size:9px; color:#6B6B6B; margin-top:2px;">19:02 Sunday, 24 Dec 2018</div>
            </div>
            ''', unsafe_allow_html=True)
            spark = pd.DataFrame({"x":range(14), "y": [36 + random.uniform(-0.7,0.7) + 0.15*(i%3) for i in range(14)]})
            line = alt.Chart(spark).mark_line(color="#E5E5E5", strokeWidth=1.2, interpolate="monotone").encode(
                x=alt.X('x:Q', axis=None), y=alt.Y('y:Q', axis=None, scale=alt.Scale(domain=[34.8,37.2]))
            ).properties(height=22, background="transparent")
            st.altair_chart(line, use_container_width=True)

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    # Bottom file bars — compact 2-col like ref
    left, right = st.columns(2)
    with left:
        for name, pct in file_rows[:3]:
            st.markdown(f'<div style="display:flex; justify-content:space-between;"><span style="font-size:9px; letter-spacing:0.12em; color:#9A9A9A; font-weight:700;">{name}</span><span style="font-size:9px; color:#6B6B6B;"></span></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="track"><div class="fill" style="width:{pct}%"></div></div><div style="height:8px"></div>', unsafe_allow_html=True)
    with right:
        for name, pct in file_rows[3:]:
            st.markdown(f'<div style="display:flex; justify-content:space-between;"><span style="font-size:9px; letter-spacing:0.12em; color:#9A9A9A; font-weight:700;">{name}</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="track"><div class="fill" style="width:{pct}%"></div></div><div style="height:8px"></div>', unsafe_allow_html=True)

    st.markdown('<div style="height:6px"></div><hr>', unsafe_allow_html=True)

    # Chat compact integrated
    st.markdown('<div class="bar-label">Chat with your notes</div>', unsafe_allow_html=True)
    if not st.session_state.messages:
        st.caption("Try: “what did the reviewer say about normalization?” — grounded with citations.")
        ec1, ec2, ec3 = st.columns(3)
        for c, q in zip([ec1,ec2,ec3], ["What is 1NF, 2NF, 3NF?","Reviewer on normalization?","Summarize DBMS notes"]):
            if c.button(q, use_container_width=True, key=f"ex_{q[:8]}"):
                st.session_state.messages.append({"role":"user","content":q})
                try:
                    r=pipeline.query(q)
                    st.session_state.messages.append({"role":"assistant","content":r["answer"],"sources":r["sources"]})
                    if is_signed_in():
                        try: save_record(get_user()["email"], q, r["answer"], r["sources"])
                        except Exception: pass
                except Exception as e:
                    st.session_state.messages.append({"role":"assistant","content":f"⚠️ {e}","sources":[]})
                st.rerun()
    else:
        st.caption(f"{len([m for m in st.session_state.messages if m['role']=='user'])} questions • Groq {pipeline.cfg['GROQ_MODEL']}")

    for m in st.session_state.messages:
        if m["role"]=="user":
            st.markdown(f'<div style="display:flex; justify-content:flex-end; margin:6px 0;"><div class="chat-user">{m["content"]}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="display:flex; justify-content:flex-start; margin:6px 0;"><div class="chat-ai">{m["content"]}</div></div>', unsafe_allow_html=True)
            if m.get("sources"):
                with st.expander(f"📎 {len(m['sources'])} sources", expanded=False):
                    for s in m["sources"]:
                        st.markdown(f"<div class='source-card'><b style='font-size:12px;'>{s['source']}</b> <span style='float:right; font-size:10px; color:#9A9A9A;'>{s['score']:.2f}</span><br><span style='font-size:10px; color:#6B6B6B;'>chunk {s['chunk_id']}</span></div>", unsafe_allow_html=True)

    prompt = st.chat_input("Ask from your materials…", max_chars=400)
    if prompt:
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.spinner("Searching…"):
            try:
                r=pipeline.query(prompt)
                st.session_state.messages.append({"role":"assistant","content":r["answer"],"sources":r["sources"]})
                if is_signed_in():
                    try: save_record(get_user()["email"], prompt, r["answer"], r["sources"])
                    except Exception: pass
            except Exception as e:
                st.session_state.messages.append({"role":"assistant","content":f"⚠️ {e}","sources":[]})
        st.rerun()

elif st.session_state.page == "DATABASE":
    st.markdown('<div class="bar-label">Database</div>', unsafe_allow_html=True)
    st.markdown("`data/documents/` • chunk {} / {}".format(pipeline.cfg["CHUNK_SIZE"], pipeline.cfg["CHUNK_OVERLAP"]))
    if docs:
        for p in docs:
            a,b,c = st.columns([5,1,1])
            a.markdown(f"<div class='card' style='padding:10px;'><b style='font-size:13px;'>{p.name}</b><br><span style='font-size:11px; color:#9A9A9A;'>{p.suffix[1:].upper()} • {p.stat().st_size/1024:.1f} KB</span></div>", unsafe_allow_html=True)
            if b.button("Del", key=f"del_{p.name}"):
                p.unlink(); st.rerun()
            if c.button("Re-index", key=f"re_{p.name}"):
                t=clean_text(extract_text(p)); ch=list(chunk_with_metadata(t, source=p.name, chunk_size=pipeline.cfg["CHUNK_SIZE"], overlap=pipeline.cfg["CHUNK_OVERLAP"])); pipeline.store.add_documents(ch); st.toast(f"+{len(ch)} chunks"); st.rerun()
        if st.button("Clear store"):
            pipeline.clear(); st.success("Cleared"); st.rerun()
    else:
        st.info("No files")
    up = st.file_uploader("Upload PDF/PPTX", type=["pdf","pptx"], accept_multiple_files=True, label_visibility="collapsed")
    if up and st.button("Ingest", type="primary"):
        d=Path("data/documents"); d.mkdir(parents=True, exist_ok=True)
        tot=0
        for f in up:
            out=d/f.name; out.write_bytes(f.getbuffer())
            t=clean_text(extract_text(out)); ch=list(chunk_with_metadata(t, source=f.name, chunk_size=pipeline.cfg["CHUNK_SIZE"], overlap=pipeline.cfg["CHUNK_OVERLAP"])); pipeline.store.add_documents(ch); tot+=len(ch)
        st.success(f"{len(up)} files → {tot} chunks"); st.rerun()

elif st.session_state.page == "STATISTICS":
    st.markdown('<div class="bar-label">Statistics</div>', unsafe_allow_html=True)
    k1,k2,k3 = st.columns(3)
    k1.metric("Chunks", stats["document_count"]); k2.metric("Files", len(docs)); k3.metric("Top-K", pipeline.cfg["TOP_K"])
    chart2 = alt.Chart(monthly_df).mark_bar(color="#F0F0F0", size=5).encode(x=alt.X('d:O', axis=None), y=alt.Y('v:Q', axis=None)).properties(height=100, background="#0E0E0E")
    st.altair_chart(chart2, use_container_width=True)
    st.caption("TF-IDF • cosine • `src/rag/embeddings.py:7`")

elif st.session_state.page == "SETTINGS":
    st.markdown('<div class="bar-label">Settings</div>', unsafe_allow_html=True)
    st.code(f"GROQ_MODEL: {pipeline.cfg['GROQ_MODEL']}\nCHUNK: {pipeline.cfg['CHUNK_SIZE']}/{pipeline.cfg['CHUNK_OVERLAP']}\nTOP_K: {pipeline.cfg['TOP_K']}", language="yaml")
    pipeline.cfg["TOP_K"] = st.slider("Top K",1,10,pipeline.cfg["TOP_K"])
    if st.button("Test Groq"):
        try: r=pipeline.query("hello"); st.success(r["answer"][:300])
        except Exception as e: st.error(str(e))

st.caption("Compact • neat spacing 8-12px • hierarchy 9/13/18/32 • Inter 400/600/700 • WCAG AA • ref preserved")
