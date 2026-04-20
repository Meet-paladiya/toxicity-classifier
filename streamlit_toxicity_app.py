# streamlit_toxicity_app.py
# Save in project root and run:
#   python -m streamlit run streamlit_toxicity_app.py

import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import base64
import html

# -------------------- helper: data uri for images --------------------
def image_data_uri(p: Path):
    """Return data URI string for file p or None if not present/readable."""
    if not p.exists():
        return None
    ext = p.suffix.lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    try:
        b = p.read_bytes()
        b64 = base64.b64encode(b).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None

# -------------------- page config --------------------
st.set_page_config(page_title="Riot — Toxicity Detector", layout="wide")

# -------------------- assets --------------------
ASSETS = Path("assets")
splash_jpg = ASSETS / "imag.png"

# prefer transparent png then jpg
splash_uri = image_data_uri(splash_jpg)
splash_css = f'url("{splash_uri}")' if splash_uri else "none"

# -------------------- CSS (Riot) --------------------
RIOT_CSS = f"""
<style>
:root{{ --bg:#061016; --panel:#0d1416; --muted:#9aa0a6; --accent:#ff4757; }}

/* Page background (splash or solid overlay) */
[data-testid="stAppViewContainer"] {{
  background: linear-gradient(180deg, rgba(6,10,18,0.55), rgba(6,10,18,0.55)), {splash_css};
  background-size: cover;
  background-position: center;
  background-attachment: fixed;
}}

/* extra top padding so main content is clearly below the fixed logo */
.block-container, .stApp {{
  color: #e6eef0 !important;
  background: transparent !important;
  padding-top: 110px !important;
}}

/* Header/title */
h1 {{
  font-size: 44px !important;
  font-weight: 800 !important;
  color: var(--accent) !important;
  letter-spacing: 0.6px;
  text-shadow: 0 6px 18px rgba(255,71,87,0.06);
}}

/* card look */
.card {{
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)) !important;
  border-radius: 12px !important;
  padding: 16px !important;
  border: 1px solid rgba(255,255,255,0.03) !important;
  box-shadow: 0 6px 28px rgba(0,0,0,0.45) !important;
}}
.card:hover {{
  box-shadow: 0 10px 40px rgba(255,71,87,0.06), 0 0 18px rgba(255,71,87,0.03);
  transform: translateY(-2px);
  transition: all 180ms ease;
}}

/* controls */
.stButton>button {{
  background: var(--accent) !important;
  color: #fff !important;
  border-radius: 10px !important;
  padding: 8px 14px !important;
  box-shadow: 0 6px 20px rgba(255,71,87,0.08) !important;
}}
textarea, .stTextArea>div>div>textarea {{
  background-color: rgba(15,20,20,0.45) !important;
  color: #e6eef0 !important;
  border: 1px solid rgba(255,255,255,0.04) !important;
  border-radius: 8px !important;
}}
.small-muted {{ color: #9aa0a6 !important; font-size:12px; }}

/* neon accent under header */
.header-accent {{
  height: 6px;
  width: 160px;
  border-radius: 8px;
  background: linear-gradient(90deg, rgba(255,71,87,0.95), rgba(255,71,87,0.6));
  box-shadow: 0 6px 30px rgba(255,71,87,0.06);
  margin-top:8px;
}}

/* code blocks readable */
.stCodeBlock, pre, code {{ background: rgba(0,0,0,0.25) !important; color: #e6eef0 !important; }}

/* LOGO: top-right, ABOVE header, fully visible (moved in & smaller) */
.logo-top-right {{
  position: fixed;   /* above all content */
  top: 90px;         /* distance from top of viewport */
  right: 10px;       /* moved inward so not cut by edge */
  z-index: 99999;
  display:flex;
  align-items:center;
  justify-content:center;
  pointer-events: none;  /* so it doesn't capture clicks */
}}
.logo-top-right img {{
  width: 80px;       /* adjust if you want different size */
  height: auto;
  border-radius: 10px;
  background: rgba(0,0,0,0.12);  /* subtle backing */
  border: 2px solid rgba(0,0,0,0.2);
  box-shadow: 0 10px 30px rgba(0,0,0,0.45);
  display:block;
}}

/* extra top padding for header container (just in case) */
.css-1e5imcs {{ padding-top: 40px !important; }}
</style>
"""
st.markdown(RIOT_CSS, unsafe_allow_html=True)

# -------------------- model paths --------------------
MODEL_DIR = Path("models")
TFIDF_PATH = MODEL_DIR / "tfidf.joblib"
CLF_PATH = MODEL_DIR / "baseline_lr.joblib"
EMB_MODEL = MODEL_DIR / "embeddings_clf.joblib"

@st.cache_resource
def load_models():
    tf = None
    clf = None
    emb = None
    if TFIDF_PATH.exists():
        try:
            tf = joblib.load(TFIDF_PATH)
        except Exception as e:
            st.warning(f"Failed to load tfidf: {e}")
    if CLF_PATH.exists():
        try:
            clf = joblib.load(CLF_PATH)
        except Exception as e:
            st.warning(f"Failed to load classifier: {e}")
    if EMB_MODEL.exists():
        try:
            emb = joblib.load(EMB_MODEL)
        except Exception:
            emb = None
    return tf, clf, emb

tfidf, clf, embeddings_clf = load_models()

# -------------------- utilities --------------------
def predict_proba(texts):
    if embeddings_clf is not None:
        try:
            if isinstance(embeddings_clf, dict) and 'embedder' in embeddings_clf and 'clf' in embeddings_clf:
                emb_vecs = embeddings_clf['embedder'].encode(texts)
                probs = embeddings_clf['clf'].predict_proba(emb_vecs)[:, 1]
                return probs
        except Exception:
            pass
    if tfidf is None or clf is None:
        raise RuntimeError("Model artifacts not found. Train baseline first.")
    X = tfidf.transform(texts)
    probs = clf.predict_proba(X)[:, 1]
    return probs

def explain_tokens(text):
    if tfidf is None or clf is None:
        return None
    try:
        feat_names = tfidf.get_feature_names_out()
        X = tfidf.transform([text])
        coefs = clf.coef_[0]
        xarr = X.toarray()[0]
        contributions = xarr * coefs
        top_pos_idx = np.argsort(-contributions)[:12]
        top_neg_idx = np.argsort(contributions)[:12]
        top_pos = [(feat_names[i], float(contributions[i])) for i in top_pos_idx if xarr[i] > 0]
        top_neg = [(feat_names[i], float(contributions[i])) for i in top_neg_idx if xarr[i] > 0]
        return top_pos, top_neg
    except Exception:
        return None

# -------------------- header --------------------
col_h1, _ = st.columns([9, 1])
with col_h1:
    st.markdown("<div style='padding-bottom:6px;'><h1>Toxicity Detector Demo</h1></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='small-muted'>Demo: A baseline toxicity detector (TF-IDF + Logistic Regression). "
        "This app is for demonstration and portfolio purposes only. Use human review for any moderation action.</div>",
        unsafe_allow_html=True,
    )

st.markdown("<div class='header-accent'></div>", unsafe_allow_html=True)
st.markdown("---")

# -------------------- text state + callback for samples --------------------
# Safe initialization BEFORE widget:
if "user_text" not in st.session_state:
    st.session_state["user_text"] = "You are trash and I hate you"

def load_sample(text: str):
    """Callback used by sample buttons to update the textarea value."""
    st.session_state["user_text"] = text

# -------------------- main layout --------------------
col_main, col_right = st.columns([2, 1])

with col_main:
    st.markdown(
        "<div class='card'><h3>Live test your text</h3>"
        "<p class='small-muted'>Type or paste chat messages (English). "
        "The model outputs a toxicity probability and a recommended action.</p></div>",
        unsafe_allow_html=True,
    )

    # Textarea bound to st.session_state["user_text"]
    user_text = st.text_area("Enter chat / comment text", key="user_text", height=160)

    st.markdown("---")
    threshold = st.slider("Action threshold (auto-hide >=)", 0.0, 1.0, 0.85, 0.01)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Predict", key="predict_btn"):
            try:
                text_to_score = st.session_state.get("user_text", "")
                probs = predict_proba([text_to_score])
                proba = float(probs[0])
                label = "TOXIC" if proba >= 0.5 else "NOT TOXIC"
                label_color = "#2ecc71" if label == "NOT TOXIC" else "#ff4757"

                if proba >= threshold:
                    action = "AUTO-HIDE"
                elif proba >= 0.4:
                    action = "FLAG FOR REVIEW"
                else:
                    action = "NO ACTION"

                st.markdown(
                    "<div style='padding:12px;border-radius:8px;background:#071018;"
                    "border:1px solid rgba(255,69,87,0.06)'>"
                    f"<h2 style='color:{label_color};margin:0'>{label} — {proba:.3f}</h2>"
                    "<div class='small-muted'>Recommended action: "
                    f"<span style='background:{label_color};color:white;padding:4px 8px;border-radius:6px'>{action}</span>"
                    "</div></div>",
                    unsafe_allow_html=True,
)

                expl = explain_tokens(text_to_score)
                if expl is not None:
                    top_pos, top_neg = expl
                    with st.expander("Show token contributions"):
                        if top_pos:
                            st.subheader("Top toxic signals")
                            for tok, score in top_pos[:8]:
                                st.write(f"{html.escape(tok)} — {score:.4f}")
                        if top_neg:
                            st.subheader("Top safe signals")
                            for tok, score in top_neg[:8]:
                                st.write(f"{html.escape(tok)} — {score:.4f}")
                else:
                    st.info("Token-level explanations not available for this model.")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    with col_b:
        st.markdown(
            "<div class='card'><h4>Quick samples</h4>"
            "<p class='small-muted'>Click to load a sample into the input box.</p></div>",
            unsafe_allow_html=True,
        )
        samples = [
            "You are an idiot",
            "Great shot! nice game",
            "I will find you and kill you",
            "Get rekt noob",
            "I love this community",
            "She is a great player",
            "Report this cheater",
        ]
        for s in samples:
            st.button(
                s,
                key=f"sample_{s}",
                on_click=load_sample,
                args=(s,),
            )

        # show currently loaded sample / text
        st.write("Current text:")
        st.code(st.session_state.get("user_text", ""))

with col_right:
    st.markdown("<div class='card'><h4>Model Status</h4></div>", unsafe_allow_html=True)

    if tfidf is None or clf is None:
        st.warning("TF-IDF baseline model not found. Train the baseline to use this demo.")
    else:
        st.success("TF-IDF baseline ready")
        st.write("Model: TF-IDF + LogisticRegression")

    if embeddings_clf is not None:
        st.info("Embeddings-based model detected (will be used preferentially)")

    st.markdown("---")
    st.markdown(
        "<div class='small-muted'>Try batch predictions by uploading a CSV with a column named "
        "<code>comment_text</code>.</div>",
        unsafe_allow_html=True,
    )

# -------------------- batch upload --------------------
st.markdown("---")
st.header("Batch predictions")
uploaded = st.file_uploader("Upload CSV (column: comment_text)", type=["csv"])
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        if "comment_text" not in df.columns:
            st.error("CSV must contain a column named 'comment_text'")
        else:
            st.info(f"Running predictions on {len(df)} rows...")
            texts = df["comment_text"].astype(str).tolist()
            probs = predict_proba(texts)
            df["toxicity_proba"] = probs
            df["action"] = df["toxicity_proba"].apply(
                lambda p: "AUTO-HIDE" if p >= threshold else ("REVIEW" if p >= 0.4 else "NO_ACTION")
            )
            st.dataframe(df.head(100))
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download predictions CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.error(f"Batch prediction failed: {e}")

# -------------------- footer --------------------
st.markdown("---")
st.markdown(
    """
    <div class='small-muted'>
    Built as a demo for a Riot-themed toxicity detector. This app uses a TF-IDF + Logistic Regression baseline saved in <code>models/</code>.
    <br><br>
    Limitations: may produce false positives and false negatives — do NOT use this for automated moderation without a human review flow.
    </div>
    """,
    unsafe_allow_html=True,
)
