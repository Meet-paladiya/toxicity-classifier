# streamlit_toxicity_app.py
# Save in project root and run:
# python -m streamlit run streamlit_toxicity_app.py

import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import base64
import html
import subprocess
import sys

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except Exception:
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None

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
st.set_page_config(page_title="Toxicity Detector", layout="wide")

# -------------------- assets --------------------
ASSETS = Path("assets")
splash_jpg = ASSETS / "imag.png"

# prefer transparent png then jpg
splash_uri = image_data_uri(splash_jpg)
splash_css = f'url("{splash_uri}")' if splash_uri else "none"

# -------------------- CSS --------------------
RIOT_CSS = f"""
<style>
:root{{ --bg:#061016; --panel:#0d1416; --muted:#9aa0a6; --accent:#fc0303; --accent-soft:#ff6b6b; }}

/* Page background (splash or solid overlay) */
[data-testid="stAppViewContainer"] {{
  background: linear-gradient(180deg, rgba(6,10,18,0.55), rgba(6,10,18,0.55)), {splash_css};
  background-size: cover;
  background-position: center;
  background-attachment: fixed;
}}

/* remove extra top spacing */
.block-container, .stApp {{
  color: #e6eef0 !important;
  background: transparent !important;
  padding-top: 0 !important;
}}

/* Header/title */
h1 {{
  font-size: 44px !important;
  font-weight: 800 !important;
  color: var(--accent) !important;
  letter-spacing: 0.6px;
  text-shadow: 0 6px 18px rgba(252,3,3,0.28);
}}

/* primary app heading glow */
.demo-title {{
  color: var(--accent) !important;
  text-shadow: 0 0 8px rgba(252,3,3,0.95), 0 0 18px rgba(252,3,3,0.58);
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
  box-shadow: 0 10px 40px rgba(252,3,3,0.2), 0 0 18px rgba(252,3,3,0.14);
  transform: translateY(-2px);
  transition: all 180ms ease;
}}

/* controls */
.stButton>button {{
  background: var(--accent) !important;
  color: #fff !important;
  border-radius: 10px !important;
  padding: 8px 14px !important;
  box-shadow: 0 6px 20px rgba(252,3,3,0.24) !important;
}}
textarea, .stTextArea>div>div>textarea {{
  background-color: rgba(15,20,20,0.45) !important;
  color: #e6eef0 !important;
  border: 1px solid rgba(255,255,255,0.04) !important;
  border-radius: 8px !important;
}}
.small-muted {{ color: #9aa0a6 !important; font-size:12px; }}

/* neon accent under header */
.stSlider [data-testid="stWidgetLabel"] p {{
  color: var(--accent) !important;
  font-weight: 700 !important;
}}

.header-accent {{
  height: 6px;
  width: 160px;
  border-radius: 8px;
  background: linear-gradient(90deg, rgba(252,3,3,0.95), rgba(255,107,107,0.72));
  box-shadow: 0 6px 30px rgba(252,3,3,0.24);
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

/* developer credit in top-right corner */
.dev-credit, h2.dev-credit {{
  position: relative;
  top: 24px;
  margin: 0 0 8px 0;
  color: #5ce488 !important;
  text-align: right;
  font-size: 28px;
  text-shadow: 0 2px 10px rgba(0,0,0,0.5);
}}

/* extra top padding for header container (just in case) */
.css-1e5imcs {{ padding-top: 0 !important; }}
</style>
"""
st.markdown(RIOT_CSS, unsafe_allow_html=True)
st.markdown("<h2 class='dev-credit'>Developed by MEET PALADIYA</h2>", unsafe_allow_html=True)

# -------------------- model paths --------------------
MODEL_DIR = Path("models")
TFIDF_PATH = MODEL_DIR / "tfidf.joblib"
CLF_PATH = MODEL_DIR / "baseline_lr.joblib"
EMB_MODEL = MODEL_DIR / "embeddings_clf.joblib"
BERT_MODEL_NAME = "unitary/toxic-bert"
BERT_LOCAL_DIR = MODEL_DIR / "toxic-bert"

def ensure_baseline_artifacts():
    """Train baseline artifacts if they are missing at app startup."""
    if TFIDF_PATH.exists() and CLF_PATH.exists():
        return True

    train_script = Path("src/train_baseline.py")
    if not train_script.exists():
        st.error("Baseline model is missing and training script `src/train_baseline.py` was not found.")
        return False

    with st.spinner("Baseline artifacts missing. Training baseline model..."):
        try:
            result = subprocess.run(
                [sys.executable, str(train_script)],
                check=True,
                capture_output=True,
                text=True,
            )
            st.success("Baseline model trained successfully.")
            if result.stdout:
                st.caption(f"Training log:\n{result.stdout[-1200:]}")
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            details = stderr or stdout or str(exc)
            st.error("Failed to train baseline model automatically.")
            st.caption(f"Training error:\n{details[-1200:]}")
            return False
        except Exception as exc:
            st.error(f"Unexpected training failure: {exc}")
            return False

    return TFIDF_PATH.exists() and CLF_PATH.exists()

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
if tfidf is None or clf is None:
    if ensure_baseline_artifacts():
        load_models.clear()
        tfidf, clf, embeddings_clf = load_models()

@st.cache_resource
def _load_bert_cached():
    if AutoTokenizer is None or AutoModelForSequenceClassification is None or torch is None:
        raise RuntimeError("Missing dependencies: install `transformers` and `torch`.")
    source = BERT_MODEL_NAME
    load_kwargs = {}

    # Prefer local model files first so app works offline/reliably.
    if BERT_LOCAL_DIR.exists():
        source = str(BERT_LOCAL_DIR)
        load_kwargs["local_files_only"] = True

    tokenizer = AutoTokenizer.from_pretrained(source, **load_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(source, **load_kwargs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device

def load_bert():
    try:
        tokenizer, model, device = _load_bert_cached()
        return tokenizer, model, device, None
    except Exception as e:
        return None, None, None, str(e)

bert_tokenizer, bert_model, bert_device, bert_load_error = load_bert()

# -------------------- utilities --------------------
def predict_proba(texts, model_mode="tfidf"):
    if model_mode == "bert":
        if bert_tokenizer is None or bert_model is None or bert_device is None:
            raise RuntimeError("BERT model is not available. Install dependencies and restart the app.")
        # unitary/toxic-bert is multi-label; use sigmoid on the explicit "toxic" logit.
        toxic_idx = 0
        if hasattr(bert_model, "config") and hasattr(bert_model.config, "label2id"):
            toxic_idx = bert_model.config.label2id.get("toxic", 0)
        all_probs = []
        batch_size = 16
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                encoded = bert_tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=256,
                    return_tensors="pt",
                )
                encoded = {k: v.to(bert_device) for k, v in encoded.items()}
                logits = bert_model(**encoded).logits
                if logits.shape[-1] == 1:
                    probs = torch.sigmoid(logits.squeeze(-1))
                else:
                    probs = torch.sigmoid(logits[:, toxic_idx])
                all_probs.extend(probs.detach().cpu().numpy().tolist())
        return np.asarray(all_probs, dtype=float)

    if model_mode == "embeddings":
        if embeddings_clf is None:
            raise RuntimeError("Embeddings model not found.")
        try:
            if isinstance(embeddings_clf, dict) and "embedder" in embeddings_clf and "clf" in embeddings_clf:
                emb_vecs = embeddings_clf["embedder"].encode(texts)
                probs = embeddings_clf["clf"].predict_proba(emb_vecs)[:, 1]
                return probs
        except Exception as e:
            raise RuntimeError(f"Embeddings model failed: {e}") from e

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

def explain_tokens_bert(text):
    """Integrated-gradients token contribution scores for BERT toxicity logit."""
    if bert_tokenizer is None or bert_model is None or bert_device is None or torch is None:
        return None
    try:
        encoded = bert_tokenizer(
            [text],
            truncation=True,
            max_length=256,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        input_ids = encoded["input_ids"].to(bert_device)
        attention_mask = encoded["attention_mask"].to(bert_device)

        toxic_idx = 0
        if hasattr(bert_model, "config") and hasattr(bert_model.config, "label2id"):
            toxic_idx = bert_model.config.label2id.get("toxic", 0)

        emb_layer = bert_model.get_input_embeddings()
        input_embeds = emb_layer(input_ids).detach()
        baseline = torch.zeros_like(input_embeds)

        steps = 24
        total_grads = torch.zeros_like(input_embeds)
        alphas = torch.linspace(0.0, 1.0, steps, device=bert_device)

        for alpha in alphas:
            interp_embeds = (baseline + alpha * (input_embeds - baseline)).detach().requires_grad_(True)
            bert_model.zero_grad(set_to_none=True)
            logits = bert_model(inputs_embeds=interp_embeds, attention_mask=attention_mask).logits
            toxic_logit = logits[:, toxic_idx] if logits.shape[-1] > 1 else logits.squeeze(-1)
            toxic_logit.sum().backward()
            total_grads += interp_embeds.grad.detach()

        avg_grads = total_grads / steps
        contrib = ((input_embeds - baseline) * avg_grads).sum(dim=-1)[0].detach().cpu().numpy()
        mask = attention_mask[0].detach().cpu().numpy().tolist()

        # Use tokenizer offsets and merge contiguous pieces for readable token spans.
        merged = []
        current_span = None
        current_score = 0.0
        for (start, end), score, m in zip(offsets, contrib, mask):
            if not m or end <= start:
                continue
            piece = text[start:end]
            if not piece.strip():
                continue
            if current_span is None:
                current_span = [start, end]
                current_score = float(score)
            elif start == current_span[1]:
                current_span[1] = end
                current_score += float(score)
            else:
                merged.append((text[current_span[0] : current_span[1]], current_score))
                current_span = [start, end]
                current_score = float(score)
        if current_span is not None:
            merged.append((text[current_span[0] : current_span[1]], current_score))

        # Normalize scores to improve comparability across different inputs.
        if merged:
            abs_sum = sum(abs(s) for _, s in merged)
            if abs_sum > 1e-8:
                merged = [(tok, s / abs_sum) for tok, s in merged]

        if not merged:
            return None

        toxic_sorted = sorted(merged, key=lambda x: x[1], reverse=True)
        safe_sorted = sorted(merged, key=lambda x: x[1])
        top_toxic = [(t, s) for t, s in toxic_sorted if s > 0][:12]
        top_safe = [(t, s) for t, s in safe_sorted if s < 0][:12]
        return top_toxic, top_safe
    except Exception:
        return None

# -------------------- header --------------------
col_h1, _ = st.columns([9, 1])
with col_h1:
    st.markdown("<div style='padding-bottom:6px;'><h1 class='demo-title'>Toxicity Detector Demo</h1></div>", unsafe_allow_html=True)
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
    model_options = [("TF-IDF Baseline", "tfidf")]
    if bert_model is not None:
        model_options.append(("BERT (unitary/toxic-bert)", "bert"))
    if embeddings_clf is not None:
        model_options.append(("Embeddings model", "embeddings"))
    model_label = st.selectbox("Inference model", [opt[0] for opt in model_options], index=0)
    model_mode = dict(model_options)[model_label]
    threshold = st.slider("Action threshold (auto-hide >=)", 0.0, 1.0, 0.85, 0.01)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Predict", key="predict_btn"):
            try:
                text_to_score = st.session_state.get("user_text", "")
                probs = predict_proba([text_to_score], model_mode=model_mode)
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
                    unsafe_allow_html=True)

                if model_mode == "tfidf":
                    expl = explain_tokens(text_to_score)
                    if expl is not None:
                        top_pos, top_neg = expl
                        with st.expander("Show token contributions (TF-IDF only)"):
                            if top_pos:
                                st.subheader("Top toxic signals")
                                for tok, score in top_pos[:8]:
                                    st.write(f"{html.escape(tok)} — {score:.4f}")
                            if top_neg:
                                st.subheader("Top safe signals")
                                for tok, score in top_neg[:8]:
                                    st.write(f"{html.escape(tok)} — {score:.4f}")
                    else:
                        st.info("Token-level contributions unavailable (TF-IDF artifacts missing).")
                elif model_mode == "bert":
                    expl = explain_tokens_bert(text_to_score)
                    if expl is not None:
                        top_pos, top_neg = expl
                        with st.expander("Show token contributions (BERT integrated gradients)"):
                            if top_pos:
                                st.subheader("Top toxic signals")
                                for tok, score in top_pos[:8]:
                                    st.write(f"{html.escape(tok)} — {score:.4f}")
                            if top_neg:
                                st.subheader("Top safe signals")
                                for tok, score in top_neg[:8]:
                                    st.write(f"{html.escape(tok)} — {score:.4f}")
                    else:
                        st.info("BERT token contributions unavailable for this input.")
                else:
                    st.info(f"Token-level contributions are not available for `{model_label}`.")

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

    if bert_model is not None:
        st.success("BERT model ready")
        st.write(f"Model: {BERT_MODEL_NAME}")
        st.caption(f"Source: `{BERT_LOCAL_DIR}`" if BERT_LOCAL_DIR.exists() else "Source: Hugging Face Hub")
    else:
        st.info("BERT not loaded")
        if bert_load_error:
            st.caption(f"BERT load error: {bert_load_error}")

    if embeddings_clf is not None:
        st.info("Embeddings-based model detected")

    st.write(f"Current inference model: {model_label}")

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
            probs = predict_proba(texts, model_mode=model_mode)
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
    Built as a demo for a toxicity detector. This app uses a TF-IDF + Logistic Regression baseline saved in <code>models/</code>.
    <br><br>
    Limitations: may produce false positives and false negatives — do NOT use this for automated moderation without a human review flow.
    </div>
    """,
    unsafe_allow_html=True,
)
