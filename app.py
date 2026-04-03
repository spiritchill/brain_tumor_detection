import streamlit as st
import keras
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:       #0a0e1a;
    --surface:  #111827;
    --surface2: #1a2234;
    --accent:   #00d4ff;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --border:   #1e293b;
    --glioma:   #ff6b6b;
    --mening:   #fbbf24;
    --pitu:     #a78bfa;
    --none:     #34d399;
}

html, body, .stApp { background-color: var(--bg) !important; color: var(--text) !important;
                     font-family: 'DM Sans', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }

/* Sidebar */
section[data-testid="stSidebar"] { background: var(--surface) !important;
                                    border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid var(--border); border-radius: 16px;
    padding: 2.5rem 3rem; margin-bottom: 2rem; position: relative; overflow: hidden;
}
.hero::before {
    content:''; position:absolute; top:-50%; left:-50%; width:200%; height:200%;
    background: radial-gradient(circle at 30% 50%, rgba(0,212,255,.07) 0%, transparent 50%),
                radial-gradient(circle at 70% 50%, rgba(167,139,250,.07) 0%, transparent 50%);
}
.hero h1 {
    font-family:'Space Mono',monospace !important; font-size:2.4rem !important;
    font-weight:700 !important;
    background:linear-gradient(90deg,var(--accent),#a78bfa);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin:0 0 .5rem !important;
}
.hero p { color:var(--muted); font-size:1rem; margin:0; }

/* Section title */
.sec { font-family:'Space Mono',monospace; font-size:.72rem; text-transform:uppercase;
       letter-spacing:.15em; color:var(--accent); margin-bottom:.8rem;
       padding-bottom:.4rem; border-bottom:1px solid var(--border); }

/* Tumor cards grid */
.tumor-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:.9rem; margin-bottom:1.8rem; }
.tcard { background:var(--surface); border-radius:12px; padding:1.1rem 1.3rem;
         border-left:3px solid; border-top:1px solid var(--border);
         border-right:1px solid var(--border); border-bottom:1px solid var(--border); }
.tcard h4 { font-family:'Space Mono',monospace; font-size:.8rem; margin:0 0 .35rem;
            letter-spacing:.05em; text-transform:uppercase; }
.tcard p  { color:var(--muted); font-size:.8rem; margin:0; line-height:1.5; }
.tcard .tx { margin-top:.3rem; font-size:.78rem; }
.tcard .tx b { color:#94a3b8; }

.cg { border-left-color:var(--glioma); } .cg h4 { color:var(--glioma); }
.cm { border-left-color:var(--mening); } .cm h4 { color:var(--mening); }
.cp { border-left-color:var(--pitu);   } .cp h4 { color:var(--pitu);   }
.cn { border-left-color:var(--none);   } .cn h4 { color:var(--none);   }

/* Result badge */
.badge { display:inline-block; padding:.45rem 1.3rem; border-radius:999px;
         font-family:'Space Mono',monospace; font-size:.95rem; font-weight:700;
         letter-spacing:.08em; text-transform:uppercase; margin:.6rem 0 1rem; }

/* Metric row */
.mrow { display:flex; gap:.9rem; margin-bottom:1.4rem; }
.mbox { flex:1; background:var(--surface); border-radius:10px;
        padding:.9rem; border:1px solid var(--border); text-align:center; }
.mbox .val { font-family:'Space Mono',monospace; font-size:1.5rem; font-weight:700; color:var(--accent); }
.mbox .lbl { font-size:.72rem; color:var(--muted); margin-top:.2rem; }

/* Upload hint */
.upload-hint { background:var(--surface2); border:1px dashed var(--border);
               border-radius:12px; padding:1.5rem; text-align:center;
               color:var(--muted); font-size:.88rem; margin-top:.5rem; }
.conf-lbl { font-family:'Space Mono',monospace; font-size:.72rem; color:var(--muted);
            text-transform:uppercase; letter-spacing:.1em; margin-bottom:.3rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Constants  ← kept exactly from YOUR code
# ─────────────────────────────────────────────
class_names  = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']   # your variable name
CLASS_COLORS = ['#ff6b6b', '#fbbf24', '#34d399', '#a78bfa']
IMG_SIZE     = (128, 128)          # your original size

TUMOR_INFO = {
    'Glioma':     {'cls':'cg', 'desc':'Arises from glial cells. One of the most aggressive primary brain tumors.',    'tx':'Surgery, radiation, temozolomide chemotherapy.'},
    'Meningioma': {'cls':'cm', 'desc':'Grows from meninges surrounding the brain. Usually benign and slow-growing.',  'tx':'Observation, surgery, or stereotactic radiosurgery.'},
    'No Tumor':   {'cls':'cn', 'desc':'No evidence of tumor. Normal brain tissue observed in the MRI scan.',          'tx':'No treatment needed. Routine follow-up recommended.'},
    'Pituitary':  {'cls':'cp', 'desc':'Forms in the pituitary gland. Often affects hormone levels and vision.',       'tx':'Medication, transsphenoidal surgery, or radiation.'},
}

# ─────────────────────────────────────────────
#  Load model (cached)  ← your exact path
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("brain_tumor_model.h5")
    except Exception as e:
        st.error(f"⚠️ Could not load model: {e}")
        return None

# ─────────────────────────────────────────────
#  Confidence bar chart (new addition)
# ─────────────────────────────────────────────
def confidence_chart(probs):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=class_names,
        y=(probs * 100).tolist(),
        marker=dict(color=CLASS_COLORS, line=dict(width=0)),
        text=[f"{p:.1f}%" for p in probs * 100],
        textposition="outside",
        textfont=dict(family="Space Mono, monospace", size=11, color="#e2e8f0"),
        width=0.5,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#94a3b8"),
        yaxis=dict(title="Confidence (%)", range=[0, 118], gridcolor="#1e293b",
                   tickfont=dict(size=10), zeroline=False),
        xaxis=dict(tickfont=dict(size=11)),
        margin=dict(l=10, r=10, t=20, b=10),
        showlegend=False, height=290,
    )
    return fig

# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sec">⚙ Settings</div>', unsafe_allow_html=True)
    threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.50, 0.01,
                          help="Show a warning if confidence is below this value.")
    show_raw  = st.checkbox("Show raw probabilities", False)
    st.markdown("---")
    st.markdown('<div class="sec">ℹ About</div>', unsafe_allow_html=True)
    st.markdown("""
<p style='color:#64748b;font-size:.82rem;line-height:1.6'>
Model trained on the <b style='color:#94a3b8'>Brain Tumor MRI Dataset</b> (Kaggle)
using a custom CNN — TensorFlow / Keras.<br><br>
⚠️ <b style='color:#fbbf24'>Not for clinical use.</b> Educational only.
</p>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Hero
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🧠 Brain Tumor Detection</h1>
  <p>Upload an MRI scan and the model will classify it into one of four categories —
     Glioma, Meningioma, Pituitary, or No Tumor — with a full confidence breakdown.</p>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Tumor info cards
# ─────────────────────────────────────────────
st.markdown('<div class="sec">📋 Tumor Types</div>', unsafe_allow_html=True)
st.markdown('<div class="tumor-grid">', unsafe_allow_html=True)
for name, info in TUMOR_INFO.items():
    st.markdown(f"""
<div class="tcard {info['cls']}">
  <h4>{name}</h4>
  <p>{info['desc']}</p>
  <p class="tx"><b>Treatment:</b> {info['tx']}</p>
</div>""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Main layout — upload + results
# ─────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.4], gap="large")

with col_left:
    st.markdown('<div class="sec">📤 Upload MRI Scan</div>', unsafe_allow_html=True)

    # ── YOUR original file uploader ──
    uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"],
                                     label_visibility="collapsed")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)        # ← YOUR variable name
        if image.mode != "RGB":                  # handle grayscale MRIs
            image = image.convert("RGB")
        st.image(image, caption="Uploaded MRI Scan", use_container_width=True)
    else:
        st.markdown("""
<div class="upload-hint">
  🖼️ Drag & drop or click to upload<br>
  <span style='font-size:.78rem'>Supports JPG, JPEG, PNG</span>
</div>""", unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="sec">🔬 Analysis Results</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        model = keras.models.load_model("brain_tumor_model.h5", compile=False)
        if model:
            with st.spinner("Analysing MRI scan…"):

                # ── YOUR exact preprocessing logic ──
                img = np.array(image)
                img = cv2.resize(img, IMG_SIZE)
                img = img / 255.0
                img = np.expand_dims(img, axis=0)

                # ── YOUR exact prediction logic ──
                prediction       = model.predict(img)
                predicted_class  = class_names[np.argmax(prediction)]
                confidence       = np.max(prediction) * 100
                all_probs        = prediction[0]          # all 4 class scores

            idx   = class_names.index(predicted_class)
            color = CLASS_COLORS[idx]

            # ── Result badge ──
            st.markdown(f"""
<p class="conf-lbl">Predicted Class</p>
<span class="badge" style="background:{color}22;color:{color};border:1.5px solid {color}55;">
  {predicted_class}
</span>""", unsafe_allow_html=True)

            # ── YOUR original success/error logic preserved ──
            if predicted_class == "No Tumor":
                st.success(f"✅ {predicted_class} detected — Confidence: {confidence:.2f}%")
            else:
                st.error(f"⚠️ {predicted_class} detected — Confidence: {confidence:.2f}%")

            # ── Extra: low-confidence warning ──
            if confidence / 100 < threshold:
                st.warning(f"🔔 Confidence is below your threshold ({threshold:.0%}). Consider specialist review.")

            # ── Metrics ──
            st.markdown(f"""
<div class="mrow">
  <div class="mbox">
    <div class="val">{confidence:.1f}%</div>
    <div class="lbl">Confidence</div>
  </div>
  <div class="mbox">
    <div class="val" style="color:{color};font-size:1.05rem">{predicted_class}</div>
    <div class="lbl">Diagnosis</div>
  </div>
</div>""", unsafe_allow_html=True)

            # ── Info card for predicted class ──
            info = TUMOR_INFO[predicted_class]
            st.markdown(f"""
<div class="tcard {info['cls']}" style="margin-bottom:1.1rem">
  <h4>About: {predicted_class}</h4>
  <p>{info['desc']}</p>
  <p class="tx"><b>Treatment:</b> {info['tx']}</p>
</div>""", unsafe_allow_html=True)

            # ── Confidence bar chart ── (new feature)
            st.markdown('<div class="conf-lbl">Confidence Distribution — All Classes</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(confidence_chart(all_probs), use_container_width=True)

            # ── Raw probabilities toggle ──
            if show_raw:
                st.markdown('<div class="sec" style="margin-top:.8rem">Raw Probabilities</div>',
                            unsafe_allow_html=True)
                for cls, prob in zip(class_names, all_probs):
                    st.markdown(f"**{cls}** — `{prob:.6f}`")

    else:
        st.markdown("""
<div style='color:#475569;font-size:.88rem;margin-top:4rem;text-align:center'>
  👈 Upload an MRI image to see results here.
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Disclaimer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style='color:#475569;font-size:.76rem;text-align:center'>
⚠️ <b>Disclaimer:</b> This tool is for educational and research purposes only.
It is not a substitute for professional medical advice, diagnosis, or treatment.
Always consult a qualified healthcare provider for medical decisions.
</p>""", unsafe_allow_html=True)
