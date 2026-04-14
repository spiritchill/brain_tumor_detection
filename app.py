import streamlit as st
import tensorflow as tf
import keras
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.platypus import Image as RLImage
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import io
import datetime
import tempfile

# ─────────────────────────────────────────────
#  Page Config
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

section[data-testid="stSidebar"] { background: var(--surface) !important;
                                    border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] * { color: var(--text) !important; }

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

.sec { font-family:'Space Mono',monospace; font-size:.72rem; text-transform:uppercase;
       letter-spacing:.15em; color:var(--accent); margin-bottom:.8rem;
       padding-bottom:.4rem; border-bottom:1px solid var(--border); }

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

.badge { display:inline-block; padding:.45rem 1.3rem; border-radius:999px;
         font-family:'Space Mono',monospace; font-size:.95rem; font-weight:700;
         letter-spacing:.08em; text-transform:uppercase; margin:.6rem 0 1rem; }

.mrow { display:flex; gap:.9rem; margin-bottom:1.4rem; }
.mbox { flex:1; background:var(--surface); border-radius:10px;
        padding:.9rem; border:1px solid var(--border); text-align:center; }
.mbox .val { font-family:'Space Mono',monospace; font-size:1.5rem; font-weight:700; color:var(--accent); }
.mbox .lbl { font-size:.72rem; color:var(--muted); margin-top:.2rem; }

.upload-hint { background:var(--surface2); border:1px dashed var(--border);
               border-radius:12px; padding:1.5rem; text-align:center;
               color:var(--muted); font-size:.88rem; margin-top:.5rem; }
.conf-lbl { font-family:'Space Mono',monospace; font-size:.72rem; color:var(--muted);
            text-transform:uppercase; letter-spacing:.1em; margin-bottom:.3rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
class_names  = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
CLASS_COLORS = ['#ff6b6b', '#fbbf24', '#34d399', '#a78bfa']
IMG_SIZE     = (128, 128)

TUMOR_INFO = {
    'Glioma':     {
        'cls':'cg',
        'desc':'Arises from glial cells. One of the most aggressive primary brain tumors.',
        'tx':'Surgery, radiation, temozolomide chemotherapy.',
        'urgency': 'HIGH',
        'icd': 'C71.9',
        'followup': 'Immediate oncology referral. MRI with contrast within 48 hours.',
        'prognosis': 'Variable; depends on grade. Median survival 14-16 months (GBM).',
    },
    'Meningioma': {
        'cls':'cm',
        'desc':'Grows from meninges surrounding the brain. Usually benign and slow-growing.',
        'tx':'Observation, surgery, or stereotactic radiosurgery.',
        'urgency': 'MODERATE',
        'icd': 'D32.9',
        'followup': 'Neurosurgery referral within 2 weeks. Repeat MRI in 3 months.',
        'prognosis': 'Generally favorable. ~95% 5-year survival for benign cases.',
    },
    'No Tumor':   {
        'cls':'cn',
        'desc':'No evidence of tumor. Normal brain tissue observed in the MRI scan.',
        'tx':'No treatment needed. Routine follow-up recommended.',
        'urgency': 'NONE',
        'icd': 'Z03.89',
        'followup': 'No immediate action required. Annual screening if symptomatic.',
        'prognosis': 'Excellent. No malignant findings detected.',
    },
    'Pituitary':  {
        'cls':'cp',
        'desc':'Forms in the pituitary gland. Often affects hormone levels and vision.',
        'tx':'Medication, transsphenoidal surgery, or radiation.',
        'urgency': 'MODERATE',
        'icd': 'D35.2',
        'followup': 'Endocrinology + ophthalmology referral. Hormone panel tests.',
        'prognosis': 'Good with treatment. Most pituitary adenomas are benign.',
    },
}

URGENCY_COLORS = {'HIGH': '#ff6b6b', 'MODERATE': '#fbbf24', 'NONE': '#34d399'}

# ─────────────────────────────────────────────
#  Load model
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model("brain_tumor_model.keras", compile=False)
        return model
    except Exception as e:
        st.error(f"⚠️ Could not load model: {e}")
        return None

# ─────────────────────────────────────────────
#  Confidence bar chart
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
#  PDF Report Generator
# ─────────────────────────────────────────────
def generate_pdf_report(
    patient_name, patient_age, patient_id, scan_date,
    predicted_class, confidence, all_probs,
    original_img_pil,
    doctor_notes=""
):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )

    info    = TUMOR_INFO[predicted_class]
    urgency = info['urgency']
    u_color = colors.HexColor(URGENCY_COLORS[urgency])

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('TitleStyle', parent=styles['Title'],
        fontSize=20, textColor=colors.HexColor('#1a1a2e'),
        spaceAfter=4, fontName='Helvetica-Bold', alignment=TA_CENTER)
    subtitle_style = ParagraphStyle('SubStyle', parent=styles['Normal'],
        fontSize=9, textColor=colors.HexColor('#64748b'),
        alignment=TA_CENTER, spaceAfter=16)
    section_style = ParagraphStyle('SectionStyle', parent=styles['Normal'],
        fontSize=10, textColor=colors.HexColor('#1e40af'),
        fontName='Helvetica-Bold', spaceBefore=12, spaceAfter=6)
    body_style = ParagraphStyle('BodyStyle', parent=styles['Normal'],
        fontSize=9, textColor=colors.HexColor('#1e293b'),
        leading=14, alignment=TA_JUSTIFY)
    label_style = ParagraphStyle('LabelStyle', parent=styles['Normal'],
        fontSize=8, textColor=colors.HexColor('#64748b'), fontName='Helvetica-Bold')
    value_style = ParagraphStyle('ValueStyle', parent=styles['Normal'],
        fontSize=9, textColor=colors.HexColor('#0f172a'))
    warning_style = ParagraphStyle('WarnStyle', parent=styles['Normal'],
        fontSize=8, textColor=colors.HexColor('#92400e'),
        leading=12, alignment=TA_CENTER)
    caption_style = ParagraphStyle('CaptionStyle', parent=styles['Normal'],
        fontSize=7.5, textColor=colors.HexColor('#64748b'), alignment=TA_CENTER)

    elements = []

    # ── Header ──
    header_table = Table([[Paragraph("🧠 BRAIN TUMOR DETECTION REPORT", title_style)]], colWidths=[17*cm])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f0f9ff')),
        ('TOPPADDING',    (0,0), (-1,-1), 14),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING',   (0,0), (-1,-1), 16),
        ('RIGHTPADDING',  (0,0), (-1,-1), 16),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#bfdbfe')),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        f"Generated on {scan_date}  •  AI-Assisted Diagnostic Tool  •  For Clinical Review Only",
        subtitle_style))

    # ── Urgency banner ──
    urg_bg     = {'HIGH': '#fef2f2', 'MODERATE': '#fffbeb', 'NONE': '#f0fdf4'}[urgency]
    urg_border = {'HIGH': '#fca5a5', 'MODERATE': '#fcd34d', 'NONE': '#86efac'}[urgency]
    urg_text   = {
        'HIGH':     '🔴  HIGH URGENCY — Immediate medical attention required',
        'MODERATE': '🟡  MODERATE URGENCY — Medical review recommended within 2 weeks',
        'NONE':     '🟢  NO TUMOR DETECTED — Routine follow-up advised',
    }[urgency]
    urg_style = ParagraphStyle('UrgStyle', parent=styles['Normal'],
        fontSize=9, textColor=u_color, fontName='Helvetica-Bold', alignment=TA_CENTER)
    urg_table = Table([[Paragraph(urg_text, urg_style)]], colWidths=[17*cm])
    urg_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(urg_bg)),
        ('BOX', (0,0), (-1,-1), 1.5, colors.HexColor(urg_border)),
        ('TOPPADDING',    (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    elements.append(urg_table)
    elements.append(Spacer(1, 12))

    # ── Patient info ──
    elements.append(Paragraph("PATIENT INFORMATION", section_style))
    pat_data = [
        [Paragraph("Patient Name", label_style), Paragraph(patient_name or "N/A", value_style),
         Paragraph("Patient ID",   label_style), Paragraph(patient_id or "N/A",   value_style)],
        [Paragraph("Age",          label_style), Paragraph(f"{patient_age} years" if patient_age else "N/A", value_style),
         Paragraph("Scan Date",    label_style), Paragraph(scan_date, value_style)],
        [Paragraph("Modality",     label_style), Paragraph("MRI Brain", value_style),
         Paragraph("Report Type",  label_style), Paragraph("AI-Assisted Analysis", value_style)],
    ]
    pat_table = Table(pat_data, colWidths=[3.5*cm, 5*cm, 3.5*cm, 5*cm])
    pat_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX',       (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('INNERGRID', (0,0), (-1,-1), 0.25, colors.HexColor('#e2e8f0')),
        ('TOPPADDING',    (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('LEFTPADDING',   (0,0), (-1,-1), 10),
        ('RIGHTPADDING',  (0,0), (-1,-1), 10),
    ]))
    elements.append(pat_table)
    elements.append(Spacer(1, 12))

    # ── Diagnosis ──
    elements.append(Paragraph("DIAGNOSIS SUMMARY", section_style))
    diag_color_map  = {'Glioma':'#fff1f2','Meningioma':'#fffbeb','No Tumor':'#f0fdf4','Pituitary':'#f5f3ff'}
    diag_border_map = {'Glioma':'#ff6b6b','Meningioma':'#fbbf24','No Tumor':'#34d399','Pituitary':'#a78bfa'}
    result_style = ParagraphStyle('ResStyle', parent=styles['Normal'],
        fontSize=15, textColor=colors.HexColor(diag_border_map[predicted_class]),
        fontName='Helvetica-Bold', alignment=TA_CENTER)
    conf_style = ParagraphStyle('ConfStyle', parent=styles['Normal'],
        fontSize=9, textColor=colors.HexColor('#64748b'), alignment=TA_CENTER)
    diag_table = Table([[
        Paragraph(f"▶  {predicted_class.upper()}", result_style),
        Paragraph(f"Confidence Score: {confidence:.2f}%\nICD-10: {info['icd']}", conf_style),
    ]], colWidths=[9*cm, 8*cm])
    diag_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), colors.HexColor(diag_color_map[predicted_class])),
        ('BACKGROUND', (1,0), (1,0), colors.HexColor('#f8fafc')),
        ('BOX',       (0,0), (-1,-1), 1, colors.HexColor(diag_border_map[predicted_class])),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING',    (0,0), (-1,-1), 14),
        ('BOTTOMPADDING', (0,0), (-1,-1), 14),
        ('LEFTPADDING',   (0,0), (-1,-1), 12),
        ('RIGHTPADDING',  (0,0), (-1,-1), 12),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    elements.append(diag_table)
    elements.append(Spacer(1, 10))

    clin_data = [
        [Paragraph("Description", label_style), Paragraph(info['desc'],      body_style)],
        [Paragraph("Treatment",   label_style), Paragraph(info['tx'],        body_style)],
        [Paragraph("Follow-up",   label_style), Paragraph(info['followup'],  body_style)],
        [Paragraph("Prognosis",   label_style), Paragraph(info['prognosis'], body_style)],
    ]
    clin_table = Table(clin_data, colWidths=[3*cm, 14*cm])
    clin_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#f1f5f9')),
        ('BOX',       (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('INNERGRID', (0,0), (-1,-1), 0.25, colors.HexColor('#e2e8f0')),
        ('TOPPADDING',    (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('LEFTPADDING',   (0,0), (-1,-1), 10),
        ('RIGHTPADDING',  (0,0), (-1,-1), 10),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    elements.append(clin_table)
    elements.append(Spacer(1, 12))

    # ── Confidence table ──
    elements.append(Paragraph("CONFIDENCE DISTRIBUTION — ALL CLASSES", section_style))
    conf_header = [Paragraph(h, ParagraphStyle('CH', parent=styles['Normal'],
        fontSize=8, textColor=colors.white, fontName='Helvetica-Bold', alignment=TA_CENTER))
        for h in ["Class", "Confidence (%)", "Status"]]
    conf_rows = [conf_header]
    for cls, prob in zip(class_names, all_probs):
        is_pred = cls == predicted_class
        bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
        conf_rows.append([
            Paragraph(f"{'★ ' if is_pred else ''}{cls}", ParagraphStyle('CC', parent=styles['Normal'],
                fontSize=8.5,
                fontName='Helvetica-Bold' if is_pred else 'Helvetica',
                textColor=colors.HexColor(diag_border_map.get(cls, '#1e293b')))),
            Paragraph(f"{prob*100:.2f}%  {bar}", ParagraphStyle('CB', parent=styles['Normal'],
                fontSize=7.5, fontName='Courier', textColor=colors.HexColor('#334155'))),
            Paragraph("✓ PREDICTED" if is_pred else "—", ParagraphStyle('CS', parent=styles['Normal'],
                fontSize=8,
                fontName='Helvetica-Bold' if is_pred else 'Helvetica',
                textColor=colors.HexColor('#16a34a') if is_pred else colors.HexColor('#94a3b8'),
                alignment=TA_CENTER)),
        ])
    conf_table = Table(conf_rows, colWidths=[3.5*cm, 10*cm, 3.5*cm])
    conf_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e293b')),
        ('BOX',       (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('INNERGRID', (0,0), (-1,-1), 0.25, colors.HexColor('#e2e8f0')),
        ('TOPPADDING',    (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('LEFTPADDING',   (0,0), (-1,-1), 10),
        ('RIGHTPADDING',  (0,0), (-1,-1), 10),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    elements.append(conf_table)
    elements.append(Spacer(1, 12))

    # ── MRI Image ──
    elements.append(Paragraph("MRI SCAN", section_style))
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    original_img_pil.resize((200, 200)).save(tmp.name)
    tmp.close()
    img_table = Table([
        [RLImage(tmp.name, width=5.5*cm, height=5.5*cm)],
        [Paragraph("Original MRI Scan", caption_style)],
    ], colWidths=[17*cm])
    img_table.setStyle(TableStyle([
        ('ALIGN',   (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',  (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING',    (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('BOX',        (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
    ]))
    elements.append(img_table)
    elements.append(Spacer(1, 12))

    # ── Doctor notes ──
    if doctor_notes and doctor_notes.strip():
        elements.append(Paragraph("DOCTOR'S NOTES & OBSERVATIONS", section_style))
        notes_table = Table([[Paragraph(doctor_notes, body_style)]], colWidths=[17*cm])
        notes_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#fffbeb')),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#fcd34d')),
            ('TOPPADDING',    (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
            ('LEFTPADDING',   (0,0), (-1,-1), 12),
            ('RIGHTPADDING',  (0,0), (-1,-1), 12),
        ]))
        elements.append(notes_table)
        elements.append(Spacer(1, 12))

    # ── Disclaimer ──
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#e2e8f0')))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(
        "⚠️ DISCLAIMER: This report is generated by an AI-assisted diagnostic tool and is intended "
        "for educational and research purposes ONLY. It is NOT a substitute for professional medical "
        "advice, diagnosis, or treatment. All findings must be reviewed and confirmed by a qualified "
        "radiologist or neurosurgeon before any clinical decisions are made.",
        warning_style))

    doc.build(elements)
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sec">⚙ Settings</div>', unsafe_allow_html=True)
    threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.50, 0.01,
                          help="Show a warning if confidence is below this value.")
    show_raw  = st.checkbox("Show raw probabilities", False)

    st.markdown("---")
    st.markdown('<div class="sec">👤 Patient Details</div>', unsafe_allow_html=True)
    patient_name = st.text_input("Patient Name", placeholder="e.g. John Doe")
    patient_age  = st.number_input("Age", min_value=1, max_value=120, value=35)
    patient_id   = st.text_input("Patient ID", placeholder="e.g. PT-2024-001")
    doctor_notes = st.text_area("Doctor's Notes", placeholder="Add clinical observations here...", height=100)

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
     Glioma, Meningioma, Pituitary, or No Tumor — with a full confidence breakdown
     and a downloadable clinical PDF report.</p>
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
#  Main layout
# ─────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.5], gap="large")

with col_left:
    st.markdown('<div class="sec">📤 Upload MRI Scan</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"],
                                     label_visibility="collapsed")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        if image.mode != "RGB":
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
        model = load_model()
        if model:
            with st.spinner("Analysing MRI scan…"):
                img_np         = np.array(image)
                img_resized    = cv2.resize(img_np, IMG_SIZE)
                img_normalized = img_resized / 255.0
                img_input      = np.expand_dims(img_normalized, axis=0)

                prediction = model.predict(img_input, verbose=0)
                all_probs  = prediction[0]

                # Apply softmax if outputs don't sum to ~1
                if not (0.98 <= float(all_probs.sum()) <= 1.02):
                    all_probs = tf.nn.softmax(all_probs).numpy()

                predicted_class = class_names[int(np.argmax(all_probs))]
                confidence      = float(np.max(all_probs)) * 100

            idx   = class_names.index(predicted_class)
            color = CLASS_COLORS[idx]
            info  = TUMOR_INFO[predicted_class]

            # Result badge
            st.markdown(f"""
<p class="conf-lbl">Predicted Class</p>
<span class="badge" style="background:{color}22;color:{color};border:1.5px solid {color}55;">
  {predicted_class}
</span>""", unsafe_allow_html=True)

            if predicted_class == "No Tumor":
                st.success(f"✅ {predicted_class} detected — Confidence: {confidence:.2f}%")
            else:
                st.error(f"⚠️ {predicted_class} detected — Confidence: {confidence:.2f}%")

            if confidence / 100 < threshold:
                st.warning(f"🔔 Confidence below threshold ({threshold:.0%}). Consider specialist review.")

            # Metrics
            urgency   = info['urgency']
            urg_color = URGENCY_COLORS[urgency]
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
  <div class="mbox">
    <div class="val" style="color:{urg_color};font-size:.95rem">{urgency}</div>
    <div class="lbl">Urgency</div>
  </div>
</div>""", unsafe_allow_html=True)

            # Clinical info card
            st.markdown(f"""
<div class="tcard {info['cls']}" style="margin-bottom:1.1rem">
  <h4>About: {predicted_class}  &nbsp;·&nbsp; ICD-10: {info['icd']}</h4>
  <p>{info['desc']}</p>
  <p class="tx"><b>Treatment:</b> {info['tx']}</p>
  <p class="tx"><b>Follow-up:</b> {info['followup']}</p>
  <p class="tx"><b>Prognosis:</b> {info['prognosis']}</p>
</div>""", unsafe_allow_html=True)

            # Confidence chart
            st.markdown('<div class="conf-lbl" style="margin-top:1rem">Confidence Distribution — All Classes</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(confidence_chart(all_probs), use_container_width=True)

            if show_raw:
                st.markdown('<div class="sec" style="margin-top:.8rem">Raw Probabilities</div>',
                            unsafe_allow_html=True)
                for cls, prob in zip(class_names, all_probs):
                    st.markdown(f"**{cls}** — `{prob:.6f}`")

            # ── PDF Report ──
            st.markdown("---")
            st.markdown('<div class="sec">📄 Generate Clinical Report</div>', unsafe_allow_html=True)
            scan_date = datetime.datetime.now().strftime("%d %B %Y, %H:%M")

            if st.button("🖨️ Generate PDF Report", use_container_width=True):
                with st.spinner("Building your clinical report…"):
                    pdf_buf = generate_pdf_report(
                        patient_name=patient_name or "Unknown",
                        patient_age=patient_age,
                        patient_id=patient_id or "N/A",
                        scan_date=scan_date,
                        predicted_class=predicted_class,
                        confidence=confidence,
                        all_probs=all_probs,
                        original_img_pil=image,
                        doctor_notes=doctor_notes,
                    )
                fname = f"brain_tumor_report_{(patient_name or 'patient').replace(' ','_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_buf,
                    file_name=fname,
                    mime="application/pdf",
                    use_container_width=True,
                )
                st.success("✅ Report ready! Click above to download.")

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
