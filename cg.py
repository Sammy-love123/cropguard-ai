# ============================================================
# CROP DISEASE PREDICTION SYSTEM — MOBILE-STYLE STREAMLIT APP
# Bowen University, Iwo
# Student: Ojugbeli Ogechi Esther
# Supervisor: Miss Busolami Oluwadamilare
#
# HOW TO RUN:
#   pip install streamlit tensorflow pillow numpy pandas matplotlib
#   streamlit run streamlit_app.py
#
# FILES NEEDED IN SAME FOLDER:
#   - BEST_MobileNetV2.h5
#   - class_info.json
# ============================================================

import streamlit as st
import tensorflow as tf
import numpy as np
import json, os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title = "CropGuard AI",
    page_icon  = "🌿",
    layout     = "centered",
    initial_sidebar_state = "collapsed"
)

# ── Demo users (username: password) ──────────────────────────
USERS = {
    "farmer"   : "farm123",
    "admin"    : "admin123",
    "student"  : "bowen2025",
    "ogechi"   : "cropguard",
}

# ── Disease database ──────────────────────────────────────────
DISEASE_DB = {
    'Cassava Bacterial Blight': {
        'crop':'Cassava','severity':'High','color':'#e74c3c',
        'icon':'🔴',
        'symptoms':['Angular water-soaked spots on leaves','Yellow halo around dark spots','Stem wilting and dieback','Gummy exudate on stems'],
        'treatment':['Remove and destroy all infected parts','Apply copper-based bactericide','Use certified disease-free planting material','Improve field drainage'],
        'description':'Serious bacterial disease caused by Xanthomonas axonopodis. Spreads through infected planting material and water splash.'
    },
    'Cassava Green Mite': {
        'crop':'Cassava','severity':'Medium','color':'#e67e22',
        'icon':'🟠',
        'symptoms':['Green mottling on young leaves','Leaf distortion and stunted growth','Reduced leaf size','Tiny mites on leaf undersides'],
        'treatment':['Apply acaricide (abamectin)','Introduce biological control agents','Remove heavily infested leaves','Use resistant cassava varieties'],
        'description':'Caused by Mononychellus tanajoa mites that feed on young leaves and growing tips.'
    },
    'Cassava Healthy': {
        'crop':'Cassava','severity':'None','color':'#2ecc71',
        'icon':'🟢',
        'symptoms':['Deep green uniform leaf color','Normal leaf shape and size','No spots or lesions','Healthy stem growth'],
        'treatment':['Continue regular monitoring','Maintain proper spacing','Apply balanced NPK fertilizer','Practice crop rotation'],
        'description':'Your cassava plant appears healthy! Maintain good agricultural practices to keep it this way.'
    },
    'Cassava Mosaic': {
        'crop':'Cassava','severity':'High','color':'#e74c3c',
        'icon':'🔴',
        'symptoms':['Yellow and green mosaic pattern','Leaf distortion and wrinkling','Stunted plant growth','Reduced tuber yield'],
        'treatment':['Use CMD-resistant varieties','Control whitefly vectors','Remove infected plants (roguing)','Use certified planting material'],
        'description':'Caused by Cassava Mosaic Virus spread by whiteflies. Most economically damaging cassava disease in Nigeria.'
    },
    'Maize Fall Armyworm': {
        'crop':'Maize','severity':'High','color':'#e74c3c',
        'icon':'🔴',
        'symptoms':['Ragged holes in leaves','Frass in leaf whorls','Damage to growing point','Visible caterpillars with Y-mark on head'],
        'treatment':['Apply emamectin benzoate insecticide','Use Bacillus thuringiensis (Bt)','Apply neem-based products','Scout fields twice per week'],
        'description':'Spodoptera frugiperda can destroy entire maize fields within days if not controlled early.'
    },
    'Maize Grasshopper': {
        'crop':'Maize','severity':'Medium','color':'#e67e22',
        'icon':'🟠',
        'symptoms':['Irregular chewing on leaf edges','Skeletonized leaves in severe cases','Visible grasshoppers on plants','Worse during dry seasons'],
        'treatment':['Apply lambda-cyhalothrin insecticide','Use neem extract as botanical control','Remove weeds around field edges','Plant early to avoid peak season'],
        'description':'Seasonal pests causing significant leaf damage especially during dry spells in Northern Nigeria.'
    },
    'Maize Healthy': {
        'crop':'Maize','severity':'None','color':'#2ecc71',
        'icon':'🟢',
        'symptoms':['Vibrant green leaf color','No spots or holes visible','Strong upright plant','Healthy ear development'],
        'treatment':['Continue weekly field scouting','Apply nitrogen top-dressing','Maintain proper plant spacing','Monitor for pest pressure'],
        'description':'Your maize plant appears healthy. Continue with good crop management practices.'
    },
    'Maize Leaf Beetle': {
        'crop':'Maize','severity':'Medium','color':'#e67e22',
        'icon':'🟠',
        'symptoms':['Long parallel strips of feeding damage','Window-pane effect on leaves','Visible yellow-brown beetles','White streak feeding marks'],
        'treatment':['Apply cypermethrin contact insecticide','Hand-collect beetles on small farms','Use neem seed kernel extract','Practice crop rotation'],
        'description':'Leaf beetles feed on leaf tissue creating characteristic patterns that reduce photosynthesis.'
    },
    'Maize Leaf Blight': {
        'crop':'Maize','severity':'High','color':'#e74c3c',
        'icon':'🔴',
        'symptoms':['Large tan cigar-shaped lesions','Lesions with dark brown borders','Disease spreads from lower leaves','Severe cases cause full leaf death'],
        'treatment':['Apply mancozeb or propiconazole fungicide','Use resistant hybrid varieties','Avoid overhead irrigation','Remove infected crop residues'],
        'description':'Northern or Southern Leaf Blight caused by fungal pathogens. Favored by warm, humid conditions.'
    },
    'Maize Leaf Spot': {
        'crop':'Maize','severity':'Medium','color':'#e67e22',
        'icon':'🟠',
        'symptoms':['Small circular spots on leaves','Spots with yellow or brown halo','Multiple spots may merge','Affects all growth stages'],
        'treatment':['Apply azoxystrobin fungicide','Improve air circulation','Avoid excess nitrogen fertilizer','Use certified disease-free seed'],
        'description':'Gray leaf spot caused by Cercospora zeae-maydis, favored by high humidity and poor airflow.'
    },
    'Maize Streak Virus': {
        'crop':'Maize','severity':'High','color':'#e74c3c',
        'icon':'🔴',
        'symptoms':['Narrow yellow streaks along leaf veins','Stunted plant growth','White streaking on young leaves','Severe yellowing of entire plant'],
        'treatment':['Use streak-resistant maize varieties','Control leafhopper vectors','Remove infected plants early','Plant early to escape peak season'],
        'description':'Maize Streak Virus transmitted by leafhopper Cicadulina mbila. Most damaging maize disease in sub-Saharan Africa.'
    },
    'Yam Anthracnose': {
        'crop':'Yam','severity':'High','color':'#e74c3c',
        'icon':'🔴',
        'symptoms':['Dark brown circular spots on leaves','Spots with yellow halo','Stem dieback from tips','Dark sunken lesions on tubers'],
        'treatment':['Apply mancozeb fungicide every 2 weeks','Remove and destroy infected parts','Use disease-free seed yams','Improve field drainage'],
        'description':'Caused by Colletotrichum gloeosporioides. One of the most serious yam diseases in Nigeria.'
    },
    'Yam Healthy': {
        'crop':'Yam','severity':'None','color':'#2ecc71',
        'icon':'🟢',
        'symptoms':['Dark green healthy leaf color','No spots or lesions','Normal vine growth','Healthy tuber development'],
        'treatment':['Continue staking vines','Apply fertilizer at 6-8 weeks','Monitor every 2 weeks','Maintain proper mound height'],
        'description':'Your yam plant appears healthy. Yam needs consistent monitoring due to its disease susceptibility.'
    },
    'Yam Mosaic Virus': {
        'crop':'Yam','severity':'Medium','color':'#e67e22',
        'icon':'🟠',
        'symptoms':['Yellow and green mosaic on leaves','Leaf distortion and puckering','Stunted vine growth','Reduced tuber yield'],
        'treatment':['Use virus-free certified planting material','Control aphid vectors','Remove infected plants','Avoid infected neighboring fields'],
        'description':'Caused by Yam Mosaic Virus spread primarily by aphids. Can cause up to 70% yield reduction.'
    },
    'Yam Leaf Spot': {
        'crop':'Yam','severity':'Medium','color':'#e67e22',
        'icon':'🟠',
        'symptoms':['Small brown spots on leaves','Lighter centers on spots','Yellowing around spots','Premature defoliation'],
        'treatment':['Apply mancozeb or copper oxychloride','Remove infected leaves','Avoid overhead irrigation','Ensure proper plant spacing'],
        'description':'Caused by fungal pathogens. Reduces photosynthetic capacity leading to yield losses.'
    },
}

DEFAULT_INFO = {
    'crop':'Unknown','severity':'Unknown','color':'#95a5a6','icon':'⚪',
    'symptoms':['Consult an agricultural extension officer'],
    'treatment':['Seek advice from your local agricultural officer'],
    'description':'Disease information not available. Please consult an expert.'
}

MODEL_RESULTS = {
    'CustomCNN'     : {'accuracy':15.39,'f1':0.0616,'precision':0.0476,'recall':0.1539},
    'VGG16'         : {'accuracy':78.57,'f1':0.7842,'precision':0.7897,'recall':0.7857},
    'ResNet50'      : {'accuracy':38.44,'f1':0.3700,'precision':0.4413,'recall':0.3844},
    'EfficientNetB0': {'accuracy':0.42, 'f1':0.0131,'precision':0.0071,'recall':0.0042},
    'MobileNetV2'   : {'accuracy':85.13,'f1':0.8499,'precision':0.8513,'recall':0.8513},
}

# ── Global CSS ────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=DM+Serif+Display&display=swap');

    * { font-family: 'Nunito', sans-serif; }

    html, body, .stApp {
        background: #f0f4f0 !important;
        margin: 0; padding: 0;
    }

    /* Hide streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
    section[data-testid="stSidebar"] { display: none; }

    /* Mobile phone frame */
    .block-container {
        max-width: 420px !important;
        margin: 0 auto !important;
        padding: 0 !important;
        background: white;
        min-height: 100vh;
        box-shadow: 0 0 60px rgba(0,0,0,0.15);
        position: relative;
    }

    /* Status bar */
    .status-bar {
        background: #1a5c2e;
        color: white;
        font-size: 11px;
        font-weight: 700;
        padding: 8px 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        letter-spacing: 0.5px;
    }

    /* App header bar */
    .app-header {
        background: linear-gradient(135deg, #1a5c2e, #2d8a4e);
        color: white;
        padding: 16px 20px;
        display: flex;
        align-items: center;
        gap: 12px;
        box-shadow: 0 3px 15px rgba(26,92,46,0.3);
    }
    .app-header-icon { font-size: 28px; }
    .app-header-title { font-size: 18px; font-weight: 800; letter-spacing: 0.5px; }
    .app-header-sub   { font-size: 10px; opacity: 0.8; margin-top: 2px; }

    /* Splash / Login screens */
    .splash-screen {
        background: linear-gradient(180deg, #1a5c2e 0%, #2d8a4e 50%, #3dac61 100%);
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        color: white;
        text-align: center;
        padding: 40px 30px;
    }
    .splash-logo { font-size: 80px; margin-bottom: 20px; animation: pulse 2s infinite; }
    @keyframes pulse { 0%,100%{transform:scale(1);} 50%{transform:scale(1.05);} }
    .splash-title { font-family:'DM Serif Display',serif; font-size:36px; margin-bottom:8px; }
    .splash-sub { font-size:14px; opacity:0.85; margin-bottom:40px; line-height:1.6; }
    .splash-tagline { font-size:12px; opacity:0.6; margin-top:30px; }

    /* Green button */
    .btn-green {
        display: block;
        background: white;
        color: #1a5c2e;
        font-weight: 800;
        font-size: 15px;
        padding: 15px 40px;
        border-radius: 50px;
        text-align: center;
        cursor: pointer;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        margin: 10px auto;
        border: none;
        width: 100%;
        letter-spacing: 0.5px;
    }
    .btn-outline {
        display: block;
        background: transparent;
        color: white;
        font-weight: 700;
        font-size: 14px;
        padding: 13px 40px;
        border-radius: 50px;
        text-align: center;
        cursor: pointer;
        border: 2px solid rgba(255,255,255,0.6);
        width: 100%;
        letter-spacing: 0.5px;
    }

    /* Login card */
    .login-card {
        background: white;
        border-radius: 30px 30px 0 0;
        padding: 35px 30px;
        margin-top: -20px;
    }
    .login-title { font-size:22px; font-weight:800; color:#1a5c2e; margin-bottom:5px; }
    .login-sub   { font-size:13px; color:#888; margin-bottom:25px; }

    /* Input fields */
    .stTextInput > div > div > input {
        background: #f5f9f5 !important;
        border: 2px solid #e0ede0 !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        font-size: 14px !important;
        color: #222 !important;
        font-family: 'Nunito', sans-serif !important;
        transition: all 0.2s;
    }
    .stTextInput > div > div > input:focus {
        border-color: #2d8a4e !important;
        box-shadow: 0 0 0 3px rgba(45,138,78,0.15) !important;
    }

    /* Streamlit button override */
    .stButton > button {
        background: linear-gradient(135deg, #1a5c2e, #2d8a4e) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        font-weight: 800 !important;
        font-size: 15px !important;
        padding: 14px 30px !important;
        width: 100% !important;
        font-family: 'Nunito', sans-serif !important;
        box-shadow: 0 4px 15px rgba(26,92,46,0.3) !important;
        transition: all 0.2s !important;
        letter-spacing: 0.5px !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(26,92,46,0.4) !important;
    }

    /* Home page */
    .home-hero {
        background: linear-gradient(135deg, #1a5c2e, #2d8a4e);
        color: white;
        padding: 25px 20px 35px;
        margin-bottom: -20px;
    }
    .home-greeting { font-size:13px; opacity:0.8; }
    .home-name     { font-size:22px; font-weight:800; margin:3px 0; }
    .home-tagline  { font-size:12px; opacity:0.7; }

    /* Cards */
    .home-card {
        background: white;
        border-radius: 20px;
        padding: 20px;
        margin: 10px 20px;
        box-shadow: 0 3px 15px rgba(0,0,0,0.08);
        cursor: pointer;
        transition: all 0.2s;
        border: 1px solid #f0f0f0;
    }
    .home-card:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.12); }

    .quick-scan-card {
        background: linear-gradient(135deg, #1a5c2e, #2d8a4e);
        color: white;
        border-radius: 20px;
        padding: 25px 20px;
        margin: 10px 20px;
        box-shadow: 0 5px 20px rgba(26,92,46,0.35);
        cursor: pointer;
    }
    .quick-scan-title { font-size:18px; font-weight:800; margin-bottom:5px; }
    .quick-scan-sub   { font-size:12px; opacity:0.8; }

    .stat-row {
        display: flex;
        gap: 10px;
        margin: 0 20px 10px;
    }
    .stat-box {
        flex: 1;
        background: white;
        border-radius: 16px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 3px 10px rgba(0,0,0,0.06);
    }
    .stat-num   { font-size: 22px; font-weight: 900; color: #1a5c2e; }
    .stat-label { font-size: 10px; color: #888; margin-top: 2px; font-weight: 600; }

    /* Prediction page */
    .predict-upload {
        background: #f5f9f5;
        border: 2.5px dashed #2d8a4e;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin: 15px 0;
        cursor: pointer;
    }
    .upload-icon  { font-size: 40px; margin-bottom: 10px; }
    .upload-text  { font-size: 15px; font-weight: 700; color: #1a5c2e; }
    .upload-sub   { font-size: 12px; color: #888; margin-top: 5px; }

    .result-card {
        border-radius: 20px;
        padding: 20px;
        margin: 15px 0;
        text-align: center;
        color: white;
    }
    .result-disease  { font-size: 20px; font-weight: 800; margin-bottom: 5px; }
    .result-crop     { font-size: 13px; opacity: 0.9; }
    .result-conf     { font-size: 28px; font-weight: 900; margin: 10px 0 5px; }
    .result-conf-sub { font-size: 11px; opacity: 0.8; }

    .severity-pill {
        display: inline-block;
        padding: 5px 16px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        margin-top: 10px;
        background: rgba(255,255,255,0.25);
        color: white;
    }

    .info-section {
        background: white;
        border-radius: 20px;
        padding: 18px;
        margin: 10px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    }
    .info-title { font-size: 15px; font-weight: 800; color: #1a5c2e; margin-bottom: 12px; }
    .info-item  { display: flex; align-items: flex-start; gap: 10px; margin: 8px 0; }
    .info-dot   { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; margin-top: 5px; }
    .info-text  { font-size: 13px; color: #444; line-height: 1.5; }

    /* Bottom nav */
    .bottom-nav {
        position: fixed;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 420px;
        background: white;
        border-top: 1px solid #f0f0f0;
        display: flex;
        justify-content: space-around;
        padding: 10px 0 20px;
        box-shadow: 0 -5px 20px rgba(0,0,0,0.08);
        z-index: 1000;
    }
    .nav-item { text-align: center; cursor: pointer; padding: 5px 15px; }
    .nav-icon { font-size: 22px; }
    .nav-label { font-size: 10px; font-weight: 700; color: #888; margin-top: 3px; }
    .nav-active .nav-label { color: #1a5c2e; }
    .nav-active .nav-icon  { filter: none; }

    /* Page content padding for bottom nav */
    .page-content { padding-bottom: 90px; }

    /* Section header */
    .section-header {
        padding: 20px 20px 5px;
        font-size: 16px;
        font-weight: 800;
        color: #1a1a1a;
    }
    .section-sub {
        padding: 0 20px 15px;
        font-size: 12px;
        color: #888;
    }

    /* Disease list item */
    .disease-item {
        display: flex;
        align-items: center;
        gap: 15px;
        background: white;
        border-radius: 16px;
        padding: 14px 16px;
        margin: 6px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        cursor: pointer;
    }
    .disease-icon-box {
        width: 45px; height: 45px;
        border-radius: 12px;
        display: flex; align-items: center;
        justify-content: center; font-size: 22px;
        flex-shrink: 0;
    }
    .disease-name { font-size: 14px; font-weight: 700; color: #1a1a1a; }
    .disease-crop { font-size: 11px; color: #888; margin-top: 2px; }

    /* Results page */
    .results-metric {
        background: white;
        border-radius: 16px;
        padding: 16px;
        margin: 6px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .metric-left  { font-size: 14px; font-weight: 700; color: #1a1a1a; }
    .metric-right { font-size: 14px; font-weight: 800; color: #1a5c2e; }
    .metric-best  { color: #e74c3c; font-size: 11px; font-weight: 700; }

    /* Profile page */
    .profile-avatar {
        width: 80px; height: 80px;
        background: linear-gradient(135deg, #1a5c2e, #2d8a4e);
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 36px;
        margin: 0 auto 15px;
        box-shadow: 0 5px 20px rgba(26,92,46,0.3);
    }
    .profile-name  { text-align:center; font-size:20px; font-weight:800; color:#1a1a1a; }
    .profile-role  { text-align:center; font-size:13px; color:#888; margin-top:3px; }

    .profile-item {
        display: flex; justify-content: space-between;
        align-items: center;
        padding: 16px 20px;
        border-bottom: 1px solid #f5f5f5;
        cursor: pointer;
    }
    .profile-item-left  { font-size:14px; font-weight:600; color:#1a1a1a; }
    .profile-item-right { font-size:18px; color:#888; }

    /* Top 3 bar */
    .conf-bar-wrap { margin: 6px 0; }
    .conf-bar-label { display: flex; justify-content: space-between; font-size: 12px; color: #444; margin-bottom: 3px; font-weight: 600; }
    .conf-bar-track { background: #f0f0f0; border-radius: 10px; height: 8px; }
    .conf-bar-fill  { height: 8px; border-radius: 10px; background: linear-gradient(90deg, #1a5c2e, #2d8a4e); transition: width 0.6s ease; }

    /* Scrollable area */
    .scroll-content { overflow-y: auto; }

    /* Notification badge */
    .notif-badge {
        background: #e74c3c;
        color: white;
        font-size: 9px;
        font-weight: 800;
        padding: 2px 6px;
        border-radius: 10px;
        margin-left: 5px;
        vertical-align: middle;
    }

    /* Success/error messages */
    .msg-success {
        background: #e8f5e9;
        border-left: 4px solid #2ecc71;
        padding: 12px 16px;
        border-radius: 0 12px 12px 0;
        font-size: 13px;
        color: #1a5c2e;
        font-weight: 600;
        margin: 10px 0;
    }
    .msg-error {
        background: #fde8e8;
        border-left: 4px solid #e74c3c;
        padding: 12px 16px;
        border-radius: 0 12px 12px 0;
        font-size: 13px;
        color: #c0392b;
        font-weight: 600;
        margin: 10px 0;
    }

    /* Hide st labels */
    .stTextInput label { font-weight: 700 !important; color: #444 !important; font-size: 13px !important; }
    </style>
    """, unsafe_allow_html=True)

inject_css()

# ── Session state init ────────────────────────────────────────
if 'page'       not in st.session_state: st.session_state.page       = 'splash'
if 'logged_in'  not in st.session_state: st.session_state.logged_in  = False
if 'username'   not in st.session_state: st.session_state.username   = ''
if 'nav'        not in st.session_state: st.session_state.nav        = 'home'
if 'prediction' not in st.session_state: st.session_state.prediction = None
if 'history'    not in st.session_state: st.session_state.history    = []

# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = None
    class_info = None
    for mf in ['BEST_VGG16.h5','best_VGG16.h5','model.h5']:
        if os.path.exists(mf):
            model = tf.keras.models.load_model(mf)
            break
    if os.path.exists('class_info.json'):
        with open('class_info.json') as f:
            class_info = json.load(f)
    return model, class_info

model, class_info = load_model()
CLASS_NAMES = class_info['class_names'] if class_info else []
IMG_SIZE    = tuple(class_info['img_size']) if class_info else (224, 224)

def preprocess(img):
    img = img.convert('RGB').resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def get_info(name):
    if name in DISEASE_DB: return DISEASE_DB[name]
    for k in DISEASE_DB:
        if k.lower() in name.lower() or name.lower() in k.lower():
            return DISEASE_DB[k]
    return DEFAULT_INFO

# ── Status bar ────────────────────────────────────────────────
st.markdown("""
<div class="status-bar">
    <span>9:41 AM</span>
    <span>🔋 98% &nbsp; 📶</span>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE: SPLASH
# ══════════════════════════════════════════════════════════════
if st.session_state.page == 'splash':
    st.markdown("""
    <div style="background:linear-gradient(180deg,#1a5c2e 0%,#2d8a4e 60%,#3dac61 100%);
                min-height:85vh; display:flex; flex-direction:column;
                align-items:center; justify-content:center;
                color:white; text-align:center; padding:40px 30px;">
        <div style="font-size:90px; margin-bottom:20px; animation:pulse 2s infinite;">🌿</div>
        <div style="font-family:'DM Serif Display',serif; font-size:38px; margin-bottom:8px;">CropGuard AI</div>
        <div style="font-size:14px; opacity:0.85; line-height:1.7; margin-bottom:10px;">
            Smart Crop Disease Detection<br>
            for Nigerian Farmers
        </div>
        <div style="margin-top:15px; font-size:11px; opacity:0.55;">
            Cassava &nbsp;•&nbsp; Maize &nbsp;•&nbsp; Yam
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='padding:20px;background:white;'>", unsafe_allow_html=True)
    if st.button("🚀  Get Started", key="splash_start"):
        st.session_state.page = 'login'
        st.rerun()

    st.markdown("""
    <div style="text-align:center; margin-top:15px; font-size:11px; color:#aaa;">
        Bowen University, Iwo &nbsp;|&nbsp; Dept. of Computer Science<br>
        <i>Ojugbeli Ogechi Esther </i>
    </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE: LOGIN
# ══════════════════════════════════════════════════════════════
elif st.session_state.page == 'login':
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a5c2e,#2d8a4e);
                padding:40px 30px 60px; text-align:center; color:white;">
        <div style="font-size:55px; margin-bottom:12px;">🌱</div>
        <div style="font-family:'DM Serif Display',serif; font-size:26px;">Welcome Back</div>
        <div style="font-size:13px; opacity:0.8; margin-top:5px;">Sign in to continue</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:white; border-radius:30px 30px 0 0;
                padding:30px 25px; margin-top:-25px;">
    """, unsafe_allow_html=True)

    username = st.text_input("👤  Username", placeholder="Enter your username")
    password = st.text_input("🔒  Password", type="password", placeholder="Enter your password")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Sign In  →", key="login_btn"):
        if username in USERS and USERS[username] == password:
            st.session_state.logged_in = True
            st.session_state.username  = username
            st.session_state.page      = 'app'
            st.session_state.nav       = 'home'
            st.rerun()
        else:
            st.markdown('<div class="msg-error">❌ Wrong username or password. Try again.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; margin-top:20px; font-size:12px; color:#999;">
        Demo accounts:<br>
        <code>farmer / farm123</code> &nbsp;|&nbsp;
        <code>admin / admin123</code>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Back to Home", key="back_splash"):
        st.session_state.page = 'splash'
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# MAIN APP (after login)
# ══════════════════════════════════════════════════════════════
elif st.session_state.page == 'app':

    if not st.session_state.logged_in:
        st.session_state.page = 'login'
        st.rerun()

    username = st.session_state.username

    # ── App header ────────────────────────────────────────────
    nav = st.session_state.nav

    page_titles = {
        'home'    : ('🌿 CropGuard AI',   'Bowen University'),
        'predict' : ('🔍 Scan Leaf',       'AI Disease Detection'),
        'diseases': ('📖 Disease Guide',   '15 disease classes'),
        'results' : ('📊 Model Results',   'Performance metrics'),
        'profile' : ('👤 Profile',         f'{username}'),
    }
    title, sub = page_titles.get(nav, ('CropGuard AI',''))

    st.markdown(f"""
    <div class="app-header">
        <div class="app-header-icon">🌿</div>
        <div>
            <div class="app-header-title">{title}</div>
            <div class="app-header-sub">{sub}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="page-content">', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # NAV: HOME
    # ══════════════════════════════════════════════════════════
    if nav == 'home':
        greeting = "Good morning" if True else "Hello"
        st.markdown(f"""
        <div class="home-hero">
            <div class="home-greeting">👋 Welcome back,</div>
            <div class="home-name">{username.capitalize()}</div>
            <div class="home-tagline">Ready to protect your crops today?</div>
        </div>
        <div style="background:white; border-radius:25px 25px 0 0; padding-top:20px; margin-top:-15px;">
        """, unsafe_allow_html=True)

        # Quick scan card
        if st.button("📷  Scan a Leaf Now  →", key="quick_scan"):
            st.session_state.nav = 'predict'
            st.rerun()

        st.markdown("""
        <div style="padding:0 20px;">
            <div style="background:linear-gradient(135deg,#1a5c2e,#2d8a4e);
                        color:white; border-radius:20px; padding:22px;
                        box-shadow:0 5px 20px rgba(26,92,46,0.3); margin-bottom:15px;">
                <div style="font-size:28px; margin-bottom:8px;">🔬</div>
                <div style="font-size:17px; font-weight:800;">Instant Disease Detection</div>
                <div style="font-size:12px; opacity:0.8; margin-top:5px;">
                    Upload a leaf photo and get AI-powered results in seconds
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Stats
        st.markdown("""
        <div class="stat-row">
            <div class="stat-box">
                <div class="stat-num">15</div>
                <div class="stat-label">DISEASE CLASSES</div>
            </div>
            <div class="stat-box">
                <div class="stat-num">85%</div>
                <div class="stat-label">ACCURACY</div>
            </div>
            <div class="stat-box">
                <div class="stat-num">3</div>
                <div class="stat-label">CROPS COVERED</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Crops covered
        st.markdown('<div class="section-header">Crops Covered</div>', unsafe_allow_html=True)
        crop_cols = st.columns(3)
        crops = [
            ("🟤", "Cassava", "4 conditions"),
            ("🌽", "Maize",   "7 conditions"),
            ("🍠", "Yam",     "4 conditions"),
        ]
        for col, (emoji, name, count) in zip(crop_cols, crops):
            with col:
                st.markdown(f"""
                <div style="background:white; border-radius:16px; padding:15px 10px;
                            text-align:center; box-shadow:0 2px 10px rgba(0,0,0,0.07);
                            margin:0 5px;">
                    <div style="font-size:28px;">{emoji}</div>
                    <div style="font-size:13px; font-weight:800; color:#1a1a1a; margin-top:6px;">{name}</div>
                    <div style="font-size:10px; color:#888; margin-top:2px;">{count}</div>
                </div>
                """, unsafe_allow_html=True)

        # Recent history
        if st.session_state.history:
            st.markdown('<div class="section-header">Recent Scans</div>', unsafe_allow_html=True)
            for item in reversed(st.session_state.history[-3:]):
                info = get_info(item['disease'])
                st.markdown(f"""
                <div class="disease-item">
                    <div class="disease-icon-box" style="background:{info['color']}22;">
                        <span>{info['icon']}</span>
                    </div>
                    <div style="flex:1;">
                        <div class="disease-name">{item['disease']}</div>
                        <div class="disease-crop">{info['crop']} &nbsp;•&nbsp; {item['conf']:.1f}% confidence</div>
                    </div>
                    <div style="font-size:18px; color:#ccc;">›</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align:center; padding:20px; color:#bbb; font-size:11px;">
            🌿 CropGuard AI &nbsp;•&nbsp; Bowen University, Iwo
        </div>
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # NAV: PREDICT
    # ══════════════════════════════════════════════════════════
    elif nav == 'predict':
        st.markdown('<div style="padding:15px 20px;">', unsafe_allow_html=True)

        if model is None or class_info is None:
            st.markdown("""
            <div class="msg-error">
                ⚠️ Model not loaded!<br>
                Make sure <strong>BEST_MobileNetV2.h5</strong> and <strong>class_info.json</strong>
                are in the same folder as this script.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center; margin-bottom:15px;">
                <div style="font-size:14px; color:#555; line-height:1.6;">
                    Take a clear photo of a <strong>cassava, maize or yam</strong> leaf
                    and upload it below for instant disease detection.
                </div>
            </div>
            """, unsafe_allow_html=True)

            uploaded = st.file_uploader(
                "Choose leaf image",
                type=['jpg','jpeg','png','bmp','webp'],
                label_visibility="collapsed"
            )

            if uploaded is None:
                st.markdown("""
                <div class="predict-upload">
                    <div class="upload-icon">📷</div>
                    <div class="upload-text">Upload Leaf Photo</div>
                    <div class="upload-sub">JPG, PNG, BMP, WEBP supported</div>
                </div>
                <div style="text-align:center; margin:15px 0; font-size:12px; color:#888;">
                    💡 <strong>Tips for best results:</strong><br>
                    Take photo in good natural lighting<br>
                    Focus on the leaf — include the whole leaf<br>
                    Avoid blurry or very dark images
                </div>
                """, unsafe_allow_html=True)
            else:
                img = Image.open(uploaded)
                st.image(img, use_column_width=True, caption="Uploaded leaf image")

                with st.spinner("🔬 Analysing leaf..."):
                    arr   = preprocess(img)
                    preds = model.predict(arr, verbose=0)[0]

                top_idx   = np.argmax(preds)
                top_class = CLASS_NAMES[top_idx]
                top_conf  = float(preds[top_idx]) * 100
                info      = get_info(top_class)

                # Save to history
                st.session_state.prediction = {
                    'disease': top_class,
                    'conf'   : top_conf,
                    'info'   : info
                }
                if not st.session_state.history or st.session_state.history[-1]['disease'] != top_class:
                    st.session_state.history.append({'disease': top_class, 'conf': top_conf})

                # Result card
                st.markdown(f"""
                <div class="result-card" style="background:linear-gradient(135deg,{info['color']},{info['color']}cc);">
                    <div style="font-size:40px; margin-bottom:10px;">{info['icon']}</div>
                    <div class="result-disease">{top_class}</div>
                    <div class="result-crop">Crop: {info['crop']}</div>
                    <div class="result-conf">{top_conf:.1f}%</div>
                    <div class="result-conf-sub">Confidence Score</div>
                    <div class="severity-pill">⚡ {info['severity']} Severity</div>
                </div>
                """, unsafe_allow_html=True)

                # Top 3
                top3_idx = np.argsort(preds)[::-1][:3]
                st.markdown('<div class="info-section">', unsafe_allow_html=True)
                st.markdown('<div class="info-title">📊 Top 3 Predictions</div>', unsafe_allow_html=True)
                for i in top3_idx:
                    cls  = CLASS_NAMES[i]
                    conf = float(preds[i]) * 100
                    st.markdown(f"""
                    <div class="conf-bar-wrap">
                        <div class="conf-bar-label">
                            <span>{cls.replace('_',' ')}</span>
                            <span>{conf:.1f}%</span>
                        </div>
                        <div class="conf-bar-track">
                            <div class="conf-bar-fill" style="width:{conf}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # Description
                st.markdown(f"""
                <div class="info-section">
                    <div class="info-title">📋 About This Disease</div>
                    <div style="font-size:13px; color:#555; line-height:1.6;">
                        {info['description']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Symptoms
                symptoms_html = ''.join([
                    f'<div class="info-item"><div class="info-dot" style="background:#FF9800;"></div><div class="info-text">{s}</div></div>'
                    for s in info['symptoms']
                ])
                st.markdown(f"""
                <div class="info-section">
                    <div class="info-title">🚨 Symptoms</div>
                    {symptoms_html}
                </div>
                """, unsafe_allow_html=True)

                # Treatment
                treatment_html = ''.join([
                    f'<div class="info-item"><div class="info-dot" style="background:#2ecc71;"></div><div class="info-text">{t}</div></div>'
                    for t in info['treatment']
                ])
                st.markdown(f"""
                <div class="info-section">
                    <div class="info-title">💊 Recommended Treatment</div>
                    {treatment_html}
                </div>
                """, unsafe_allow_html=True)

                # Help
                st.markdown("""
                <div class="info-section" style="border:2px solid #1a5c2e22;">
                    <div class="info-title">📞 Need Expert Help?</div>
                    <div class="info-item">
                        <div class="info-dot" style="background:#1a5c2e;"></div>
                        <div class="info-text">Contact your local Agricultural Extension Officer</div>
                    </div>
                    <div class="info-item">
                        <div class="info-dot" style="background:#1a5c2e;"></div>
                        <div class="info-text">NASC Helpline: <strong>0800-FARMER</strong></div>
                    </div>
                    <div class="info-item">
                        <div class="info-dot" style="background:#1a5c2e;"></div>
                        <div class="info-text">IITA Nigeria: <strong>www.iita.org</strong></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # NAV: DISEASES
    # ══════════════════════════════════════════════════════════
    elif nav == 'diseases':
        crop_filter = st.selectbox(
            "Filter",
            ["All Crops 🌿", "Cassava 🟤", "Maize 🌽", "Yam 🍠"],
            label_visibility="collapsed"
        )
        crop_map = {
            "All Crops 🌿": None,
            "Cassava 🟤"  : "Cassava",
            "Maize 🌽"    : "Maize",
            "Yam 🍠"      : "Yam"
        }
        selected_crop = crop_map[crop_filter]

        for disease, info in DISEASE_DB.items():
            if selected_crop and info['crop'] != selected_crop:
                continue
            crop_emoji = {'Cassava':'🟤','Maize':'🌽','Yam':'🍠'}.get(info['crop'],'🌿')

            with st.expander(f"{info['icon']} {disease}"):
                st.markdown(f"""
                <div style="background:{info['color']}15; border-radius:12px; padding:12px; margin-bottom:12px;">
                    <strong style="color:{info['color']};">
                        {crop_emoji} {info['crop']} &nbsp;|&nbsp; {info['severity']} Severity
                    </strong><br>
                    <span style="font-size:13px; color:#555;">{info['description']}</span>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("**🚨 Symptoms:**")
                for s in info['symptoms']:
                    st.markdown(f"- {s}")

                st.markdown("**💊 Treatment:**")
                for t in info['treatment']:
                    st.markdown(f"- ✅ {t}")

    # ══════════════════════════════════════════════════════════
    # NAV: RESULTS
    # ══════════════════════════════════════════════════════════
    elif nav == 'results':
        st.markdown('<div style="padding:15px 20px;">', unsafe_allow_html=True)

        st.markdown("""
        <div style="background:linear-gradient(135deg,#1a5c2e,#2d8a4e);
                    border-radius:20px; padding:20px; color:white; margin-bottom:15px; text-align:center;">
            <div style="font-size:13px; opacity:0.8;">Best Model</div>
            <div style="font-size:24px; font-weight:900; margin:5px 0;">MobileNetV2</div>
            <div style="display:flex; justify-content:space-around; margin-top:15px;">
                <div><div style="font-size:22px; font-weight:900;">85.1%</div><div style="font-size:10px; opacity:0.7;">ACCURACY</div></div>
                <div><div style="font-size:22px; font-weight:900;">0.850</div><div style="font-size:10px; opacity:0.7;">F1-SCORE</div></div>
                <div><div style="font-size:22px; font-weight:900;">85.1%</div><div style="font-size:10px; opacity:0.7;">PRECISION</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**All 5 Models — Test Set Results:**")
        for name, r in MODEL_RESULTS.items():
            is_best = name == 'MobileNetV2'
            border  = "border:2px solid #1a5c2e;" if is_best else ""
            st.markdown(f"""
            <div style="background:white; border-radius:16px; padding:15px; margin:8px 0;
                        box-shadow:0 2px 8px rgba(0,0,0,0.07); {border}">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                    <div>
                        <span style="font-size:14px; font-weight:800; color:#1a1a1a;">{name}</span>
                        {'<span style="background:#1a5c2e; color:white; font-size:9px; font-weight:700; padding:2px 8px; border-radius:10px; margin-left:8px;">BEST</span>' if is_best else ''}
                    </div>
                    <span style="font-size:16px; font-weight:900; color:{"#1a5c2e" if is_best else "#666"};">{r["accuracy"]:.1f}%</span>
                </div>
                <div style="background:#f0f0f0; border-radius:10px; height:6px;">
                    <div style="width:{r["accuracy"]}%; height:6px; border-radius:10px;
                                background:{"linear-gradient(90deg,#1a5c2e,#2d8a4e)" if is_best else "#ccc"};"></div>
                </div>
                <div style="display:flex; justify-content:space-between; margin-top:8px; font-size:11px; color:#888;">
                    <span>P: {r["precision"]:.3f}</span>
                    <span>R: {r["recall"]:.3f}</span>
                    <span>F1: {r["f1"]:.3f}</span>
                    <span>{"Transfer Learning" if name != "CustomCNN" else "From Scratch"}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Chart
        st.markdown("<br>**Accuracy Chart:**", unsafe_allow_html=True)
        models = list(MODEL_RESULTS.keys())
        accs   = [MODEL_RESULTS[m]['accuracy'] for m in models]
        fig, ax = plt.subplots(figsize=(5, 3))
        colors = ['#2d8a4e' if m == 'MobileNetV2' else '#c8e6c9' for m in models]
        bars   = ax.barh(models, accs, color=colors, height=0.5)
        for bar, acc in zip(bars, accs):
            ax.text(acc+0.5, bar.get_y()+bar.get_height()/2,
                    f'{acc:.1f}%', va='center', fontsize=9, fontweight='bold', color='#333')
        ax.set_xlim([0, 100])
        ax.set_xlabel('Accuracy (%)', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=9)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("""
        <div style="background:#f5f9f5; border-radius:16px; padding:15px; margin-top:10px;">
            <div style="font-size:13px; font-weight:700; color:#1a5c2e; margin-bottom:8px;">📂 Dataset Info</div>
            <div style="font-size:12px; color:#555; line-height:1.8;">
                📸 Total images: ~11,500<br>
                🎓 Training: 70% &nbsp;|&nbsp; Validation: 15% &nbsp;|&nbsp; Test: 15%<br>
                🌾 Crops: Cassava, Maize, Yam<br>
                🏷️ Classes: 15 disease categories<br>
                📐 Image size: 224 × 224 pixels
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # NAV: PROFILE
    # ══════════════════════════════════════════════════════════
    elif nav == 'profile':
        st.markdown(f"""
        <div style="padding:25px 20px; text-align:center;">
            <div class="profile-avatar">👤</div>
            <div class="profile-name">{username.capitalize()}</div>
            <div class="profile-role">CropGuard AI User</div>
        </div>
        """, unsafe_allow_html=True)

        # Stats
        scans = len(st.session_state.history)
        st.markdown(f"""
        <div class="stat-row" style="margin:0 20px 20px;">
            <div class="stat-box">
                <div class="stat-num">{scans}</div>
                <div class="stat-label">TOTAL SCANS</div>
            </div>
            <div class="stat-box">
                <div class="stat-num">3</div>
                <div class="stat-label">CROPS</div>
            </div>
            <div class="stat-box">
                <div class="stat-num">15</div>
                <div class="stat-label">DISEASES</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Profile items
        st.markdown("""
        <div style="background:white; border-radius:20px; margin:0 20px; overflow:hidden;
                    box-shadow:0 3px 15px rgba(0,0,0,0.07);">
            <div class="profile-item">
                <div class="profile-item-left">🌿 About CropGuard AI</div>
                <div class="profile-item-right">›</div>
            </div>
            <div class="profile-item">
                <div class="profile-item-left">🏫 Bowen University, Iwo</div>
                <div class="profile-item-right">›</div>
            </div>
            <div class="profile-item">
                <div class="profile-item-left">👩‍🎓 Ojugbeli Ogechi Marvellous</div>
                <div class="profile-item-right">›</div>
            </div>
            <div class="profile-item">
                <div class="profile-item-left">👩‍🏫 Supervisor: Miss Busolami</div>
                <div class="profile-item-right">›</div>
            </div>
            <div class="profile-item">
                <div class="profile-item-left">📖 Computer Science Dept.</div>
                <div class="profile-item-right">›</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div style='padding:0 20px;'>", unsafe_allow_html=True)
        if st.button("🚪  Sign Out", key="logout_btn"):
            st.session_state.logged_in  = False
            st.session_state.username   = ''
            st.session_state.page       = 'login'
            st.session_state.prediction = None
            st.session_state.history    = []
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align:center; padding:20px; color:#bbb; font-size:11px;">
            CropGuard AI v1.0 &nbsp;•&nbsp; December 2025<br>
            Bowen University, Iwo
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Bottom Navigation ─────────────────────────────────────
    st.markdown("""
    <div class="bottom-nav">
        <div class="nav-item {h}" id="nav-home">
            <div class="nav-icon">🏠</div>
            <div class="nav-label">Home</div>
        </div>
        <div class="nav-item {p}" id="nav-predict">
            <div class="nav-icon">🔍</div>
            <div class="nav-label">Scan</div>
        </div>
        <div class="nav-item {d}" id="nav-diseases">
            <div class="nav-icon">📖</div>
            <div class="nav-label">Diseases</div>
        </div>
        <div class="nav-item {r}" id="nav-results">
            <div class="nav-icon">📊</div>
            <div class="nav-label">Results</div>
        </div>
        <div class="nav-item {pr}" id="nav-profile">
            <div class="nav-icon">👤</div>
            <div class="nav-label">Profile</div>
        </div>
    </div>
    """.format(
        h  = "nav-active" if nav == 'home'     else "",
        p  = "nav-active" if nav == 'predict'  else "",
        d  = "nav-active" if nav == 'diseases' else "",
        r  = "nav-active" if nav == 'results'  else "",
        pr = "nav-active" if nav == 'profile'  else "",
    ), unsafe_allow_html=True)

    # Navigation buttons (hidden but functional)
    st.markdown("<div style='display:flex; gap:5px; padding:0 10px; margin-bottom:5px;'>", unsafe_allow_html=True)
    nav_cols = st.columns(5)
    nav_items = [
        ("home",     "🏠"),
        ("predict",  "🔍"),
        ("diseases", "📖"),
        ("results",  "📊"),
        ("profile",  "👤"),
    ]
    for col, (page_nav, icon) in zip(nav_cols, nav_items):
        with col:
            if st.button(icon, key=f"nav_{page_nav}", help=page_nav.capitalize()):
                st.session_state.nav = page_nav
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
