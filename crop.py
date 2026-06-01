# ============================================================
# CROP DISEASE PREDICTION SYSTEM — STREAMLIT WEB APP
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

# Fix for batch_shape compatibility issue
tf.keras.backend.clear_session()

st.set_page_config(
    page_title = "CropGuard AI",
    page_icon  = "🌿",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap');
* { font-family: 'Nunito', sans-serif; }
.stApp { background: #f5f9f5; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a5c2e, #2d8a4e) !important; }
[data-testid="stSidebar"] * { color: white !important; }
#MainMenu { visibility: hidden; } footer { visibility: hidden; }
.stButton > button {
    background: linear-gradient(135deg, #1a5c2e, #2d8a4e) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; font-weight: 700 !important;
    font-size: 15px !important; padding: 12px 25px !important;
    box-shadow: 0 4px 15px rgba(26,92,46,0.3) !important;
}
.card { background:white; border-radius:20px; padding:25px; margin:10px 0; box-shadow:0 3px 15px rgba(0,0,0,0.07); }
.result-box { border-radius:20px; padding:25px; text-align:center; color:white; margin:15px 0; }
.result-name { font-size:24px; font-weight:900; margin-bottom:5px; }
.result-conf { font-size:42px; font-weight:900; margin:10px 0 5px; }
.result-conf-sub { font-size:12px; opacity:0.8; }
.pill-high    { background:#e74c3c; color:white; padding:5px 15px; border-radius:20px; font-size:13px; font-weight:700; display:inline-block; margin-top:10px; }
.pill-medium  { background:#e67e22; color:white; padding:5px 15px; border-radius:20px; font-size:13px; font-weight:700; display:inline-block; margin-top:10px; }
.pill-none    { background:#2ecc71; color:white; padding:5px 15px; border-radius:20px; font-size:13px; font-weight:700; display:inline-block; margin-top:10px; }
.pill-unknown { background:#95a5a6; color:white; padding:5px 15px; border-radius:20px; font-size:13px; font-weight:700; display:inline-block; margin-top:10px; }
.symptom-item { background:#fff8f0; border-left:4px solid #FF9800; padding:10px 14px; margin:6px 0; border-radius:0 10px 10px 0; font-size:14px; color:#444; }
.treatment-item { background:#f0fff4; border-left:4px solid #2ecc71; padding:10px 14px; margin:6px 0; border-radius:0 10px 10px 0; font-size:14px; color:#444; }
.conf-wrap { margin:8px 0; }
.conf-row  { display:flex; justify-content:space-between; font-size:13px; color:black; margin-bottom:4px; font-weight:600; }
.conf-track { background:#f0f0f0; border-radius:10px; height:10px; }
.conf-fill  { height:10px; border-radius:10px; background:linear-gradient(90deg,#1a5c2e,#2d8a4e); }
.metrics-row { display:flex; gap:15px; margin:15px 0; }
.metric-box  { flex:1; background:white; border-radius:16px; padding:18px; text-align:center; box-shadow:0 2px 10px rgba(0,0,0,0.06); }
.metric-val  { font-size:26px; font-weight:900; color:#1a5c2e; }
.metric-lbl  { font-size:11px; color:#888; margin-top:4px; font-weight:600; }
.section-title { font-size:17px; font-weight:800; color:#1a5c2e; margin:20px 0 10px; padding-bottom:8px; border-bottom:2px solid #e8f5e9; }
.stTabs [data-baseweb="tab-list"] { background:white; border-radius:14px; padding:5px; box-shadow:0 2px 10px rgba(0,0,0,0.06); gap:5px; }
.stTabs [data-baseweb="tab"] { border-radius:10px !important; font-weight:700 !important; font-size:14px !important; padding:10px 20px !important; color: black !important }
.stTabs [aria-selected="true"] { background:linear-gradient(135deg,#1a5c2e,#2d8a4e) !important; color:white !important; }
            /* Fix expander text color */
.streamlit-expanderHeader {
    color: black !important;
    font-weight: 700 !important;
}
[data-testid="stExpander"] summary {
    color: black !important;
    font-weight: 700 !important;
}
[data-testid="stExpander"] summary span {
    color: black !important;
}
            /* Fix all text inside expanders to black */
[data-testid="stExpander"] p {
    color: black !important;
}
[data-testid="stExpander"] li {
    color: black !important;
}
[data-testid="stExpander"] span {
    color: black !important;
}
[data-testid="stExpander"] div {
    color: black !important;
}
[data-testid="stExpander"] strong {
    color: black !important;
}
            /* Fix Tab 1 disease report text */
.card {
    color: black !important;
}
.section-title {
    color: #1a5c2e !important;
}
h3 {
    color: black !important;
}
.stMarkdown p {
    color: black !important;
}
.stMarkdown li {
    color: black !important;
}
.stMarkdown span {
    color: black !important;
}
            /* Fix image caption color */
[data-testid="stImage"] p {
    color: black !important;
}
            /* Fix metric text color */
[data-testid="stMetricLabel"] {
    color: black !important;
}
[data-testid="stMetricValue"] {
    color: black !important;
}
[data-testid="metric-container"] {
    color: black !important;
}
[data-testid="metric-container"] p {
    color: black !important;
}
            
            /* Fix radio button text color */
[data-testid="stRadio"] label {
    color: black !important;
}
[data-testid="stRadio"] p {
    color: black !important;
}
[data-testid="stRadio"] span {
    color: black !important;
}

            /* Fix deprecation warning text color */
.stAlert p {
    color: black !important;
}
.stAlert {
    color: black !important;
}
[data-testid="stAlert"] {
    color: black !important;
}
[data-testid="stAlert"] p {
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

DISEASE_DB = {
    'Cassava Bacterial Blight': {'crop':'Cassava','severity':'High','color':'#e74c3c','pill':'pill-high','icon':'🔴','symptoms':['Angular water-soaked spots on leaves','Yellow halo around dark spots','Stem wilting and dieback','Gummy exudate on infected stems'],'treatment':['Remove and destroy all infected plant parts','Apply copper-based bactericide (Kocide 3000)','Use certified disease-free planting material','Improve field drainage'],'description':'A serious bacterial disease caused by Xanthomonas axonopodis. Spreads through infected planting material and water splash.'},
    'Cassava Green Mite': {'crop':'Cassava','severity':'Medium','color':'#e67e22','pill':'pill-medium','icon':'🟠','symptoms':['Green mottling on young leaves','Leaf distortion and stunted growth','Reduced leaf size','Tiny mites on leaf undersides'],'treatment':['Apply acaricide (abamectin)','Introduce biological control agents','Remove heavily infested leaves','Use resistant cassava varieties'],'description':'Caused by Mononychellus tanajoa mites that feed on young leaves and growing tips.'},
    'Cassava Healthy': {'crop':'Cassava','severity':'None','color':'#2ecc71','pill':'pill-none','icon':'🟢','symptoms':['Deep green uniform leaf color','Normal leaf shape and size','No spots or lesions','Healthy stem growth'],'treatment':['Continue regular crop monitoring','Maintain proper spacing','Apply balanced NPK fertilizer','Practice crop rotation'],'description':'Your cassava plant appears healthy! Maintain good agricultural practices to keep it this way.'},
    'Cassava Mosaic': {'crop':'Cassava','severity':'High','color':'#e74c3c','pill':'pill-high','icon':'🔴','symptoms':['Yellow and green mosaic pattern','Leaf distortion and wrinkling','Stunted plant growth','Reduced tuber yield'],'treatment':['Use CMD-resistant cassava varieties','Control whitefly vectors','Remove infected plants (roguing)','Use certified planting material'],'description':'Caused by Cassava Mosaic Virus spread by whiteflies. Most economically damaging cassava disease in Nigeria.'},
    'Maize Fall Armyworm': {'crop':'Maize','severity':'High','color':'#e74c3c','pill':'pill-high','icon':'🔴','symptoms':['Ragged holes in leaves','Frass in leaf whorls','Damage to growing point','Caterpillars with Y-mark on head'],'treatment':['Apply emamectin benzoate insecticide','Use Bacillus thuringiensis (Bt)','Apply neem-based products','Scout fields twice per week'],'description':'Spodoptera frugiperda can destroy entire maize fields within days if not controlled early.'},
    'Maize Grasshopper': {'crop':'Maize','severity':'Medium','color':'#e67e22','pill':'pill-medium','icon':'🟠','symptoms':['Irregular chewing on leaf edges','Skeletonized leaves in severe cases','Visible grasshoppers on plants','Worse during dry seasons'],'treatment':['Apply lambda-cyhalothrin insecticide','Use neem extract as botanical control','Remove weeds around field edges','Plant early to avoid peak season'],'description':'Seasonal pests causing significant leaf damage especially during dry spells in Northern Nigeria.'},
    'Maize Healthy': {'crop':'Maize','severity':'None','color':'#2ecc71','pill':'pill-none','icon':'🟢','symptoms':['Vibrant green leaf color','No spots or holes visible','Strong upright plant','Healthy ear development'],'treatment':['Continue weekly field scouting','Apply nitrogen top-dressing','Maintain proper plant spacing','Monitor for pest pressure'],'description':'Your maize plant appears healthy. Continue with good crop management practices.'},
    'Maize Leaf Beetle': {'crop':'Maize','severity':'Medium','color':'#e67e22','pill':'pill-medium','icon':'🟠','symptoms':['Long parallel strips of feeding damage','Window-pane effect on leaves','Visible yellow-brown beetles','White streak feeding marks'],'treatment':['Apply cypermethrin insecticide','Hand-collect beetles on small farms','Use neem seed kernel extract','Practice crop rotation'],'description':'Leaf beetles feed on leaf tissue creating characteristic patterns that reduce photosynthesis.'},
    'Maize Leaf Blight': {'crop':'Maize','severity':'High','color':'#e74c3c','pill':'pill-high','icon':'🔴','symptoms':['Large tan cigar-shaped lesions on leaves','Lesions with dark brown borders','Disease spreads from lower leaves','Severe cases cause full leaf death'],'treatment':['Apply mancozeb or propiconazole fungicide','Use resistant hybrid varieties','Avoid overhead irrigation','Remove infected crop residues'],'description':'Northern or Southern Leaf Blight caused by fungal pathogens. Favored by warm humid conditions.'},
    'Maize Leaf Spot': {'crop':'Maize','severity':'Medium','color':'#e67e22','pill':'pill-medium','icon':'🟠','symptoms':['Small circular spots on leaves','Spots with yellow or brown halo','Multiple spots may merge','Affects all growth stages'],'treatment':['Apply azoxystrobin fungicide','Improve air circulation','Avoid excess nitrogen fertilizer','Use certified disease-free seed'],'description':'Gray leaf spot caused by Cercospora zeae-maydis, favored by high humidity and poor airflow.'},
    'Maize Streak Virus': {'crop':'Maize','severity':'High','color':'#e74c3c','pill':'pill-high','icon':'🔴','symptoms':['Narrow yellow streaks along leaf veins','Stunted plant growth','White streaking on young leaves','Severe yellowing of entire plant'],'treatment':['Use streak-resistant maize varieties','Control leafhopper vectors','Remove infected plants early','Plant early to escape peak season'],'description':'Maize Streak Virus transmitted by leafhopper Cicadulina mbila. Most damaging maize disease in sub-Saharan Africa.'},
    'Yam Anthracnose': {'crop':'Yam','severity':'High','color':'#e74c3c','pill':'pill-high','icon':'🔴','symptoms':['Dark brown circular spots on leaves','Spots with yellow halo','Stem dieback from tips','Dark sunken lesions on tubers'],'treatment':['Apply mancozeb fungicide every 2 weeks','Remove and destroy infected parts','Use disease-free seed yams','Improve field drainage'],'description':'Caused by Colletotrichum gloeosporioides. One of the most serious yam diseases in Nigeria.'},
    'Yam Healthy': {'crop':'Yam','severity':'None','color':'#2ecc71','pill':'pill-none','icon':'🟢','symptoms':['Dark green healthy leaf color','No spots or lesions','Normal vine growth','Healthy tuber development'],'treatment':['Continue staking vines','Apply fertilizer at 6-8 weeks','Monitor every 2 weeks','Maintain proper mound height'],'description':'Your yam plant appears healthy. Yam needs consistent monitoring due to its disease susceptibility.'},
    'Yam Mosaic Virus': {'crop':'Yam','severity':'Medium','color':'#e67e22','pill':'pill-medium','icon':'🟠','symptoms':['Yellow and green mosaic on leaves','Leaf distortion and puckering','Stunted vine growth','Reduced tuber yield'],'treatment':['Use virus-free certified planting material','Control aphid vectors','Remove infected plants','Avoid infected neighboring fields'],'description':'Caused by Yam Mosaic Virus spread primarily by aphids. Can cause up to 70% yield reduction.'},
    'Yam Leaf Spot': {'crop':'Yam','severity':'Medium','color':'#e67e22','pill':'pill-medium','icon':'🟠','symptoms':['Small brown spots on leaves','Lighter centers on spots','Yellowing around spots','Premature defoliation'],'treatment':['Apply mancozeb or copper oxychloride','Remove infected leaves','Avoid overhead irrigation','Ensure proper plant spacing'],'description':'Caused by fungal pathogens. Reduces photosynthetic capacity leading to yield losses.'},
}

DEFAULT_INFO = {'crop':'Unknown','severity':'Unknown','color':'#95a5a6','pill':'pill-unknown','icon':'⚪','symptoms':['Consult an agricultural extension officer'],'treatment':['Seek advice from your local agricultural officer'],'description':'Disease information not available. Please consult an expert.'}

MODEL_RESULTS = {
    'CustomCNN'     : {'accuracy':15.39,'f1':0.0616,'precision':0.0476,'recall':0.1539,'type':'From Scratch'},
    'VGG16'         : {'accuracy':78.57,'f1':0.7842,'precision':0.7897,'recall':0.7857,'type':'Transfer Learning'},
    'ResNet50'      : {'accuracy':38.44,'f1':0.3700,'precision':0.4413,'recall':0.3844,'type':'Transfer Learning'},
    'EfficientNetB0': {'accuracy':0.42, 'f1':0.0131,'precision':0.0071,'recall':0.0042,'type':'Transfer Learning'},
    'MobileNetV2'   : {'accuracy':85.13,'f1':0.8499,'precision':0.8513,'recall':0.8513,'type':'Transfer Learning'},
}

@st.cache_resource(show_spinner="🌿 Loading model... please wait")
def load_model():
    model      = None
    class_info = None
    
    # for mf in ['BEST_MobileNetV2.h5', 'best_MobileNetV2.h5', 'model.h5']:
    #     if os.path.exists(mf):
    #         try:
    #             # Try normal load first
    #             model = tf.keras.models.load_model(mf)
    #         except Exception:
    #             try:
    #                 # Try with compile=False if normal load fails
    #                 model = tf.keras.models.load_model(mf, compile=False)
    #             except Exception as e:
    #                 st.error(f"Model loading error: {e}")
    #         break

    for mf in ['VGG16_fixed.h5', 'best_VGG16.h5', 'BEST_VGG16.h5']:
        if os.path.exists(mf):
            try:
                model = tf.keras.models.load_model(mf, compile=False)
            except Exception as e:
                st.error(f"Model loading error: {e}")
            break
            
    if os.path.exists('class_info.json'):
        with open('class_info.json') as f:
            class_info = json.load(f)
            
    return model, class_info

def preprocess(img, size=(224,224)):
    img = img.convert('RGB').resize(size)
    return np.expand_dims(np.array(img)/255.0, axis=0)

def get_info(name):
    if name in DISEASE_DB: return DISEASE_DB[name]
    for k in DISEASE_DB:
        if k.lower() in name.lower() or name.lower() in k.lower(): return DISEASE_DB[k]
    return DEFAULT_INFO

model, class_info = load_model()
CLASS_NAMES = class_info['class_names'] if class_info else []
IMG_SIZE    = tuple(class_info['img_size']) if class_info else (224,224)

# SIDEBAR
with st.sidebar:
    st.markdown("<div style='text-align:center;padding:20px 0 10px;'><div style='font-size:55px;'>🌿</div><div style='font-size:22px;font-weight:900;margin-top:8px;'>CropGuard AI</div><div style='font-size:11px;opacity:0.8;margin-top:5px;'>Crop Disease Prediction System</div></div><hr style='border-color:rgba(255,255,255,0.2);margin:15px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:13px;line-height:1.9;opacity:0.9;'>🌾 <b>Crops Covered:</b><br>&nbsp;&nbsp;🟤 Cassava (4 classes)<br>&nbsp;&nbsp;🌽 Maize (7 classes)<br>&nbsp;&nbsp;🍠 Yam (4 classes)<br><br>🤖 <b>Best Model:</b> MobileNetV2<br>🎯 <b>Accuracy:</b> 85.13%<br>🏷️ <b>Total Classes:</b> 15</div><hr style='border-color:rgba(255,255,255,0.2);margin:15px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:12px;opacity:0.8;line-height:1.9;'>📋 <b>How to use:</b><br>1. Go to <b>Predict Disease</b> tab<br>2. Upload a leaf image<br>3. View the AI results<br>4. Follow the treatment advice</div><hr style='border-color:rgba(255,255,255,0.2);margin:15px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:11px;opacity:0.7;text-align:center;line-height:1.9;'><b>Bowen University, Iwo</b><br>Dept. of Computer Science<br><i>Ojugbeli Ogechi Esther</i><br>Supervisor: Miss Busolami<br><br>⚠️ For educational purposes only.<br>Always consult an agricultural<br>expert before any treatment.</div>", unsafe_allow_html=True)

# HEADER
st.markdown("<div style='background:linear-gradient(135deg,#1a5c2e,#2d8a4e);border-radius:20px;padding:25px 30px;color:white;margin-bottom:25px;'><div style='display:flex;align-items:center;gap:20px;'><div style='font-size:55px;'>🌿</div><div><div style='font-size:28px;font-weight:900;'>CropGuard AI</div><div style='font-size:14px;opacity:0.85;margin-top:4px;'>AI-powered crop disease detection for Cassava, Maize and Yam</div><div style='font-size:11px;opacity:0.65;margin-top:6px;'>Bowen University, Iwo &nbsp;|&nbsp; Dept. of Computer Science</div></div></div></div>", unsafe_allow_html=True)

if model is None or class_info is None:
    st.error("⚠️ Model files not found! Place `BEST_VGG16.h5` and `class_info.json` in the same folder.")
    st.stop()
else:
    st.success(f"✅ MobileNetV2 model loaded — {len(CLASS_NAMES)} disease classes ready")

tab1, tab2, tab3 = st.tabs(["🔍 Predict Disease", "📊 Model Results", "📖 Disease Guide"])

# TAB 1
with tab1:
    st.markdown("<div style='font-size:22px;font-weight:900;color:#1a1a1a;margin-bottom:5px;'>Upload Leaf Image</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:14px;color:#888;margin-bottom:20px;'>Upload a clear photo of a cassava, maize or yam leaf to detect diseases instantly.</div>", unsafe_allow_html=True)

    col_upload, col_result = st.columns([1,1], gap="large")

with col_upload:
    # uploaded = st.file_uploader("Choose a leaf image", type=['jpg','jpeg','png','bmp','webp'])
    # Let user choose input method
        input_method = st.radio(
            "How do you want to provide the leaf image?",
            ["📁 Upload from device", "📷 Take a photo now"],
            horizontal=True)

        if input_method == "📁 Upload from device":
            uploaded = st.file_uploader(
                "Choose a leaf image",
                type=['jpg','jpeg','png','bmp','webp']
            )
        else:
            uploaded = st.camera_input("Take a photo of the leaf")

if uploaded:
        img = Image.open(uploaded)
        st.image(img, use_container_width=True, caption="Uploaded leaf image")
        st.markdown(f"<div class='card' style='padding:15px; color:black;'><small>📐 Size: {img.size[0]}×{img.size[1]}px<br>🔄 Resized to: {IMG_SIZE[0]}×{IMG_SIZE[1]}px<br>📁 {uploaded.name}</small></div>", unsafe_allow_html=True)
else:
        st.markdown("<div style='background:white;border:2.5px dashed #2d8a4e;border-radius:20px;padding:50px 20px;text-align:center;color:#888;'><div style='font-size:50px;margin-bottom:15px;'>📷</div><div style='font-size:16px;font-weight:700;color:#1a5c2e;margin-bottom:8px;'>Upload a Leaf Photo</div><div style='font-size:13px;line-height:1.8;'>💡 <b>Tips:</b><br>• Good natural lighting<br>• Full leaf in the frame<br>• Focus on affected area<br>• Avoid blurry images</div></div>", unsafe_allow_html=True)

with col_result:
    if uploaded:
        with st.spinner("🔬 Analysing leaf..."):
            arr   = preprocess(img, IMG_SIZE)
            preds = model.predict(arr, verbose=0)[0]
        top_idx   = np.argmax(preds)
        top_class = CLASS_NAMES[top_idx]
        top_conf  = float(preds[top_idx]) * 100
        info      = get_info(top_class)
        top3_idx  = np.argsort(preds)[::-1][:3]

        st.markdown(f"<div class='result-box' style='background:linear-gradient(135deg,{info['color']},{info['color']}bb);'><div style='font-size:45px;margin-bottom:10px;'>{info['icon']}</div><div class='result-name'>{top_class}</div><div style='font-size:14px;opacity:0.9;'>Crop: {info['crop']}</div><div class='result-conf'>{top_conf:.1f}%</div><div class='result-conf-sub'>Confidence Score</div><div><span class='{info['pill']}'>⚡ {info['severity']} Severity</span></div></div>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Top 3 Predictions</div>", unsafe_allow_html=True)
        for i in top3_idx:
            conf = float(preds[i]) * 100
            st.markdown(f"<div class='conf-wrap'><div class='conf-row'><span>{CLASS_NAMES[i]}</span><span>{conf:.1f}%</span></div><div class='conf-track'><div class='conf-fill' style='width:{conf}%;'></div></div></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='background:white;border-radius:20px;padding:40px;text-align:center;color:#aaa;box-shadow:0 3px 15px rgba(0,0,0,0.05);'><div style='font-size:50px;margin-bottom:15px;'>🤖</div><div style='font-size:16px;font-weight:700;color:#888;'>AI Results Appear Here</div><div style='font-size:13px;margin-top:10px;color:#bbb;'>Upload a leaf image on the left<br>to get instant disease detection</div></div>", unsafe_allow_html=True)

    if uploaded and 'top_class' in dir():
        st.markdown("---")
        st.markdown(f"### 📋 Disease Report: {top_class}")
        col_a, col_b = st.columns(2, gap="large")
        with col_a:
            st.markdown(f"<div class='card'><div class='section-title'>📋 Description</div><div style='font-size:14px;color:black;line-height:1.7;'>{info['description']}</div></div>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>🚨 Symptoms</div>", unsafe_allow_html=True)
            for s in info['symptoms']: st.markdown(f"<div class='symptom-item'>⚠️ {s}</div>", unsafe_allow_html=True)
        with col_b:
            st.markdown("<div class='section-title'>💊 Recommended Treatment</div>", unsafe_allow_html=True)
            for t in info['treatment']: st.markdown(f"<div class='treatment-item'>✅ {t}</div>", unsafe_allow_html=True)
            st.markdown("<div class='card' style='border:2px solid #e8f5e9;margin-top:15px;'><div class='section-title'>📞 Need Expert Help?</div><div style='font-size:13px;color:black;line-height:1.9;'>📍 Contact your local Agricultural Extension Officer<br>📞 NASC Helpline: <b>0800-FARMER</b><br>🌐 IITA Nigeria: <b>www.iita.org</b></div></div>", unsafe_allow_html=True)

# TAB 2
with tab2:
    st.markdown("<div style='font-size:22px;font-weight:900;color:#1a1a1a;margin-bottom:5px;'>Model Performance Results</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:14px;color:#888;margin-bottom:20px;'>Comparison of all 5 models trained on the Nigerian crop disease dataset.</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>🏆 Best Model — MobileNetV2</div>", unsafe_allow_html=True)
    st.markdown("<div class='metrics-row'><div class='metric-box'><div class='metric-val'>85.13%</div><div class='metric-lbl'>ACCURACY</div></div><div class='metric-box'><div class='metric-val'>0.8513</div><div class='metric-lbl'>PRECISION</div></div><div class='metric-box'><div class='metric-val'>0.8513</div><div class='metric-lbl'>RECALL</div></div><div class='metric-box'><div class='metric-val'>0.8499</div><div class='metric-lbl'>F1-SCORE</div></div></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>📋 All 5 Models Comparison</div>", unsafe_allow_html=True)
    df = pd.DataFrame([{'Model':n,'Type':r['type'],'Accuracy':f"{r['accuracy']:.2f}%",'Precision':f"{r['precision']:.4f}",'Recall':f"{r['recall']:.4f}",'F1-Score':f"{r['f1']:.4f}",'Rank':'🥇 BEST' if n=='MobileNetV2' else ''} for n,r in MODEL_RESULTS.items()])
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("<div class='section-title'>📈 Accuracy Comparison Chart</div>", unsafe_allow_html=True)
    models = list(MODEL_RESULTS.keys())
    accs   = [MODEL_RESULTS[m]['accuracy'] for m in models]
    colors = ['#2d8a4e' if m=='MobileNetV2' else '#a8d5b5' for m in models]
    fig, ax = plt.subplots(figsize=(10,5))
    fig.patch.set_facecolor('white'); ax.set_facecolor('#f9fdf9')
    bars = ax.bar(models, accs, color=colors, edgecolor='white', linewidth=1.5, width=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold', color='#333')
    ax.set_ylim([0,100]); ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Model Performance Comparison — Test Set', fontsize=13, fontweight='bold', pad=15)
    ax.axhline(y=85.13, color='#1a5c2e', linestyle='--', alpha=0.4, linewidth=1.5)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#ddd'); ax.spines['left'].set_color('#ddd')
    plt.tight_layout(); st.pyplot(fig)

    st.markdown("<div class='section-title'>📂 Dataset Information</div>", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Images","~11,500"); c2.metric("Training","70%"); c3.metric("Validation","15%"); c4.metric("Test","15%")
    st.markdown("<div class='card'><div style='font-size:14px;color:black;line-height:2;'>🌾 <b>Crops:</b> Cassava, Maize, Yam<br>🏷️ <b>Classes:</b> 15 disease categories<br>📐 <b>Image size:</b> 224 × 224 pixels<br>🔄 <b>Augmentation:</b> Rotation, flip, zoom, brightness<br>✂️ <b>Split:</b> 70% train | 15% val | 15% test (stratified)<br>⚖️ <b>Class weights:</b> Applied to handle class imbalance</div></div>", unsafe_allow_html=True)

# TAB 3
with tab3:
    st.markdown("<div style='font-size:22px;font-weight:900;color:#1a1a1a;margin-bottom:5px;'>Disease Reference Guide</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:14px;color:black;margin-bottom:20px;'>Complete information on all 15 disease classes covered by this system.</div>", unsafe_allow_html=True)

    crop_filter = st.selectbox("Filter by crop", ["All Crops","Cassava 🟤","Maize 🌽","Yam 🍠"])
    crop_map    = {"All Crops":None,"Cassava 🟤":"Cassava","Maize 🌽":"Maize","Yam 🍠":"Yam"}
    selected    = crop_map[crop_filter]

    for disease, info in DISEASE_DB.items():
        if selected and info['crop'] != selected: continue
        emoji = {'Cassava':'🟤','Maize':'🌽','Yam':'🍠'}.get(info['crop'],'🌿')
        with st.expander(f"{info['icon']} {disease} — {emoji} {info['crop']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"<div style='background:{info['color']}15;border-radius:12px;padding:12px;margin-bottom:12px;'><span class='{info['pill']}'>{info['severity']} Severity</span><br><br><span style='font-size:13px;color:black;'>{info['description']}</span></div>", unsafe_allow_html=True)
                st.markdown("**🚨 Symptoms:**")
                for s in info['symptoms']: st.markdown(f"- ⚠️ {s}")
            with col2:
                st.markdown("**💊 Treatment:**")
                for t in info['treatment']: st.markdown(f"- ✅ {t}")

    st.markdown("<div class='card' style='text-align:center;margin-top:20px;border:2px solid #e8f5e9;'><div style='font-size:18px;font-weight:800;color:black;margin-bottom:10px;'>🌾 Important Notice</div><div style='font-size:13px;color:black;line-height:1.8;'>This guide is for educational and early detection purposes only.<br>Always consult a qualified <b>agricultural extension officer</b> for accurate diagnosis.<br>Early detection can save up to <b>80% of your crop</b>.</div></div>", unsafe_allow_html=True)

# FOOTER
st.markdown("<div style='text-align:center;padding:30px;color:#aaa;font-size:12px;margin-top:20px;border-top:1px solid #eee;'>🌿 <b>CropGuard AI</b> — Crop Disease Prediction System<br>Bowen University, Iwo &nbsp;|&nbsp; Department of Computer Science<br><i>Ojugbeli Ogechi Marvellous</i> &nbsp;|&nbsp; Supervisor: Miss Busolami Oluwadamilare</div>", unsafe_allow_html=True)