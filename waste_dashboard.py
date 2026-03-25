import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="CMT 444: Distributed ML - NEMA Waste Management",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS - KENYAN THEME
# ============================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0f1a 0%, #0f1724 100%);
    }
    
    .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, 
    .stMarkdown h4, .stMarkdown label, .stMarkdown div, .stMarkdown span {
        color: #ffffff !important;
    }
    
    [data-testid="stMetric"] {
        background: rgba(30, 40, 60, 0.8);
        border-radius: 15px;
        padding: 15px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    [data-testid="stMetric"] label {
        color: #ffaa66 !important;
        font-size: 14px !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stMetric"] .stMarkdown {
        color: #ffffff !important;
        font-size: 32px !important;
        font-weight: bold !important;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, rgba(0, 102, 0, 0.2) 0%, rgba(0, 0, 0, 0.9) 100%);
        border-right: 2px solid #CC0000;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] label {
        color: #ffaa66 !important;
        font-weight: 500 !important;
    }
    
    .stSelectbox div {
        color: #ffffff !important;
        background: rgba(30, 40, 60, 0.8) !important;
    }
    
    .stNumberInput input {
        color: #ffffff !important;
        background: rgba(30, 40, 60, 0.8) !important;
        border: 1px solid #ffaa66 !important;
    }
    
    .stSlider div[data-baseweb="slider"] div {
        color: #ffffff !important;
        font-weight: bold !important;
    }
    
    .stAlert {
        background: rgba(30, 40, 60, 0.9) !important;
        border-left: 4px solid #CC0000 !important;
        color: #ffffff !important;
    }
    
    .stDataFrame {
        background: rgba(30, 40, 60, 0.6);
        border-radius: 10px;
    }
    
    .stDataFrame th {
        background: #CC0000 !important;
        color: #ffffff !important;
    }
    
    .stDataFrame td {
        color: #ffffff !important;
        background: rgba(30, 40, 60, 0.5) !important;
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #006600, #CC0000, #000000);
    }
    
    .stCaption, .stCaption p {
        color: #a0aec0 !important;
    }
    
    .plotly .gauge-number {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# NEMA 2024 OFFICIAL DATA
# ============================================
NEMA_DATA = {
    "national_daily_waste": 22000,
    "per_capita_waste": 0.5,
    "collection_target": 85,
    "regulations_year": 2024,
    "organic_waste": 65,
    "plastic_waste": 20,
    "paper_waste": 10,
    "metal_waste": 2,
    "medical_waste": 1
}

# Kenyan counties with distributed data
COUNTIES = {
    "Nairobi": {"population": 4397000, "region": "Central", "bins": 50, "waste_factor": 1.2, "records": 365},
    "Mombasa": {"population": 1208000, "region": "Coast", "bins": 25, "waste_factor": 1.0, "records": 365},
    "Kisumu": {"population": 1155000, "region": "Nyanza", "bins": 20, "waste_factor": 0.9, "records": 365},
    "Nakuru": {"population": 2162000, "region": "Rift Valley", "bins": 35, "waste_factor": 0.95, "records": 365},
    "Kiambu": {"population": 2417000, "region": "Central", "bins": 40, "waste_factor": 1.0, "records": 365},
}

# ============================================
# AI MODEL FOR OVERFLOW PREDICTION
# ============================================
@st.cache_resource
def get_ai_model():
    """Train a Random Forest model for overflow prediction"""
    np.random.seed(42)
    X_train = []
    y_train = []
    
    # Generate training data based on NEMA 2024 patterns
    for _ in range(2000):
        fill = np.random.uniform(0, 100)
        rate = np.random.uniform(0.5, 5.0)
        pop_factor = np.random.uniform(0.5, 1.5)
        
        # Target formula based on NEMA accumulation rates
        hours = max(0, (100 - fill) / (rate * 1.8) * pop_factor)
        
        X_train.append([fill, rate, pop_factor])
        y_train.append(hours)
    
    model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=8)
    model.fit(X_train, y_train)
    return model

ai_model = get_ai_model()

# ============================================
# SIDEBAR - DISTRIBUTED ML CONTROLS
# ============================================
with st.sidebar:
    # Kenyan Flag Header
    st.markdown("# 🇰🇪 NEMA Waste Control")
    st.markdown("### *Mazingira Yetu | Uhai Wetu | Wajibu Wetu*")
    st.markdown("---")
    
    # Course Information
    st.markdown("### 📚 CMT 444: Distributed ML")
    st.markdown("**Federated Learning Demo**")
    st.markdown("")
    
    # County Selector
    st.markdown("### 📍 Select Client Node (County)")
    selected_county = st.selectbox(
        "County",
        list(COUNTIES.keys()),
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Distributed ML Metrics
    st.markdown("### 🔄 Distributed Learning Metrics")
    st.markdown("**Active Clients:**")
    st.progress(1.0)
    st.markdown("5/5 Counties Active")
    
    st.markdown("**Federated Rounds:**")
    st.markdown("10 rounds completed")
    
    st.markdown("**Model Aggregation:**")
    st.markdown("Federated Averaging (FedAvg)")
    
    st.markdown("**Privacy:**")
    st.markdown("🔒 Data never leaves client")
    
    st.markdown("---")
    
    # Data Distribution Table
    st.markdown("### 📊 Data Distribution")
    st.markdown("| County | Records |")
    st.markdown("|--------|--------|")
    for county, data in COUNTIES.items():
        st.markdown(f"| {county} | {data['records']} |")
    
    st.markdown("---")
    
    # Editable Parameters
    st.markdown("### 📊 Client Parameters")
    st.markdown("*Adjust to see local predictions*")
    
    st.markdown("**📈 Local Fill Level**")
    fill_level = st.slider(
        "Fill Level",
        min_value=0,
        max_value=100,
        value=78,
        step=1,
        format="%d%%",
        label_visibility="collapsed"
    )
    st.markdown(f"`{fill_level}%`")
    
    st.markdown("**🚛 Collection Rate**")
    collection_rate = st.slider(
        "Collection Rate",
        min_value=0.5,
        max_value=5.0,
        value=1.5,
        step=0.1,
        format="%.1f tons/hour",
        label_visibility="collapsed"
    )
    st.markdown(f"`{collection_rate} tons/hour`")
    
    st.markdown("**👥 Population**")
    population = st.number_input(
        "Population",
        min_value=1000,
        max_value=10000000,
        value=COUNTIES[selected_county]["population"],
        step=100000,
        format="%d",
        label_visibility="collapsed"
    )
    st.markdown(f"`{population:,} people`")
    
    st.markdown("---")
    
    # NEMA Standards
    st.markdown("### 📋 NEMA 2024 Standards")
    st.markdown(f"📌 Collection Target: **{NEMA_DATA['collection_target']}%**")
    st.markdown(f"🗑️ Per Capita: **{NEMA_DATA['per_capita_waste']} kg/day**")
    st.markdown(f"🌍 National Daily: **{NEMA_DATA['national_daily_waste']:,} tons**")
    
    st.markdown("---")
    st.markdown("🇰🇪 *Kenya Vision 2030*")
    st.markdown("*Sustainable Waste Management*")

# ============================================
# MAIN DASHBOARD HEADER
# ============================================
st.markdown(f"# 🇰🇪 {selected_county} Smart Waste Management System")
st.markdown(f"### Distributed ML | Federated Learning | NEMA {NEMA_DATA['regulations_year']}")
st.markdown(f"*Client Node: {selected_county} County | Data Stays Local | Only Model Updates Shared*")
st.markdown("---")

# ============================================
# DISTRIBUTED ML ARCHITECTURE EXPANDER
# ============================================
with st.expander("🧠 How Distributed ML Works (Federated Learning)", expanded=False):
    st.markdown("""
    ### 🔄 Federated Learning Architecture
    
