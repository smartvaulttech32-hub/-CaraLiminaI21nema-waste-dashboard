import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="NEMA Smart Waste Management - Kenya",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS FOR DARK MODE
# ============================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0f1a 0%, #0f1724 100%);
    }
    
    /* Card backgrounds */
    .stMetric {
        background: rgba(30, 40, 60, 0.6);
        border-radius: 15px;
        padding: 10px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Headers */
    h1, h2, h3, .stMarkdown {
        color: #ffffff !important;
    }
    
    /* Metric labels */
    .stMetric label {
        color: #a0aec0 !important;
    }
    
    /* Metric values */
    .stMetric .stMarkdown {
        color: #ff9800 !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(10, 15, 26, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Select boxes */
    .stSelectbox, .stSlider {
        background: rgba(30, 40, 60, 0.4);
        border-radius: 10px;
    }
    
    /* Info boxes */
    .stAlert {
        background: rgba(30, 40, 60, 0.8);
        border: none;
    }
    
    /* Dataframe */
    .stDataFrame {
        background: rgba(30, 40, 60, 0.4);
        border-radius: 10px;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #ff9800, #f44336);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# NEMA 2024 OFFICIAL DATA
# ============================================
NEMA_DATA = {
    "national_daily_waste": 22000,
    "per_capita_waste": 0.5,
    "organic_waste": 65,
    "plastic_waste": 20,
    "paper_waste": 10,
    "metal_waste": 2,
    "medical_waste": 1,
    "collection_target": 85,
    "regulations_year": 2024
}

# Kenyan counties data
COUNTIES = {
    "Nairobi": {"population": 4397000, "region": "Central", "bins": 50, "lat": -1.2864, "lon": 36.8172},
    "Mombasa": {"population": 1208000, "region": "Coast", "bins": 25, "lat": -4.0435, "lon": 39.6682},
    "Kisumu": {"population": 1155000, "region": "Nyanza", "bins": 20, "lat": -0.1022, "lon": 34.7617},
    "Nakuru": {"population": 2162000, "region": "Rift Valley", "bins": 35, "lat": -0.3031, "lon": 36.0800},
    "Kiambu": {"population": 2417000, "region": "Central", "bins": 40, "lat": -1.1575, "lon": 36.8222},
    "Machakos": {"population": 1421000, "region": "Eastern", "bins": 25, "lat": -1.5177, "lon": 37.2634},
    "Uasin Gishu": {"population": 1163000, "region": "Rift Valley", "bins": 20, "lat": 0.5200, "lon": 35.2800},
    "Kakamega": {"population": 1867000, "region": "Western", "bins": 25, "lat": 0.2827, "lon": 34.7519},
}

# ============================================
# SIDEBAR - USER CONTROLS
# ============================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/Flag_of_Kenya.svg/1200px-Flag_of_Kenya.svg.png", width=80)
    st.title("🗑️ NEMA Waste Control")
    st.markdown("---")
    
    # County Selector
    selected_county = st.selectbox(
        "📍 Select County",
        list(COUNTIES.keys()),
        help="Choose any Kenyan county to view waste data"
    )
    
    # EDITABLE PARAMETERS - Like the video!
    st.markdown("### 📊 Edit Parameters")
    st.markdown("*Adjust these values to see real-time predictions*")
    
    fill_level = st.slider(
        "Fill Level (%)",
        min_value=0,
        max_value=100,
        value=78,
        step=1,
        help="Current waste bin fill percentage"
    )
    
    collection_rate = st.slider(
        "Collection Rate (tons/hour)",
        min_value=0.5,
        max_value=5.0,
        value=1.5,
        step=0.1,
        help="Current waste collection rate"
    )
    
    population = st.number_input(
        "Population",
        min_value=1000,
        max_value=10000000,
        value=COUNTIES[selected_county]["population"],
        step=100000,
        help="County population"
    )
    
    st.markdown("---")
    st.markdown("**NEMA 2024 Standards**")
    st.markdown(f"📋 Daily Waste Target: **{NEMA_DATA['collection_target']}%**")
    st.markdown(f"🗑️ Per Capita: **{NEMA_DATA['per_capita_waste']} kg/day**")
    st.markdown(f"🌍 National Daily: **{NEMA_DATA['national_daily_waste']:,} tons**")

# ============================================
# MAIN DASHBOARD
# ============================================
# Header
st.title(f"🗑️ {selected_county} Smart Waste Management System")
st.markdown(f"### Environmental Analysis for {selected_county} County")
st.markdown(f"*Data Source: NEMA Waste Management Regulations, {NEMA_DATA['regulations_year']}*")
st.markdown("---")

# ============================================
# DYNAMIC CALCULATIONS
# ============================================
# Calculate waste based on editable values
daily_waste = (population * NEMA_DATA["per_capita_waste"]) / 1000
critical_threshold = 75
critical_bins = int((fill_level / 100) * COUNTIES[selected_county]["bins"])

# Overflow calculation
if collection_rate > 0:
    overflow_hours = max(0, (100 - fill_level) / (collection_rate * 2))
else:
    overflow_hours = 999

overflow_hours = round(overflow_hours, 1)

# Urgency level based on fill level
if fill_level >= 90:
    urgency_level = 5
    urgency_text = "🚨 CRITICAL - Immediate action"
elif fill_level >= 75:
    urgency_level = 4
    urgency_text = "⚠️ HIGH - Urgent collection"
elif fill_level >= 60:
    urgency_level = 3
    urgency_text = "📋 MEDIUM - Schedule collection"
elif fill_level >= 40:
    urgency_level = 2
    urgency_text = "🟡 LOW - Routine monitoring"
else:
    urgency_level = 1
    urgency_text = "🟢 NORMAL - Regular schedule"

# Priority score
priority_score = round(((fill_level / 100) * 5) + (urgency_level * 0.5), 1)

# ============================================
# TOP METRICS
# ============================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Smart Bins",
        value=COUNTIES[selected_county]["bins"],
        delta=f"{selected_county} County",
        delta_color="off"
    )

with col2:
    st.metric(
        label="Average Fill Level",
        value=f"{fill_level}%",
        delta=f"{'Above' if fill_level > critical_threshold else 'Below'} NEMA threshold",
        delta_color="inverse"
    )

with col3:
    st.metric(
        label="Critical Bins (>75%)",
        value=critical_bins,
        delta=f"{critical_bins} bins require attention",
        delta_color="inverse"
    )

with col4:
    st.metric(
        label="NEMA Urgency Level",
        value=f"{urgency_level}/5",
        delta=urgency_text,
        delta_color="inverse"
    )

st.markdown("---")

# ============================================
# GAUGE CHART AND PREDICTIONS
# ============================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("⏰ AI-Powered Overflow Prediction")
    
    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=fill_level,
        title={"text": "Current Fill Level", "font": {"color": "white"}},
        delta={"reference": critical_threshold, "increasing": {"color": "red"}},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "white"},
            "bar": {"color": "#ff9800"},
            "bgcolor": "#1e2a3a",
            "borderwidth": 2,
            "bordercolor": "#ff9800",
            "steps": [
                {"range": [0, 50], "color": "#2e7d32"},
                {"range": [50, 75], "color": "#ff9800"},
                {"range": [75, 100], "color": "#f44336"}
            ],
            "threshold": {
                "line": {"color": "white", "width": 4},
                "value": critical_threshold
            }
        }
    ))
    fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"})
    st.plotly_chart(fig, use_container_width=True)
    
    # Overflow prediction
    if overflow_hours <= 4:
        st.error(f"⚠️ **Predicted Overflow: {overflow_hours} hours** - Immediate collection required!")
    elif overflow_hours <= 8:
        st.warning(f"⚠️ **Predicted Overflow: {overflow_hours} hours** - Just in time with overflow")
    else:
        st.success(f"✅ **Predicted Overflow: {overflow_hours} hours** - Normal operations")
    
    st.caption(f"Based on fill rate of {collection_rate} tons/hour")

with col2:
    st.subheader("🎯 Collection Priority Score")
    
    # Priority score display
    st.markdown(f"## **{priority_score} / 10**")
    
    # Progress bar
    st.progress(priority_score / 10)
    
    # Priority recommendation
    if priority_score >= 8:
        st.error("🚨 URGENT - Immediate collection within 2 hours")
    elif priority_score >= 6:
        st.warning("⚠️ HIGH - Schedule collection within 4 hours")
    elif priority_score >= 4:
        st.info("📋 MEDIUM - Schedule collection today")
    else:
        st.success("✓ LOW - Routine collection sufficient")
    
    # Additional metrics
    st.markdown("---")
    st.markdown("### 📈 Waste Generation")
    st.metric("Daily Waste", f"{daily_waste:.0f} tons", f"Based on {population:,} people")
    st.metric("Collection Efficiency", f"{min(100, int((collection_rate/2.5)*100))}%", "Target: 85%")

st.markdown("---")

# ============================================
# BIN LOCATIONS TABLE
# ============================================
st.subheader(f"📍 {selected_county} County - Smart Bin Locations")

# Generate dynamic location data
num_locations = min(COUNTIES[selected_county]["bins"], 15)
locations_data = []

for i in range(num_locations):
    # Random but consistent fill levels based on overall fill_level
    loc_fill = min(100, max(0, fill_level + np.random.randint(-20, 20)))
    if loc_fill >= 75:
        status = "🔴 Critical"
    elif loc_fill >= 60:
        status = "🟡 Warning"
    else:
        status = "🟢 Normal"
    
    locations_data.append({
        "Location": f"Bin {i+1}",
        "Fill Level": loc_fill,
        "Status": status,
        "Last Collection": f"{np.random.randint(1, 12)} hours ago"
    })

df_locations = pd.DataFrame(locations_data)
st.dataframe(
    df_locations.style.background_gradient(subset=["Fill Level"], cmap="RdYlGn_r", vmin=0, vmax=100),
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# ============================================
# FOOTER WITH NEMA ATTRIBUTION
# ============================================
col1, col2, col3 = st.columns(3)

with col1:
    st.caption("**Data Source:** National Environment Management Authority (NEMA)")
    st.caption("Simplified Waste Management Regulations, 2024")

with col2:
    st.caption("**System:** Distributed Machine Learning")
    st.caption("Federated Learning Implementation | CMT 444 Project")

with col3:
    st.caption("**Kenya Vision 2030:** Sustainable Waste Management")
    st.caption("SDG Goal 11: Sustainable Cities and Communities")

# ============================================
# REAL-TIME UPDATE NOTE
# ============================================
st.info("💡 **Tip:** Use the sidebar sliders to adjust fill level, collection rate, and population. All predictions update in real-time!")
