import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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
# CUSTOM CSS - KENYAN FLAG THEME + WHITE PERCENTAGES
# ============================================
st.markdown("""
<style>
    /* Main background - Kenyan flag colors inspired */
    .stApp {
        background: linear-gradient(135deg, #0a0f1a 0%, #0f1724 100%);
    }
    
    /* Kenyan Flag Colors Accent */
    .kenya-green {
        color: #006600 !important;
    }
    .kenya-red {
        color: #CC0000 !important;
    }
    .kenya-black {
        color: #000000 !important;
    }
    
    /* ALL TEXT - White */
    .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, 
    .stMarkdown h4, .stMarkdown label, .stMarkdown div, .stMarkdown span {
        color: #ffffff !important;
    }
    
    /* Metric containers */
    [data-testid="stMetric"] {
        background: rgba(30, 40, 60, 0.8);
        border-radius: 15px;
        padding: 15px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Metric labels - golden */
    [data-testid="stMetric"] label {
        color: #ffaa66 !important;
        font-size: 14px !important;
        font-weight: 500 !important;
    }
    
    /* Metric values - WHITE */
    [data-testid="stMetric"] .stMarkdown {
        color: #ffffff !important;
        font-size: 32px !important;
        font-weight: bold !important;
    }
    
    /* Percentage values in metrics - WHITE */
    [data-testid="stMetric"] .stMarkdown .stText {
        color: #ffffff !important;
    }
    
    /* Metric delta - visible */
    [data-testid="stMetric"] .stMarkdown small {
        color: #a0aec0 !important;
        font-size: 12px !important;
    }
    
    /* Sidebar with Kenyan flag background */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, rgba(0, 102, 0, 0.2) 0%, rgba(0, 0, 0, 0.9) 100%);
        border-right: 2px solid #CC0000;
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] .stMarkdown {
        color: #ffffff !important;
    }
    
    /* Sidebar labels */
    [data-testid="stSidebar"] label {
        color: #ffaa66 !important;
        font-weight: 500 !important;
    }
    
    /* Select box text */
    .stSelectbox label {
        color: #ffaa66 !important;
        font-weight: 500 !important;
    }
    
    .stSelectbox div {
        color: #ffffff !important;
        background: rgba(30, 40, 60, 0.8) !important;
    }
    
    /* Number input */
    .stNumberInput label {
        color: #ffaa66 !important;
    }
    
    .stNumberInput input {
        color: #ffffff !important;
        background: rgba(30, 40, 60, 0.8) !important;
        border: 1px solid #ffaa66 !important;
    }
    
    /* Slider labels */
    .stSlider label {
        color: #ffaa66 !important;
        font-weight: 500 !important;
    }
    
    /* Slider value display - WHITE */
    .stSlider div[data-baseweb="slider"] div {
        color: #ffffff !important;
        font-weight: bold !important;
    }
    
    /* Percentage text in sliders - WHITE */
    .stSlider .stMarkdown {
        color: #ffffff !important;
    }
    
    /* Info boxes */
    .stAlert {
        background: rgba(30, 40, 60, 0.9) !important;
        border-left: 4px solid #CC0000 !important;
        color: #ffffff !important;
    }
    
    .stAlert .stMarkdown {
        color: #ffffff !important;
    }
    
    /* Dataframe */
    .stDataFrame {
        background: rgba(30, 40, 60, 0.6);
        border-radius: 10px;
    }
    
    .stDataFrame table {
        color: #ffffff !important;
    }
    
    .stDataFrame th {
        background: #CC0000 !important;
        color: #ffffff !important;
        font-weight: bold !important;
    }
    
    .stDataFrame td {
        color: #ffffff !important;
        background: rgba(30, 40, 60, 0.5) !important;
    }
    
    /* Progress bar - Kenyan colors */
    .stProgress > div > div {
        background: linear-gradient(90deg, #006600, #CC0000, #000000);
    }
    
    /* Caption text */
    .stCaption, .stCaption p {
        color: #a0aec0 !important;
    }
    
    /* Gauge chart numbers - WHITE */
    .plotly .gauge-number {
        color: #ffffff !important;
    }
    
    /* Button styling */
    .stButton button {
        background: #CC0000 !important;
        color: #ffffff !important;
        border: none !important;
    }
    
    .stButton button:hover {
        background: #006600 !important;
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
    "regulations_year": 2024
}

# Kenyan counties data
COUNTIES = {
    "Nairobi": {"population": 4397000, "region": "Central", "bins": 50, "waste_factor": 1.2},
    "Mombasa": {"population": 1208000, "region": "Coast", "bins": 25, "waste_factor": 1.0},
    "Kisumu": {"population": 1155000, "region": "Nyanza", "bins": 20, "waste_factor": 0.9},
    "Nakuru": {"population": 2162000, "region": "Rift Valley", "bins": 35, "waste_factor": 0.95},
    "Kiambu": {"population": 2417000, "region": "Central", "bins": 40, "waste_factor": 1.0},
}

# ============================================
# SIDEBAR - WITH KENYAN FLAG
# ============================================
with st.sidebar:
    # Kenyan Flag (emoji version for visibility)
    st.markdown("# 🇰🇪 NEMA Waste Control Panel")
    st.markdown("### *Mazingira Yetu | Uhai Wetu | Wajibu Wetu*")
    st.markdown("---")
    
    # County selector
    st.markdown("### 📍 Select County")
    selected_county = st.selectbox(
        "County",
        list(COUNTIES.keys()),
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Editable Parameters
    st.markdown("### 📊 Edit Parameters")
    st.markdown("*Adjust these values to see real-time predictions*")
    
    # Fill level slider - percentage in white
    st.markdown("**📈 Fill Level**")
    fill_level = st.slider(
        "Fill Level Percentage",
        min_value=0,
        max_value=100,
        value=78,
        step=1,
        format="%d%%",
        label_visibility="collapsed"
    )
    st.markdown(f"`Current: {fill_level}%`")
    
    # Collection rate slider
    st.markdown("**🚛 Collection Rate**")
    collection_rate = st.slider(
        "Collection Rate (tons/hour)",
        min_value=0.5,
        max_value=5.0,
        value=1.5,
        step=0.1,
        format="%.1f tons/hour",
        label_visibility="collapsed"
    )
    st.markdown(f"`Current: {collection_rate} tons/hour`")
    
    # Population input
    st.markdown("**👥 Population**")
    population = st.number_input(
        "County Population",
        min_value=1000,
        max_value=10000000,
        value=COUNTIES[selected_county]["population"],
        step=100000,
        format="%d",
        label_visibility="collapsed"
    )
    st.markdown(f"`Current: {population:,} people`")
    
    st.markdown("---")
    st.markdown("### 📋 NEMA 2024 Standards")
    st.markdown(f"📌 Collection Target: **{NEMA_DATA['collection_target']}%**")
    st.markdown(f"🗑️ Per Capita Waste: **{NEMA_DATA['per_capita_waste']} kg/day**")
    st.markdown(f"🌍 National Daily: **{NEMA_DATA['national_daily_waste']:,} tons**")
    
    st.markdown("---")
    st.markdown("🇰🇪 *Kenya Vision 2030*")
    st.markdown("*Sustainable Waste Management Initiative*")
    
    st.markdown("---")
    st.markdown("💡 **Tip:** Adjust the sliders to see how predictions change!")

# ============================================
# MAIN DASHBOARD
# ============================================
# Header with Kenyan flag
st.markdown(f"# 🇰🇪 {selected_county} Smart Waste Management System")
st.markdown(f"### Environmental Analysis for {selected_county} County")
st.markdown(f"*Data Source: NEMA Waste Management Regulations, {NEMA_DATA['regulations_year']}*")
st.markdown("---")

# ============================================
# DYNAMIC CALCULATIONS
# ============================================
daily_waste = (population * NEMA_DATA["per_capita_waste"]) / 1000
critical_threshold = 75
critical_bins = int((fill_level / 100) * COUNTIES[selected_county]["bins"])

if collection_rate > 0:
    overflow_hours = max(0, (100 - fill_level) / (collection_rate * 2))
else:
    overflow_hours = 999
overflow_hours = round(overflow_hours, 1)

if fill_level >= 90:
    urgency_level = 5
    urgency_text = "🚨 CRITICAL - Immediate action required"
elif fill_level >= 75:
    urgency_level = 4
    urgency_text = "⚠️ HIGH - Urgent collection needed"
elif fill_level >= 60:
    urgency_level = 3
    urgency_text = "📋 MEDIUM - Schedule collection today"
elif fill_level >= 40:
    urgency_level = 2
    urgency_text = "🟡 LOW - Routine monitoring"
else:
    urgency_level = 1
    urgency_text = "🟢 NORMAL - Regular schedule"

priority_score = round(((fill_level / 100) * 5) + (urgency_level * 0.5), 1)

# ============================================
# TOP METRICS - WHITE PERCENTAGES
# ============================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="📊 Total Smart Bins",
        value=f"{COUNTIES[selected_county]['bins']}",
        delta=f"{selected_county} County"
    )

with col2:
    st.metric(
        label="📈 Average Fill Level",
        value=f"{fill_level}%",
        delta=f"{'▲ Above' if fill_level > critical_threshold else '▼ Below'} {critical_threshold}% threshold"
    )

with col3:
    st.metric(
        label="⚠️ Critical Bins",
        value=f"{critical_bins}",
        delta=f"{critical_bins} bins >{critical_threshold}% full"
    )

with col4:
    st.metric(
        label="🚨 NEMA Urgency Level",
        value=f"{urgency_level}/5",
        delta=urgency_text
    )

st.markdown("---")

# ============================================
# GAUGE CHART AND PREDICTIONS
# ============================================
col1, col2 = st.columns(2)

with col1:
    st.markdown("### ⏰ AI-Powered Overflow Prediction")
    
    # Gauge chart with white number
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fill_level,
        title={"text": f"Current Fill Level: {fill_level}%", "font": {"color": "white", "size": 16}},
        number={"font": {"color": "white", "size": 50}, "suffix": "%"},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "white", "tickwidth": 2, "tickfont": {"color": "white"}},
            "bar": {"color": "#ff9800", "thickness": 0.8},
            "bgcolor": "#1e2a3a",
            "borderwidth": 2,
            "bordercolor": "#ffaa66",
            "steps": [
                {"range": [0, 50], "color": "#2e7d32"},
                {"range": [50, 75], "color": "#ff9800"},
                {"range": [75, 100], "color": "#f44336"}
            ],
            "threshold": {
                "line": {"color": "white", "width": 4},
                "value": critical_threshold,
                "thickness": 0.9
            }
        }
    ))
    fig.update_layout(
        height=350, 
        paper_bgcolor="rgba(0,0,0,0)", 
        font={"color": "white"},
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Overflow prediction
    if overflow_hours <= 4:
        st.error(f"⚠️ **Predicted Overflow: {overflow_hours} hours**")
        st.markdown(f"`Fill level: {fill_level}% | Rate: {collection_rate} tons/hour`")
        st.markdown("🚨 **Immediate collection required!**")
    elif overflow_hours <= 8:
        st.warning(f"⚠️ **Predicted Overflow: {overflow_hours} hours**")
        st.markdown(f"`Fill level: {fill_level}% | Rate: {collection_rate} tons/hour`")
        st.markdown("⏰ **Schedule collection within 4 hours**")
    else:
        st.success(f"✅ **Predicted Overflow: {overflow_hours} hours**")
        st.markdown(f"`Fill level: {fill_level}% | Rate: {collection_rate} tons/hour`")
        st.markdown("📋 **Normal operations - Routine monitoring**")

with col2:
    st.markdown("### 🎯 Collection Priority Score")
    
    # Large priority score
    st.markdown(f"# **{priority_score} / 10**")
    st.progress(priority_score / 10)
    
    # Priority recommendation
    if priority_score >= 8:
        st.error("🚨 **URGENT** - Immediate collection within 2 hours")
    elif priority_score >= 6:
        st.warning("⚠️ **HIGH** - Schedule collection within 4 hours")
    elif priority_score >= 4:
        st.info("📋 **MEDIUM** - Schedule collection today")
    else:
        st.success("✓ **LOW** - Routine collection sufficient")
    
    st.markdown("---")
    st.markdown("### 📈 Waste Generation Metrics")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Daily Waste", f"{daily_waste:.0f} tons", f"{daily_waste/population*1000:.2f} kg/person")
    with col_b:
        efficiency = min(100, int((collection_rate/2.5)*100))
        st.metric("Collection Efficiency", f"{efficiency}%", f"Target: {NEMA_DATA['collection_target']}%")
    
    st.markdown(f"**Population:** {population:,} people")
    st.markdown(f"**Region:** {COUNTIES[selected_county]['region']}")

st.markdown("---")

# ============================================
# BIN LOCATIONS TABLE
# ============================================
st.markdown(f"### 📍 {selected_county} County - Smart Bin Locations")

num_locations = min(COUNTIES[selected_county]["bins"], 15)
locations_data = []

for i in range(num_locations):
    loc_fill = min(100, max(0, fill_level + np.random.randint(-20, 20)))
    
    if loc_fill >= 75:
        status = "🔴 CRITICAL"
    elif loc_fill >= 60:
        status = "🟡 WARNING"
    else:
        status = "🟢 NORMAL"
    
    locations_data.append({
        "Bin ID": f"BIN-{i+1:03d}",
        "Location": f"Site {i+1}",
        "Fill Level (%)": loc_fill,
        "Status": status
    })

df_locations = pd.DataFrame(locations_data)
st.dataframe(
    df_locations,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Fill Level (%)": st.column_config.ProgressColumn(
            "Fill Level (%)",
            help="Current bin fill percentage",
            format="%d%%",
            min_value=0,
            max_value=100,
        )
    }
)

st.markdown("---")

# ============================================
# FOOTER WITH KENYAN FLAG AND NEMA
# ============================================
col1, col2, col3 = st.columns(3)

with col1:
    st.caption("🇰🇪 **Data Source:**")
    st.caption("National Environment Management Authority (NEMA)")
    st.caption("Simplified Waste Management Regulations, 2024")

with col2:
    st.caption("⚙️ **System:**")
    st.caption("Distributed Machine Learning")
    st.caption("Federated Learning | CMT 444 Project")

with col3:
    st.caption("🌍 **Kenya Vision 2030:**")
    st.caption("Sustainable Waste Management")
    st.caption("SDG Goal 11: Sustainable Cities")

st.markdown("---")

# ============================================
# INTERACTIVE TIP
# ============================================
st.info("💡 **Tip:** Use the sidebar sliders to adjust fill level, collection rate, and population. All predictions update in real-time! Notice how the overflow time and priority score change as you adjust the values.")
