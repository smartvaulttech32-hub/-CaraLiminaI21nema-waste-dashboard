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
# CUSTOM CSS FOR BETTER VISIBILITY
# ============================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0f1a 0%, #0f1724 100%);
    }
    
    /* ALL TEXT - Make everything white and visible */
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
    
    /* Metric labels - make them bright */
    [data-testid="stMetric"] label {
        color: #ffaa66 !important;
        font-size: 14px !important;
        font-weight: 500 !important;
    }
    
    /* Metric values - LARGE and BRIGHT */
    [data-testid="stMetric"] .stMarkdown {
        color: #ff9800 !important;
        font-size: 32px !important;
        font-weight: bold !important;
    }
    
    /* Metric delta - visible */
    [data-testid="stMetric"] .stMarkdown small {
        color: #a0aec0 !important;
        font-size: 12px !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(10, 15, 26, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.2);
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
        border: 1px solid #ff9800 !important;
    }
    
    /* Slider labels */
    .stSlider label {
        color: #ffaa66 !important;
        font-weight: 500 !important;
    }
    
    /* Slider value display */
    .stSlider div[data-baseweb="slider"] div {
        color: #ff9800 !important;
        font-weight: bold !important;
    }
    
    /* Info boxes */
    .stAlert {
        background: rgba(30, 40, 60, 0.9) !important;
        border-left: 4px solid #ff9800 !important;
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
        background: #ff9800 !important;
        color: #000000 !important;
        font-weight: bold !important;
    }
    
    .stDataFrame td {
        color: #ffffff !important;
        background: rgba(30, 40, 60, 0.5) !important;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #ff9800, #f44336);
    }
    
    /* Caption text */
    .stCaption, .stCaption p {
        color: #a0aec0 !important;
    }
    
    /* Metric delta */
    .stMetricDelta {
        color: #ffaa66 !important;
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

# Kenyan counties data with more details
COUNTIES = {
    "Nairobi": {"population": 4397000, "region": "Central", "bins": 50, "waste_factor": 1.2},
    "Mombasa": {"population": 1208000, "region": "Coast", "bins": 25, "waste_factor": 1.0},
    "Kisumu": {"population": 1155000, "region": "Nyanza", "bins": 20, "waste_factor": 0.9},
    "Nakuru": {"population": 2162000, "region": "Rift Valley", "bins": 35, "waste_factor": 0.95},
    "Kiambu": {"population": 2417000, "region": "Central", "bins": 40, "waste_factor": 1.0},
}

# ============================================
# SIDEBAR - USER CONTROLS
# ============================================
with st.sidebar:
    st.markdown("# 🗑️ NEMA Waste Control Panel")
    st.markdown("---")
    
    # County selector with visible label
    st.markdown("### 📍 Select County")
    selected_county = st.selectbox(
        "County",
        list(COUNTIES.keys()),
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Editable Parameters with visible labels
    st.markdown("### 📊 Edit Parameters")
    st.markdown("*Adjust these values to see real-time predictions*")
    
    # Fill level slider with visible percentage
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
    
    # Population input with visible label
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
    st.markdown("💡 **Tip:** Adjust the sliders above to see how predictions change in real-time!")

# ============================================
# MAIN DASHBOARD
# ============================================
# Header with visible title
st.markdown(f"# 🗑️ {selected_county} Smart Waste Management System")
st.markdown(f"### Environmental Analysis for {selected_county} County")
st.markdown(f"*Data Source: NEMA Waste Management Regulations, {NEMA_DATA['regulations_year']}*")
st.markdown("---")

# ============================================
# DYNAMIC CALCULATIONS
# ============================================
# Calculate waste generation
daily_waste = (population * NEMA_DATA["per_capita_waste"]) / 1000
critical_threshold = 75
critical_bins = int((fill_level / 100) * COUNTIES[selected_county]["bins"])

# Overflow calculation
if collection_rate > 0:
    overflow_hours = max(0, (100 - fill_level) / (collection_rate * 2))
else:
    overflow_hours = 999
overflow_hours = round(overflow_hours, 1)

# Urgency level calculation
if fill_level >= 90:
    urgency_level = 5
    urgency_text = "🚨 CRITICAL - Immediate action required"
    urgency_color = "#f44336"
elif fill_level >= 75:
    urgency_level = 4
    urgency_text = "⚠️ HIGH - Urgent collection needed"
    urgency_color = "#ff9800"
elif fill_level >= 60:
    urgency_level = 3
    urgency_text = "📋 MEDIUM - Schedule collection today"
    urgency_color = "#ffc107"
elif fill_level >= 40:
    urgency_level = 2
    urgency_text = "🟡 LOW - Routine monitoring"
    urgency_color = "#90be6d"
else:
    urgency_level = 1
    urgency_text = "🟢 NORMAL - Regular schedule"
    urgency_color = "#4caf50"

# Priority score calculation
priority_score = round(((fill_level / 100) * 5) + (urgency_level * 0.5), 1)

# ============================================
# TOP METRICS - VISIBLE PERCENTAGES
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
        delta=f"{'▲ Above' if fill_level > critical_threshold else '▼ Below'} {critical_threshold}% threshold",
        delta_color="inverse"
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
        delta=urgency_text,
        delta_color="inverse"
    )

st.markdown("---")

# ============================================
# GAUGE CHART AND PREDICTIONS
# ============================================
col1, col2 = st.columns(2)

with col1:
    st.markdown("### ⏰ AI-Powered Overflow Prediction")
    
    # Gauge chart with visible percentage
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fill_level,
        title={"text": f"Current Fill Level: {fill_level}%", "font": {"color": "white", "size": 16}},
        number={"font": {"color": "#ff9800", "size": 40}, "suffix": "%"},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "white", "tickwidth": 2, "tickfont": {"color": "white"}},
            "bar": {"color": "#ff9800", "thickness": 0.8},
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
    
    # Overflow prediction with visible hours
    if overflow_hours <= 4:
        st.error(f"⚠️ **Predicted Overflow: {overflow_hours} hours**")
        st.markdown(f"`Current fill rate: {collection_rate} tons/hour | Fill level: {fill_level}%`")
        st.markdown("🚨 **Immediate collection required!**")
    elif overflow_hours <= 8:
        st.warning(f"⚠️ **Predicted Overflow: {overflow_hours} hours**")
        st.markdown(f"`Current fill rate: {collection_rate} tons/hour | Fill level: {fill_level}%`")
        st.markdown("⏰ **Just in time with overflow - Schedule collection within 4 hours**")
    else:
        st.success(f"✅ **Predicted Overflow: {overflow_hours} hours**")
        st.markdown(f"`Current fill rate: {collection_rate} tons/hour | Fill level: {fill_level}%`")
        st.markdown("📋 **Normal operations - Routine monitoring sufficient**")

with col2:
    st.markdown("### 🎯 Collection Priority Score")
    
    # Large priority score display
    st.markdown(f"# **{priority_score} / 10**")
    st.progress(priority_score / 10)
    
    # Priority recommendation
    if priority_score >= 8:
        st.error("🚨 **URGENT** - Immediate collection within 2 hours")
        st.markdown("`Priority: Maximum | Response: Emergency dispatch`")
    elif priority_score >= 6:
        st.warning("⚠️ **HIGH** - Schedule collection within 4 hours")
        st.markdown("`Priority: High | Response: Priority dispatch`")
    elif priority_score >= 4:
        st.info("📋 **MEDIUM** - Schedule collection today")
        st.markdown("`Priority: Medium | Response: Standard dispatch`")
    else:
        st.success("✓ **LOW** - Routine collection sufficient")
        st.markdown("`Priority: Low | Response: Regular schedule`")
    
    st.markdown("---")
    st.markdown("### 📈 Waste Generation Metrics")
    
    # Waste metrics with visible percentages
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Daily Waste", f"{daily_waste:.0f} tons", f"{daily_waste/population*1000:.2f} kg/person")
    with col_b:
        efficiency = min(100, int((collection_rate/2.5)*100))
        st.metric("Collection Efficiency", f"{efficiency}%", f"Target: {NEMA_DATA['collection_target']}%")
    
    st.markdown(f"**Population:** {population:,} people")
    st.markdown(f"**Waste Factor:** {COUNTIES[selected_county]['waste_factor']:.1f}x")
    st.markdown(f"**Region:** {COUNTIES[selected_county]['region']}")

st.markdown("---")

# ============================================
# BIN LOCATIONS TABLE - VISIBLE STATUS
# ============================================
st.markdown(f"### 📍 {selected_county} County - Smart Bin Locations")

num_locations = min(COUNTIES[selected_county]["bins"], 15)
locations_data = []

for i in range(num_locations):
    # Generate fill levels based on overall fill_level
    loc_fill = min(100, max(0, fill_level + np.random.randint(-20, 20)))
    
    if loc_fill >= 75:
        status = "🔴 CRITICAL"
        status_class = "critical"
    elif loc_fill >= 60:
        status = "🟡 WARNING"
        status_class = "warning"
    else:
        status = "🟢 NORMAL"
        status_class = "normal"
    
    locations_data.append({
        "Bin ID": f"BIN-{i+1:03d}",
        "Location": f"Site {i+1}",
        "Fill Level (%)": loc_fill,
        "Status": status,
        "Status Indicator": "⚠️" if loc_fill >= 75 else "🟢" if loc_fill < 60 else "🟡"
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
        ),
        "Status Indicator": st.column_config.TextColumn(
            "Alert",
            help="Alert indicator",
            width="small"
        )
    }
)

st.markdown("---")

# ============================================
# FOOTER WITH NEMA ATTRIBUTION
# ============================================
col1, col2, col3 = st.columns(3)

with col1:
    st.caption("**📋 Data Source:**")
    st.caption("National Environment Management Authority (NEMA)")
    st.caption("Simplified Waste Management Regulations, 2024")

with col2:
    st.caption("**⚙️ System:**")
    st.caption("Distributed Machine Learning")
    st.caption("Federated Learning | CMT 444 Project")

with col3:
    st.caption("**🌍 Kenya Vision 2030:**")
    st.caption("Sustainable Waste Management")
    st.caption("SDG Goal 11: Sustainable Cities")

st.markdown("---")

# ============================================
# INTERACTIVE TIP
# ============================================
st.info("💡 **Tip:** Use the sidebar sliders to adjust fill level, collection rate, and population. All predictions update in real-time! Notice how the overflow time and priority score change as you adjust the values.")
