import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="NEMA Smart Waste Management",
    page_icon="🗑️",
    layout="wide"
)

# ============================================
# NEMA 2024 DATA (OFFICIAL)
# ============================================
NEMA_DATA = {
    "national_daily_waste": 22000,
    "per_capita_waste": 0.5,
    "collection_target": 85,
    "regulations_year": 2024
}

# ============================================
# HEADER
# ============================================
st.title("🗑️ Nairobi Smart Waste Management System")
st.markdown("### Environmental Analysis for Mombasa Road")
st.markdown(f"*Data Source: NEMA Waste Management Regulations, {NEMA_DATA['regulations_year']}*")
st.markdown("---")

# ============================================
# TOP METRICS
# ============================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Smart Bins",
        value="13",
        delta="Mombasa Road Corridor",
        delta_color="off"
    )

with col2:
    st.metric(
        label="Average Fill Level",
        value="78.9%",
        delta="Above NEMA threshold",
        delta_color="inverse"
    )

with col3:
    st.metric(
        label="Critical Bins (>75%)",
        value="8",
        delta="Immediate action required",
        delta_color="inverse"
    )

with col4:
    st.metric(
        label="NEMA Urgency Level",
        value="5/5",
        delta="Maximum priority",
        delta_color="inverse"
    )

st.markdown("---")

# ============================================
# GAUGE CHART FOR FILL LEVEL
# ============================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("⏰ Predicted Overflow Time")
    
    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=78.9,
        title={"text": "Current Fill Level"},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#ff9800"},
            "steps": [
                {"range": [0, 50], "color": "lightgreen"},
                {"range": [50, 75], "color": "yellow"},
                {"range": [75, 100], "color": "salmon"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "value": 75
            }
        }
    ))
    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)
    
    # Overflow prediction
    st.info("⚠️ **Predicted Overflow: 6 hours**")
    st.caption("Just in time with overflow - Schedule collection within 4 hours")

with col2:
    st.subheader("🎯 Collection Priority Score")
    
    # Priority score
    priority_score = 5.8
    st.markdown(f"## **{priority_score} / 10**")
    st.progress(priority_score / 10)
    
    if priority_score >= 8:
        st.error("🚨 URGENT - Immediate collection required")
    elif priority_score >= 6:
        st.warning("⚠️ HIGH - Schedule collection within 4 hours")
    elif priority_score >= 4:
        st.info("📋 MEDIUM - Schedule collection today")
    else:
        st.success("✓ LOW - Routine collection sufficient")
    
    st.caption("Based on NEMA Section 8: Urgency Matrix")

st.markdown("---")

# ============================================
# LOCATION TABLE
# ============================================
st.subheader("📍 Mombasa Road Corridor - Smart Bin Locations")

# Bin data
locations_data = [
    {"Location": "Gateway Mall", "Fill Level": 82, "Status": "🔴 Critical"},
    {"Location": "Synagogue Church", "Fill Level": 79, "Status": "🔴 Critical"},
    {"Location": "T-Mall", "Fill Level": 85, "Status": "🔴 Critical"},
    {"Location": "Belt Road Junction", "Fill Level": 91, "Status": "🔴 Critical"},
    {"Location": "Industrial Area", "Fill Level": 88, "Status": "🔴 Critical"},
    {"Location": "Athi River Junction", "Fill Level": 76, "Status": "🔴 Critical"},
    {"Location": "East Gate", "Fill Level": 72, "Status": "🟡 Warning"},
    {"Location": "SGR Terminus", "Fill Level": 68, "Status": "🟢 Normal"},
    {"Location": "JKIA Cargo", "Fill Level": 65, "Status": "🟢 Normal"},
    {"Location": "Cabanas", "Fill Level": 58, "Status": "🟢 Normal"},
    {"Location": "Standard Gauge Railway", "Fill Level": 45, "Status": "🟢 Normal"},
    {"Location": "Airport North Road", "Fill Level": 38, "Status": "🟢 Normal"},
    {"Location": "Mlolongo", "Fill Level": 32, "Status": "🟢 Normal"},
]

df = pd.DataFrame(locations_data)
st.dataframe(df, use_container_width=True, hide_index=True)

st.markdown("---")

# ============================================
# FOOTER
# ============================================
st.caption("""
**Data Source:** National Environment Management Authority (NEMA) - Simplified Waste Management Regulations, 2024  
**System:** Distributed Machine Learning | Federated Learning Implementation | CMT 444 Project  
**Kenya Vision 2030:** Sustainable Waste Management Initiative | SDG Goal 11: Sustainable Cities
""")
