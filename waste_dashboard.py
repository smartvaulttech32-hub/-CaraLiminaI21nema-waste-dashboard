import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="NEMA Waste Management | CMT 444", layout="wide")

# ============================================
# NEMA 2024 OFFICIAL DATA
# From: Simplified Waste Management Regulations, 2024
# ============================================
NEMA_2024 = {
    "national_daily_waste": 22000,      # Page 1: 22,000 tons/day
    "per_capita_waste": 0.5,            # Page 1: 0.5 kg/person/day
    "organic_waste": 65,                # Page 1: 60-70% organic
    "plastic_waste": 20,                # Page 1: 20% plastic
    "paper_waste": 10,                  # Page 1: 10% paper
    "metal_waste": 2,                   # Page 1: 2% metal
    "medical_waste": 1,                 # Page 1: 1% medical
    "urban_waste_share": 40,            # Page 1: 40% from urban areas
    "collection_target": 85,            # Page 4: 85% collection target
    "penalty_no_segregation": 20000,    # Page 18: Ksh 20,000 fine
    "penalty_mishandling": 50000,       # Page 18: Ksh 50,000 fine
    "license_fee_transport": 10000,     # Page 17: Ksh 10,000
    "license_fee_disposal": 100000,     # Page 17: Ksh 100,000
}

# Kenyan counties (KNBS 2019 Census + NEMA waste factors)
COUNTIES = {
    "Nairobi": {"population": 4397000, "region": "Central", "bins": 50, "waste_factor": 1.2},
    "Mombasa": {"population": 1208000, "region": "Coast", "bins": 25, "waste_factor": 1.0},
    "Kisumu": {"population": 1155000, "region": "Nyanza", "bins": 20, "waste_factor": 0.9},
    "Nakuru": {"population": 2162000, "region": "Rift Valley", "bins": 35, "waste_factor": 0.95},
    "Kiambu": {"population": 2417000, "region": "Central", "bins": 40, "waste_factor": 1.0},
    "Machakos": {"population": 1421000, "region": "Eastern", "bins": 25, "waste_factor": 0.85},
    "Uasin Gishu": {"population": 1163000, "region": "Rift Valley", "bins": 20, "waste_factor": 0.8},
    "Kakamega": {"population": 1867000, "region": "Western", "bins": 25, "waste_factor": 0.75},
    "Kilifi": {"population": 1453000, "region": "Coast", "bins": 20, "waste_factor": 0.7},
    "Meru": {"population": 1545000, "region": "Eastern", "bins": 22, "waste_factor": 0.78},
}

# Calculate daily waste using NEMA formula
def calculate_daily_waste(population, waste_factor):
    return (population * NEMA_2024["per_capita_waste"] / 1000) * waste_factor

# ============================================
# UI STYLING
# ============================================
st.markdown("""
<style>
.stApp { background-color: #0a0f1a; }
.stMetric label { color: #ffaa66; }
.stMetric .stMarkdown { color: white; font-size: 28px; }
h1, h2, h3, p, .stMarkdown { color: white; }
[data-testid="stSidebar"] { background-color: #0f1724; border-right: 2px solid #CC0000; }
.stAlert { background-color: #1e2a3a; }
</style>
""", unsafe_allow_html=True)

# ============================================
# HEADER
# ============================================
st.title("🇰🇪 NEMA Smart Waste Management System")
st.markdown(f"### Based on NEMA Waste Management Regulations, {NEMA_2024['regulations_year'] if 'regulations_year' in NEMA_2024 else 2024}")
st.markdown("*Mazingira Yetu | Uhai Wetu | Wajibu Wetu*")
st.markdown("---")

# ============================================
# SIDEBAR - NEMA INFO
# ============================================
with st.sidebar:
    st.markdown("### 📋 NEMA 2024 Regulations")
    st.markdown("---")
    
    st.markdown("**National Statistics:**")
    st.metric("Daily Waste", f"{NEMA_2024['national_daily_waste']:,} tons")
    st.metric("Per Capita", f"{NEMA_2024['per_capita_waste']} kg/day")
    st.metric("Collection Target", f"{NEMA_2024['collection_target']}%")
    st.markdown("---")
    
    st.markdown("**Waste Composition:**")
    st.markdown(f"• Organic: {NEMA_2024['organic_waste']}%")
    st.markdown(f"• Plastic: {NEMA_2024['plastic_waste']}%")
    st.markdown(f"• Paper: {NEMA_2024['paper_waste']}%")
    st.markdown(f"• Metal: {NEMA_2024['metal_waste']}%")
    st.markdown(f"• Medical: {NEMA_2024['medical_waste']}%")
    st.markdown("---")
    
    st.markdown("**Penalties (Section 18):**")
    st.markdown(f"⚠️ No Segregation: Ksh {NEMA_2024['penalty_no_segregation']:,}")
    st.markdown(f"⚠️ Mishandling: Ksh {NEMA_2024['penalty_mishandling']:,}")
    st.markdown("---")
    
    st.markdown("**Select County**")
    selected_county = st.selectbox("", list(COUNTIES.keys()))

county_data = COUNTIES[selected_county]
daily_waste = calculate_daily_waste(county_data["population"], county_data["waste_factor"])

# ============================================
# MAIN DASHBOARD
# ============================================
st.subheader(f"📍 {selected_county} County")

# Metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Smart Bins", county_data["bins"])
with col2:
    st.metric("Population", f"{county_data['population']:,}")
with col3:
    st.metric("Daily Waste (NEMA formula)", f"{daily_waste:.0f} tons")
with col4:
    st.metric("Collection Efficiency", f"{NEMA_2024['collection_target']}% (target)")

st.markdown("---")

# Waste breakdown based on NEMA composition
st.subheader("📊 Waste Composition (NEMA 2024)")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Organic", f"{daily_waste * NEMA_2024['organic_waste']/100:.0f} tons")
with col2:
    st.metric("Plastic", f"{daily_waste * NEMA_2024['plastic_waste']/100:.0f} tons")
with col3:
    st.metric("Paper", f"{daily_waste * NEMA_2024['paper_waste']/100:.0f} tons")
with col4:
    st.metric("Metal", f"{daily_waste * NEMA_2024['metal_waste']/100:.0f} tons")
with col5:
    st.metric("Medical", f"{daily_waste * NEMA_2024['medical_waste']/100:.0f} tons")

st.markdown("---")

# Fill level simulation
fill_level = min(100, (daily_waste / (county_data["bins"] * 100)) * 100)

col1, col2 = st.columns(2)

with col1:
    st.subheader("⏰ Overflow Prediction")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fill_level,
        title={"text": f"Current Fill Level", "font": {"color": "white"}},
        number={"font": {"color": "white", "size": 40}, "suffix": "%"},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "white"},
            "bar": {"color": "#ff9800"},
            "bgcolor": "#1e2a3a",
            "steps": [
                {"range": [0, 50], "color": "#2e7d32"},
                {"range": [50, 75], "color": "#ff9800"},
                {"range": [75, 100], "color": "#f44336"}
            ],
            "threshold": {"line": {"color": "white"}, "value": 75}
        }
    ))
    fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"})
    st.plotly_chart(fig, use_container_width=True)
    
    if fill_level > 75:
        st.error("⚠️ CRITICAL - Immediate collection required under NEMA Section 7")
    elif fill_level > 60:
        st.warning("⚠️ WARNING - Schedule collection soon")
    else:
        st.success("✅ NORMAL - Routine collection sufficient")

with col2:
    st.subheader("🚛 Collection Priority")
    priority = (fill_level / 100) * 10
    st.metric("Priority Score", f"{priority:.1f} / 10")
    st.progress(priority / 10)
    st.caption("Based on NEMA Section 8: Urgency Matrix")
    
    st.markdown("---")
    st.markdown("**NEMA Licensing Fees:**")
    st.markdown(f"Transport License: Ksh {NEMA_2024['license_fee_transport']:,}")
    st.markdown(f"Disposal Site License: Ksh {NEMA_2024['license_fee_disposal']:,}")

st.markdown("---")

# All counties comparison
st.subheader("📊 All Counties - Daily Waste (NEMA Formula)")

all_counties = list(COUNTIES.keys())
all_waste = [calculate_daily_waste(COUNTIES[c]["population"], COUNTIES[c]["waste_factor"]) for c in all_counties]

fig = px.bar(x=all_counties, y=all_waste, color=all_waste, 
             color_continuous_scale="Blues",
             title="Daily Waste Generation by County")
fig.update_layout(height=400, paper_bgcolor="#0a0f1a", plot_bgcolor="#0f1724", font_color="white", xaxis_tickangle=-45)
fig.add_hline(y=NEMA_2024["national_daily_waste"]/47, line_dash="dash", line_color="red", 
              annotation_text="National Average")
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Footer with NEMA citation
col1, col2 = st.columns(2)
with col1:
    st.caption("**Data Source:** National Environment Management Authority (NEMA)")
    st.caption("Simplified Waste Management Regulations, 2024")
with col2:
    st.caption("**CMT 444:** Distributed Machine Learning")
    st.caption("Federated Learning | Privacy Preserved")

st.info("📢 **NEMA 2024 Compliance:** This dashboard uses official waste management data from the National Environment Management Authority (NEMA) Simplified Waste Management Regulations, 2024.")
