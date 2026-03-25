# ============================================
# CMT 444: FEDERATED LEARNING - WASTE MANAGEMENT
# NEMA 2024 | VISUAL DASHBOARD
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import random

st.set_page_config(page_title="NEMA Waste Management", layout="wide")

# ============================================
# NEMA 2024 DATA
# ============================================
NEMA = {"national": 22000, "per_capita": 0.5, "threshold": 75, "target": 85, "year": 2024}

COUNTIES = {
    "Nairobi": {"pop": 4397000, "bins": 50, "factor": 1.2},
    "Mombasa": {"pop": 1208000, "bins": 25, "factor": 1.0},
    "Kisumu": {"pop": 1155000, "bins": 20, "factor": 0.9},
    "Nakuru": {"pop": 2162000, "bins": 35, "factor": 0.95},
    "Kiambu": {"pop": 2417000, "bins": 40, "factor": 1.0}
}

# ============================================
# FEDERATED LEARNING
# ============================================
class FedAvg:
    def __init__(self):
        self.weights = {}
    def train(self, county, data):
        self.weights[county] = data["factor"] * (data["bins"] / 50)
    def aggregate(self):
        return sum(self.weights.values()) / len(self.weights)

fl = FedAvg()
for c, d in COUNTIES.items():
    fl.train(c, d)
global_w = fl.aggregate()

# ============================================
# DARK THEME
# ============================================
st.markdown("""
<style>
.stApp { background-color: #0a0f1a; }
.stMetric label { color: #ffaa66; font-size: 13px; font-weight: 500; }
.stMetric .stMarkdown { color: white; font-size: 32px; font-weight: bold; }
[data-testid="stSidebar"] { background-color: #0f1724; border-right: 2px solid #CC0000; }
h1, h2, h3, p { color: white; }
</style>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/Flag_of_Kenya.svg/1200px-Flag_of_Kenya.svg.png", width=60)
    st.markdown("### NEMA 2024")
    st.markdown(f"**{NEMA['national']:,}** tons/day")
    st.markdown(f"**{NEMA['per_capita']}** kg/person")
    st.markdown(f"**{NEMA['target']}%** target")
    st.markdown("---")
    st.markdown("### Select County")
    county = st.selectbox("", list(COUNTIES.keys()), label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### Federated Learning")
    st.markdown(f"**{len(fl.weights)}** active clients")
    st.markdown(f"**{global_w:.2f}** global weight")
    st.markdown("**FedAvg** aggregation")

data = COUNTIES[county]
daily = (data["pop"] * NEMA["per_capita"]) / 1000
fill = min(100, (daily / (data["bins"] * 50)) * 100)
priority = fill / 10
hours = max(0, (100 - fill) / 12)

# ============================================
# HEADER
# ============================================
st.title(f"🗑️ {county} Waste Management")
st.markdown(f"NEMA {NEMA['year']} | Federated Learning")

# ============================================
# METRICS ROW
# ============================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Smart Bins", data["bins"])
with col2:
    st.metric("Daily Waste", f"{daily:.0f} tons")
with col3:
    st.metric("Fill Level", f"{fill:.0f}%")
with col4:
    st.metric("Priority", f"{priority:.1f}/10")

st.markdown("---")

# ============================================
# MAIN VISUALS
# ============================================
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Fill Level")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fill,
        number={"suffix": "%", "font": {"size": 40, "color": "white"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "white"},
            "bar": {"color": "#ff9800"},
            "steps": [
                {"range": [0, 50], "color": "#2e7d32"},
                {"range": [50, 75], "color": "#ff9800"},
                {"range": [75, 100], "color": "#f44336"}
            ],
            "threshold": {"value": 75, "line": {"color": "white", "width": 4}}
        }
    ))
    fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"})
    st.plotly_chart(fig, use_container_width=True)
    
    if fill > 75:
        st.warning(f"⚠️ Overflow in {hours:.0f} hours")
    else:
        st.info(f"⏰ Overflow in {hours:.0f} hours")

with col2:
    st.markdown("#### Priority Score")
    fig2 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=priority,
        number={"suffix": "/10", "font": {"size": 40, "color": "white"}},
        gauge={
            "axis": {"range": [0, 10], "tickcolor": "white"},
            "bar": {"color": "#ffaa66"},
            "steps": [
                {"range": [0, 4], "color": "#2e7d32"},
                {"range": [4, 7], "color": "#ff9800"},
                {"range": [7, 10], "color": "#f44336"}
            ]
        }
    ))
    fig2.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"})
    st.plotly_chart(fig2, use_container_width=True)
    
    if priority >= 7:
        st.error("URGENT - Dispatch now")
    elif priority >= 5:
        st.warning("HIGH - Schedule today")
    else:
        st.success("NORMAL - Routine")

st.markdown("---")

# ============================================
# BIN STATUS CHART
# ============================================
st.markdown("#### Bin Status")

random.seed(hash(county) % 100)
bins_data = [random.uniform(20, 100) for _ in range(data["bins"])]
colors = ["#f44336" if x > 75 else "#ff9800" if x > 60 else "#2e7d32" for x in bins_data]
critical_count = len([x for x in bins_data if x > 75])

fig3 = go.Figure(data=[go.Bar(
    x=[f"B{i+1}" for i in range(data["bins"])],
    y=bins_data,
    marker_color=colors,
    text=[f"{x:.0f}%" for x in bins_data],
    textposition="outside"
)])
fig3.update_layout(
    height=300,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="white",
    xaxis_title="",
    yaxis_title="Fill %",
    showlegend=False
)
st.plotly_chart(fig3, use_container_width=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Critical Bins", critical_count)
with col2:
    st.metric("Warning Bins", len([x for x in bins_data if 60 < x <= 75]))
with col3:
    st.metric("Normal Bins", len([x for x in bins_data if x <= 60]))

st.markdown("---")

# ============================================
# ALL COUNTIES COMPARISON
# ============================================
st.markdown("#### All Counties")

all_waste = [COUNTIES[c]["pop"] * NEMA["per_capita"] / 1000 for c in COUNTIES.keys()]
all_fill = [min(100, (w / (COUNTIES[c]["bins"] * 50)) * 100) for w, c in zip(all_waste, COUNTIES.keys())]

fig4 = go.Figure()
fig4.add_trace(go.Bar(
    x=list(COUNTIES.keys()),
    y=all_fill,
    marker_color=all_fill,
    marker_colorscale="RdYlGn_r",
    text=[f"{x:.0f}%" for x in all_fill],
    textposition="outside",
    name="Fill Level"
))
fig4.add_trace(go.Scatter(
    x=list(COUNTIES.keys()),
    y=[75] * 5,
    mode="lines",
    line=dict(color="red", dash="dash", width=2),
    name="NEMA Threshold (75%)"
))
fig4.update_layout(
    height=400,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="white",
    yaxis_title="Fill %",
    legend=dict(orientation="h", yanchor="bottom", y=1.02)
)
st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# ============================================
# FOOTER
# ============================================
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"NEMA {NEMA['year']}: {NEMA['national']:,} tons/day")
with col2:
    st.caption(f"Federated Learning | {len(fl.weights)} clients")
with col3:
    st.caption(f"Global weight: {global_w:.2f} | FedAvg")
