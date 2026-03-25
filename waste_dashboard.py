# ============================================
# CMT 444: FEDERATED LEARNING - WASTE MANAGEMENT
# NEMA 2024 | VISUAL DASHBOARD
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import random

st.set_page_config(page_title="NEMA Waste Dashboard", layout="wide")

# ============================================
# NEMA 2024 DATA
# ============================================
NEMA = {"national": 22000, "per_capita": 0.5, "threshold": 75, "year": 2024}

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
.stMetric label { color: #ffaa66; font-size: 12px; }
.stMetric .stMarkdown { color: white; font-size: 28px; }
[data-testid="stSidebar"] { background-color: #0f1724; border-right: 2px solid #CC0000; }
</style>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR - MINIMAL
# ============================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/Flag_of_Kenya.svg/1200px-Flag_of_Kenya.svg.png", width=50)
    st.markdown("---")
    county = st.selectbox("", list(COUNTIES.keys()), label_visibility="collapsed")
    st.markdown("---")
    st.markdown(f"**{NEMA['national']:,}**")
    st.markdown(f"**{NEMA['per_capita']} kg**")
    st.markdown(f"**{global_w:.2f}**")

data = COUNTIES[county]
daily = (data["pop"] * NEMA["per_capita"]) / 1000
fill = min(100, (daily / (data["bins"] * 50)) * 100)
priority = fill / 10
hours = max(0, (100 - fill) / 12)

# ============================================
# TOP METRICS
# ============================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("", data["bins"], delta=f"{county}")
with col2:
    st.metric("", f"{daily:.0f}", delta="tons/day")
with col3:
    st.metric("", f"{fill:.0f}%", delta="fill")
with col4:
    st.metric("", f"{priority:.1f}", delta="/10")

st.markdown("---")

# ============================================
# MAIN VISUALS
# ============================================
col1, col2 = st.columns(2)

with col1:
    # Gauge
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
    fig.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"})
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Priority Meter
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
    fig2.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"})
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ============================================
# BIN STATUS - VISUAL ONLY
# ============================================
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    random.seed(hash(county) % 100)
    bins_data = []
    for i in range(data["bins"]):
        fill_bin = random.uniform(20, 100)
        bins_data.append(fill_bin)
    
    colors = ["#f44336" if x > 75 else "#ff9800" if x > 60 else "#2e7d32" for x in bins_data]
    
    fig3 = go.Figure(data=[go.Bar(
        x=[f"B{i+1}" for i in range(data["bins"])],
        y=bins_data,
        marker_color=colors,
        text=[f"{x:.0f}%" for x in bins_data],
        textposition="outside"
    )])
    fig3.update_layout(
        height=300,
        title="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        xaxis_title="",
        yaxis_title="Fill %"
    )
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ============================================
# ALL COUNTIES COMPARISON
# ============================================
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
    name="NEMA Threshold"
))
fig4.update_layout(
    height=400,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="white",
    yaxis_title="Fill %",
    showlegend=True
)
st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# ============================================
# BOTTOM METRICS
# ============================================
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("", f"{hours:.0f}", delta="hours")
with col2:
    st.metric("", f"{len([x for x in bins_data if x > 75])}", delta="critical")
with col3:
    st.metric("", f"{len(fl.weights)}", delta="clients")
with col4:
    st.metric("", f"{global_w:.2f}", delta="global")
with col5:
    st.metric("", f"{NEMA['year']}", delta="NEMA")
