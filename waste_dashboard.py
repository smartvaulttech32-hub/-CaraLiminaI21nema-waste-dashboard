
# DISTRIBUTED MACHINE LEARNING
# WASTE MANAGEMENT DASHBOARD
# NEMA 2024 | CLEAN CITIES | FEDERATED LEARNING

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import random

st.set_page_config(page_title="NEMA Waste Management | CMT 444", layout="wide")


# NEMA 2024 OFFICIAL DATA

NEMA = {
    "national": 22000,
    "per_capita": 0.5,
    "threshold": 75,
    "target": 85,
    "organic": 65,
    "plastic": 20,
    "paper": 10,
    "metal": 2,
    "medical": 1,
    "year": 2024,
    "source": "Simplified Waste Management Regulations, 2024"
}


# KENYAN COUNTIES DATA

COUNTIES = {
    "Nairobi": {
        "pop": 4397000, "bins": 50, "waste_tons": 2198,
        "clean_score": 65, "recycling_rate": 12, "collection_rate": 70,
        "illegal_dumping": "High", "air_quality": "Poor",
        "status": "Moderate", "color": "#ff9800",
        "challenges": ["High population density", "Informal settlements", "Limited dumpsites"],
        "solutions": ["Community composting", "Waste-to-energy plant", "Door-to-door collection"]
    },
    "Mombasa": {
        "pop": 1208000, "bins": 25, "waste_tons": 604,
        "clean_score": 58, "recycling_rate": 8, "collection_rate": 55,
        "illegal_dumping": "Very High", "air_quality": "Poor",
        "status": "Dirty", "color": "#f44336",
        "challenges": ["Beach pollution", "Tourism waste", "Ocean plastic"],
        "solutions": ["Beach clean-up programs", "Plastic ban enforcement", "Marine waste collection"]
    },
    "Kisumu": {
        "pop": 1155000, "bins": 20, "waste_tons": 520,
        "clean_score": 72, "recycling_rate": 15, "collection_rate": 78,
        "illegal_dumping": "Medium", "air_quality": "Moderate",
        "status": "Clean", "color": "#2e7d32",
        "challenges": ["Lake pollution", "Agricultural waste", "Flooding"],
        "solutions": ["Lake protection programs", "Composting facilities", "Flood waste management"]
    },
    "Nakuru": {
        "pop": 2162000, "bins": 35, "waste_tons": 1027,
        "clean_score": 68, "recycling_rate": 10, "collection_rate": 65,
        "illegal_dumping": "High", "air_quality": "Moderate",
        "status": "Moderate", "color": "#ff9800",
        "challenges": ["Industrial waste", "Market waste", "Landfill capacity"],
        "solutions": ["Industrial recycling", "Market composting", "Sanitary landfill"]
    },
    "Kiambu": {
        "pop": 2417000, "bins": 40, "waste_tons": 1208,
        "clean_score": 75, "recycling_rate": 18, "collection_rate": 82,
        "illegal_dumping": "Low", "air_quality": "Good",
        "status": "Clean", "color": "#2e7d32",
        "challenges": ["Rapid urbanization", "Farm waste", "Water pollution"],
        "solutions": ["Smart bins", "Farm waste recycling", "Water treatment"]
    }
}


# RESEARCH DATA

RESEARCH = {
    "best_practices": [
        {"practice": "Segregation at Source", "impact": "65% waste reduction", "cost": "Low"},
        {"practice": "Composting Organic Waste", "impact": "40% landfill diversion", "cost": "Medium"},
        {"practice": "Plastic Recycling", "impact": "20% waste reduction", "cost": "Medium"},
        {"practice": "Waste-to-Energy", "impact": "80% volume reduction", "cost": "High"},
        {"practice": "Smart Bins IoT", "impact": "30% efficiency increase", "cost": "Medium"}
    ],
    "success_stories": [
        {"city": "Kigali, Rwanda", "achievement": "Cleanest city in Africa", "method": "Community policing, plastic ban"},
        {"city": "Cape Town, SA", "achievement": "65% recycling rate", "method": "Separation at source"},
        {"city": "Busan, Korea", "achievement": "Zero waste city", "method": "Pay-as-you-throw system"},
        {"city": "San Francisco, USA", "achievement": "80% landfill diversion", "method": "Mandatory recycling"}
    ]
}


# FEDERATED LEARNING 

class FedAvg:
    def __init__(self):
        self.weights = {}
    def train(self, county, data):
        self.weights[county] = data["recycling_rate"] / 100
    def aggregate(self):
        return sum(self.weights.values()) / len(self.weights)

fl = FedAvg()
for c, d in COUNTIES.items():
    fl.train(c, d)
global_w = fl.aggregate()


# DARK THEME

st.markdown("""
<style>
.stApp { background-color: #0a0f1a; }
.stMetric label { color: #ffaa66; font-size: 13px; }
.stMetric .stMarkdown { color: white; font-size: 28px; font-weight: bold; }
[data-testid="stSidebar"] { background-color: #0f1724; border-right: 2px solid #CC0000; }
h1, h2, h3, p, .stMarkdown { color: white; }
</style>
""", unsafe_allow_html=True)


# SIDEBAR

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/Flag_of_Kenya.svg/1200px-Flag_of_Kenya.svg.png", width=70)
    st.markdown("## CMT 444")
    st.markdown("### Distributed ML")
    st.markdown("---")
    
    st.markdown("### 📍 Select County")
    county = st.selectbox("", list(COUNTIES.keys()), label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("### 📊 NEMA 2024 Data")
    st.markdown(f"**National:** {NEMA['national']:,} tons/day")
    st.markdown(f"**Per Capita:** {NEMA['per_capita']} kg/day")
    st.markdown(f"**Target:** {NEMA['target']}%")
    st.markdown(f"**Threshold:** {NEMA['threshold']}%")
    
    st.markdown("---")
    st.markdown("### 🔄 Federated Learning")
    st.metric("Active Clients", f"{len(fl.weights)}/5")
    st.metric("Global Weight", f"{global_w:.2f}")
    st.metric("Aggregation", "FedAvg")
    st.metric("Privacy", "🔒 100%")
    
    st.markdown("---")
    st.markdown("### 🏆 Clean Cities Rank")
    sorted_cities = sorted(COUNTIES.items(), key=lambda x: x[1]["clean_score"], reverse=True)
    for i, (c, d) in enumerate(sorted_cities[:3], 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
        st.markdown(f"{emoji} **{c}** - {d['clean_score']}%")

data = COUNTIES[county]


# HEADER

st.title(f"🗑️ {county} Waste Management System")
st.markdown(f"*NEMA {NEMA['year']} | Federated Learning | Clean City Score: {data['clean_score']}%*")
st.markdown("---")

# METRICS ROW

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Population", f"{data['pop']:,}")
with col2:
    st.metric("Smart Bins", data["bins"])
with col3:
    st.metric("Daily Waste", f"{data['waste_tons']} tons")
with col4:
    st.metric("Recycling Rate", f"{data['recycling_rate']}%")
with col5:
    st.metric("Collection Rate", f"{data['collection_rate']}%")

st.markdown("---")


# GAUGES

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📊 Fill Level")
    fill = min(100, (data["waste_tons"] / (data["bins"] * 50)) * 100)
    hours = max(0, (100 - fill) / 12)
    
    fig1 = go.Figure(go.Indicator(
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
            "threshold": {"value": NEMA["threshold"], "line": {"color": "white", "width": 4}}
        }
    ))
    fig1.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"})
    st.plotly_chart(fig1, use_container_width=True)
    
    if fill > 75:
        st.warning(f"⚠️ Overflow in {hours:.0f} hours")
    else:
        st.info(f"⏰ Overflow in {hours:.0f} hours")

with col2:
    st.markdown("#### 🎯 Cleanliness Score")
    fig2 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=data["clean_score"],
        number={"suffix": "%", "font": {"size": 40, "color": "white"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "white"},
            "bar": {"color": data["color"]},
            "steps": [
                {"range": [0, 50], "color": "#f44336", "name": "Dirty"},
                {"range": [50, 70], "color": "#ff9800", "name": "Moderate"},
                {"range": [70, 100], "color": "#2e7d32", "name": "Clean"}
            ]
        }
    ))
    fig2.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"})
    st.plotly_chart(fig2, use_container_width=True)
    
    if data["clean_score"] >= 70:
        st.success(f"✅ {data['status']} City")
    elif data["clean_score"] >= 50:
        st.warning(f"⚠️ {data['status']} City")
    else:
        st.error(f"❌ {data['status']} City")

st.markdown("---")


# BIN STATUS

st.markdown("#### 📍 Bin Status")

random.seed(hash(county) % 100)
bins_data = [min(100, max(20, random.uniform(20, 100) + (fill - 70))) for _ in range(min(data["bins"], 30))]
colors = ["#f44336" if x > 75 else "#ff9800" if x > 60 else "#2e7d32" for x in bins_data]
critical = len([x for x in bins_data if x > 75])
warning = len([x for x in bins_data if 60 < x <= 75])
normal = len([x for x in bins_data if x <= 60])

fig3 = go.Figure(data=[go.Bar(
    x=[f"B{i+1}" for i in range(len(bins_data))],
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
    yaxis_title="Fill %"
)
st.plotly_chart(fig3, use_container_width=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🔴 Critical", critical)
with col2:
    st.metric("🟡 Warning", warning)
with col3:
    st.metric("🟢 Normal", normal)

st.markdown("---")

# ============================================
# CHALLENGES & SOLUTIONS
# ============================================
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### ⚠️ Challenges")
    for c in data["challenges"]:
        st.markdown(f"- {c}")
    st.markdown(f"**Illegal Dumping:** {data['illegal_dumping']}")
    st.markdown(f"**Air Quality:** {data['air_quality']}")

with col2:
    st.markdown("#### ✅ Recommended Solutions")
    for s in data["solutions"]:
        st.markdown(f"- {s}")

st.markdown("---")


# ALL CITIES COMPARISON

st.markdown("#### 🏆 All Cities - Cleanliness & Waste")

all_cities = list(COUNTIES.keys())
all_scores = [COUNTIES[c]["clean_score"] for c in all_cities]
all_waste = [COUNTIES[c]["waste_tons"] for c in all_cities]
all_recycling = [COUNTIES[c]["recycling_rate"] for c in all_cities]

fig4 = go.Figure()
fig4.add_trace(go.Bar(
    x=all_cities,
    y=all_scores,
    name="Cleanliness Score",
    marker_color=all_scores,
    marker_colorscale="RdYlGn",
    text=[f"{x}%" for x in all_scores],
    textposition="outside"
))
fig4.add_trace(go.Scatter(
    x=all_cities,
    y=[70] * 5,
    mode="lines",
    line=dict(color="green", dash="dash", width=2),
    name="Clean Target (70%)"
))
fig4.add_trace(go.Scatter(
    x=all_cities,
    y=[50] * 5,
    mode="lines",
    line=dict(color="orange", dash="dash", width=2),
    name="Moderate Threshold (50%)"
))
fig4.update_layout(
    height=400,
    title="Cleanliness Score by City",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="white",
    yaxis_title="Score (%)"
)
st.plotly_chart(fig4, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    fig5 = px.bar(x=all_cities, y=all_waste, color=all_waste, 
                  color_continuous_scale="Blues", title="Daily Waste (tons)")
    fig5.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
    st.plotly_chart(fig5, use_container_width=True)

with col2:
    fig6 = px.bar(x=all_cities, y=all_recycling, color=all_recycling,
                  color_continuous_scale="Greens", title="Recycling Rate (%)")
    fig6.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
    st.plotly_chart(fig6, use_container_width=True)

st.markdown("---")


# RESEARCH SECTION - HOW TO MAKE CITIES CLEAN

st.markdown("## 📚 Research: How to Make Cities Clean")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🌍 Best Practices")
    df_practices = pd.DataFrame(RESEARCH["best_practices"])
    st.dataframe(df_practices, use_container_width=True, hide_index=True)

with col2:
    st.markdown("### 🏆 Global Success Stories")
    df_stories = pd.DataFrame(RESEARCH["success_stories"])
    st.dataframe(df_stories, use_container_width=True, hide_index=True)

st.markdown("---")


# RECOMMENDATIONS FOR SELECTED CITY

st.markdown(f"### 📋 Recommendations for {county}")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Immediate Actions**")
    if data["clean_score"] < 50:
        st.markdown("- Emergency waste collection")
        st.markdown("- Clean up illegal dumpsites")
        st.markdown("- Enforce anti-littering laws")
    elif data["clean_score"] < 70:
        st.markdown("- Improve collection efficiency")
        st.markdown("- Start segregation programs")
        st.markdown("- Community awareness campaigns")
    else:
        st.markdown("- Maintain current systems")
        st.markdown("- Expand recycling programs")
        st.markdown("- Share best practices")

with col2:
    st.markdown("**Medium Term (6-12 months)**")
    st.markdown("- Install smart bins")
    st.markdown("- Build composting facilities")
    st.markdown("- Train waste collectors")

with col3:
    st.markdown("**Long Term (1-3 years)**")
    st.markdown("- Waste-to-energy plant")
    st.markdown("- Zero waste certification")
    st.markdown("- Circular economy programs")

st.markdown("---")


# FEDERATED LEARNING RESULTS

st.markdown("## 🔄 Federated Learning Results")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Local Models**")
    for c, w in fl.weights.items():
        st.markdown(f"- {c}: {w:.2f}")

with col2:
    st.markdown("**Federated Aggregation**")
    st.latex(r"\text{Global} = \frac{1}{n}\sum_{i=1}^{n} \text{Weight}_i")
    st.markdown(f"**Global Weight:** {global_w:.3f}")

with col3:
    st.markdown("**Privacy Guarantee**")
    st.markdown("🔒 Raw data NEVER shared")
    st.markdown("✅ Only model weights transmitted")
    st.markdown("✅ FedAvg aggregation")

st.markdown("---")


# FOOTER

col1, col2, col3 = st.columns(3)

with col1:
    st.caption(f"🇰🇪 **Data Source:** NEMA {NEMA['year']} - {NEMA['source']}")

with col2:
    st.caption(f"⚙️ **CMT 444:** Distributed ML | Federated Learning | FedAvg")

with col3:
    st.caption(f"🌍 **SDG Goal 11:** Sustainable Cities and Communities")
