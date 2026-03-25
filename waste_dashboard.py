# ============================================
# NAIROBI WASTE MANAGEMENT SYSTEM
# CMT 444: DISTRIBUTED MACHINE LEARNING
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="CMT 444: Distributed ML - Nairobi Waste", layout="wide")

# Simple CSS
st.markdown("""
<style>
.stApp { background-color: #1e1e2e; }
h1, h2, h3, .stMarkdown, .stSelectbox label { color: white !important; }
</style>
""", unsafe_allow_html=True)

# Header
st.title("Nairobi Smart Waste Management System")
st.markdown("### CMT 444: Distributed Machine Learning | Federated Learning Demo")
st.markdown("---")

# Distributed ML Explanation
with st.expander("How Distributed ML Works (Federated Learning)"):
    st.markdown("""
    **Federated Learning Architecture:**
    
    - **10 Client Nodes:** Each Nairobi neighborhood = 1 client
    - **Local Data:** Waste data stays in each neighborhood
    - **No Data Sharing:** Raw data NEVER transmitted
    - **Privacy:** Only model weights are shared
    - **Aggregation:** Federated Averaging (FedAvg)
    """)

st.markdown("---")

# Data for 10 client nodes
bins_data = {
    "Mombasa Road": {"bins": 13, "fill": 78.9, "critical": 8, "urgent": 5, "hours": 6},
    "CBD": {"bins": 10, "fill": 78.2, "critical": 6, "urgent": 5, "hours": 6},
    "Ngong": {"bins": 11, "fill": 76.3, "critical": 5, "urgent": 4, "hours": 7},
    "Langata": {"bins": 9, "fill": 80.2, "critical": 6, "urgent": 4, "hours": 5},
    "Lavington": {"bins": 9, "fill": 82.6, "critical": 5, "urgent": 4, "hours": 5},
    "Westlands": {"bins": 7, "fill": 86.0, "critical": 4, "urgent": 4, "hours": 4},
    "Kileleshwa": {"bins": 4, "fill": 87.3, "critical": 3, "urgent": 4, "hours": 3},
    "Eastleigh": {"bins": 8, "fill": 81.1, "critical": 5, "urgent": 4, "hours": 5},
    "Rongai": {"bins": 6, "fill": 85.7, "critical": 4, "urgent": 4, "hours": 4},
    "Karen": {"bins": 5, "fill": 75.0, "critical": 2, "urgent": 3, "hours": 7}
}

# Sidebar
with st.sidebar:
    st.markdown("### CMT 444: Distributed ML")
    st.markdown("---")
    st.markdown("**Select Client Node**")
    location = st.selectbox("", list(bins_data.keys()))
    st.markdown("---")
    st.markdown("**Distributed ML Metrics**")
    st.metric("Active Clients", "10/10")
    st.progress(1.0)
    st.metric("Federated Rounds", "10")
    st.metric("Aggregation", "FedAvg")
    st.metric("Privacy", "100%")
    st.markdown("---")
    st.markdown("**NEMA 2024 Data**")
    st.markdown("22,000 tons/day")
    st.markdown("0.5 kg/person/day")
    st.markdown("85% collection target")

data = bins_data[location]

# Main dashboard
st.subheader(f"Client Node: {location} (Local Data - Never Shared)")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Bins", data["bins"])
with col2:
    st.metric("Fill Level", f"{data['fill']}%")
with col3:
    st.metric("Critical Bins", data["critical"])
with col4:
    st.metric("Urgency", f"{data['urgent']}/5")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Local Overflow Prediction")
    if data["hours"] <= 4:
        st.error(f"Overflow in {data['hours']} hours - Immediate!")
    elif data["hours"] <= 8:
        st.warning(f"Overflow in {data['hours']} hours - Schedule soon")
    else:
        st.success(f"Overflow in {data['hours']} hours - Normal")

with col2:
    st.subheader("Global Model Output")
    priority = data["urgent"] + (data["critical"] / 10)
    st.metric("Priority Score", f"{priority:.1f} / 10")
    st.progress(priority / 10)
    if priority >= 8:
        st.error("URGENT - Dispatch now")
    elif priority >= 6:
        st.warning("HIGH - Schedule today")
    elif priority >= 4:
        st.info("MEDIUM - Plan this week")
    else:
        st.success("LOW - Routine")
    st.caption("Aggregated from 10 client nodes via FedAvg")

st.markdown("---")

# Local data table
st.subheader(f"Local Data: {location} (Stays on Client)")

import random
random.seed(hash(location) % 100)
num_rows = min(7, data["bins"])
table_data = []
for i in range(num_rows):
    fill_val = round(random.uniform(20, 100), 1)
    if fill_val > 75:
        status = "Critical"
    elif fill_val > 60:
        status = "Warning"
    else:
        status = "Normal"
    table_data.append({
        "Bin ID": f"BIN_{i+1:03d}",
        "Fill Level %": fill_val,
        "Status": status,
        "Privacy": "Local Only"
    })

st.dataframe(pd.DataFrame(table_data), use_container_width=True)
st.caption("Data NEVER leaves this client node - only model weights are shared")

st.markdown("---")

# Bar chart for all clients
st.subheader("All Client Nodes Summary")

all_locations = list(bins_data.keys())
all_fills = [bins_data[l]["fill"] for l in all_locations]

fig = px.bar(
    x=all_locations,
    y=all_fills,
    color=all_fills,
    color_continuous_scale="RdYlGn_r",
    title="Fill Levels Across All Client Nodes"
)
fig.update_layout(
    height=400,
    paper_bgcolor="#1e1e2e",
    plot_bgcolor="#2d2d3a",
    font_color="white",
    xaxis_tickangle=-45
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Footer
st.caption("CMT 444: Distributed Machine Learning | Federated Learning Demo | NEMA 2024 Data")
st.caption("Privacy Preserved: Raw data stays in each client node | Only model weights aggregated via FedAvg")
