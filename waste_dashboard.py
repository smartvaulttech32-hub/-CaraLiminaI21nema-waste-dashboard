import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Nairobi Waste Management", layout="wide")

st.title("Nairobi Smart Waste Management System")
st.markdown("---")

# Data for 10 locations
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

with st.sidebar:
    st.markdown("### Select Location")
    location = st.selectbox("", list(bins_data.keys()))
    st.markdown("---")
    st.markdown("**NEMA 2024 Data**")
    st.markdown("22,000 tons/day")
    st.markdown("0.5 kg/person/day")

data = bins_data[location]

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
    st.subheader("Overflow Prediction")
    if data["hours"] <= 4:
        st.error(f"Overflow in {data['hours']} hours")
    elif data["hours"] <= 8:
        st.warning(f"Overflow in {data['hours']} hours")
    else:
        st.success(f"Overflow in {data['hours']} hours")

with col2:
    st.subheader("Priority Score")
    score = data["urgent"] + (data["critical"] / 10)
    st.metric("Score", f"{score:.1f} / 10")
    st.progress(score / 10)

st.markdown("---")

st.subheader("All Locations")
all_locations = list(bins_data.keys())
all_fills = [bins_data[l]["fill"] for l in all_locations]

fig = px.bar(x=all_locations, y=all_fills, color=all_fills, 
             color_continuous_scale="RdYlGn_r")
fig.update_layout(height=400)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("CMT 444: Distributed Machine Learning | NEMA 2024 Data")
