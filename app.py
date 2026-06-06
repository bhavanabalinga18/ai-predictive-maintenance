import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import plotly.graph_objects as go
import os

# ==========================================
# 1. PAGE LAYOUT CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Industry 5.0 Predictive Maintenance Space",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Industrial Predictive Maintenance Platform")
st.caption("Synchronized Google Colab Core, GitHub Source Control, and Streamlit Engine Execution")

# Initialize real-time tracking array in session memory
if "temp_history" not in st.session_state:
    st.session_state.temp_history = list(np.random.normal(55, 2, 15))

# ==========================================
# 2. DEFINE THE PYTORCH PINN ARCHITECTURE
# ==========================================
class ThermalPINN(nn.Module):
    def __init__(self):
        super(ThermalPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, t, rpm):
        X = torch.cat([t, rpm], dim=1)
        return self.net(X)

@st.cache_resource
def load_production_model():
    """Loads the pre-trained model weights exported from Google Colab."""
    model = ThermalPINN()
    weights_path = "models/pinn_weights.pt"
    
    # Check if weights file exists in the repository
    if os.path.exists(weights_path):
        try:
            model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
            st.sidebar.success(" Loaded production-grade Colab weights.")
        except Exception as e:
            st.sidebar.error(f"Error loading weights: {str(e)}")
    else:
        st.sidebar.warning(" Running on un-optimized structural baseline weights. Please add models/pinn_weights.pt")
        
    model.eval()
    return model

# Initialize network pipeline
pinn_network = load_production_model()

# ==========================================
# 3. LIVE INTERACTIVE TELEMETRY UI
# ==========================================
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("📊 Live Asset Telemetry")
    
    # Simulating standard IoT streaming metrics from a mechanical drive
    sim_rpm = float(np.random.normal(1520, 40))
    sim_vib = float(np.random.normal(2.3, 0.2))
    
    # Run rapid machine inference with zero computational overhead
    with torch.no_grad():
        t_tensor = torch.tensor([[0.5]], dtype=torch.float32)
        rpm_tensor = torch.tensor([[sim_rpm / 2000.0]], dtype=torch.float32) # Normalized input step
        
        # Scale mathematical output range back to standard temperature space
        pinn_temp_pred = float(pinn_network(t_tensor, rpm_tensor).item() * 100)

    # Append new prediction point to the moving window chart matrix
    st.session_state.temp_history.append(pinn_temp_pred)
    if len(st.session_state.temp_history) > 20:
        st.session_state.temp_history.pop(0)

    # Display clear telemetry reading widgets
    m1, m2 = st.columns(2)
    m1.metric("PINN Temp Prediction", f"{pinn_temp_pred:.1f} °C", delta=f"{(pinn_temp_pred-60.0):.1f} °C")
    m2.metric("Vibration Core Status", f"{sim_vib:.2f} mm/s")
    
    m3, m4 = st.columns(2)
    m3.metric("Rotational Speed", f"{sim_rpm:.0f} RPM")
    m4.metric("Hydraulic System Pressure", f"{float(np.random.normal(6.0, 0.1)):.2f} bar")
    
    # User polling button interface hook
    if st.button("Poll Live Sensors Engine Line", type="primary"):
        st.rerun()

with col_right:
    st.subheader("📈 Real-Time Thermodynamic Simulation Profile")
    
    # Build clean, interactive Plotly time series visualization
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=st.session_state.temp_history, 
        mode='lines+markers', 
        name='PINN Trend Line',
        line=dict(color='#00fff2', width=3)
    ))
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20), 
        height=260, 
        template="plotly_dark",
        xaxis=dict(showgrid=False, title="Polling Step Window Index"),
        yaxis=dict(title="Temperature (°C)")
    )
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 4. DIGITAL TWIN HEALTH CLASSIFIER MATRIX
# ==========================================
st.markdown("---")
status_label, status_color = "OPERATIONAL STRATIFICATION SUCCESS (HEALTHY)", "green"

if pinn_temp_pred > 75 or sim_vib > 2.8:
    status_label, status_color = "THERMAL DEGRADATION THRESHOLD DETECTED (WARNING)", "orange"
if pinn_temp_pred > 90 or sim_vib > 3.8:
    status_label, status_color = "CRITICAL METRIC FAILURE INFRASTRUCTURE EVENT (ANOMALY)", "red"

st.markdown(f"### Digital Twin Diagnostic State: :{status_color}[{status_label}]")
