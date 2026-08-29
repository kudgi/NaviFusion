import streamlit as st
import pydeck as pdk
import pandas as pd
import numpy as np
import math
import os
import time

# 1. Streamlit Page Config & Theme
st.set_page_config(
    page_title="NaviFusion 3D Telemetry Engine",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Cyberpunk & Glassmorphism UI CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0B0E14;
        color: #E6EDF3;
    }
    .main-title {
        font-family: 'Segoe UI', Roboto, sans-serif;
        font-weight: 800;
        font-size: 2.2rem;
        background: linear-gradient(90deg, #00F3FF 0%, #7000FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 0.9rem;
        color: #8B949E;
        margin-bottom: 20px;
    }
    .hud-card {
        background: rgba(22, 27, 34, 0.85);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #FFFFFF;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #8B949E;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .status-active {
        color: #00F3FF;
        border: 1px solid #00F3FF;
        background: rgba(0, 243, 255, 0.1);
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
    }
    .status-outage {
        color: #FF0055;
        border: 1px solid #FF0055;
        background: rgba(255, 0, 85, 0.1);
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        animation: blink 1.2s infinite;
    }
    @keyframes blink {
        50% { opacity: 0.4; }
    }
</style>
""", unsafe_allow_html=True)

# 2. Header
st.markdown('<div class="main-title">🛰️ NaviFusion 3D Autonomous Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Real-Time GNSS-Independent Localization & 3D Vehicle Trajectory Engine</div>', unsafe_allow_html=True)

# Coordinates Reference Origin (VIT Vellore)
LAT_ORIGIN, LON_ORIGIN = 12.9692, 79.1559

def meters_to_latlon(x, y):
    lat = LAT_ORIGIN + (y / 111111.0)
    lon = LON_ORIGIN + (x / (111111.0 * math.cos(math.radians(LAT_ORIGIN))))
    return lon, lat

# Official Deck.gl 3D Vehicle GLTF Asset (CORS Enabled)
CAR_MODEL_URL = "https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/3d-model/car.glb"

csv_path = "live_telemetry.csv"

# Placeholders for dynamic rendering
hud_col1, hud_col2, hud_col3, hud_col4 = st.columns(4)
map_placeholder = st.empty()

if os.path.exists(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if len(df) > 1:
            # Convert meter coordinates to Global Lat/Lon
            df['ekf_lon'], df['ekf_lat'] = zip(*df.apply(lambda row: meters_to_latlon(row['ekf_x'], row['ekf_y']), axis=1))
            df['imu_lon'], df['imu_lat'] = zip(*df.apply(lambda row: meters_to_latlon(row['imu_x'], row['imu_y']), axis=1))
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Calculate dynamic heading angle (Car orientation)
            dx = latest['ekf_x'] - prev['ekf_x']
            dy = latest['ekf_y'] - prev['ekf_y']
            heading = math.degrees(math.atan2(dx, dy)) if (dx != 0 or dy != 0) else 0

            # HUD Metric Updates
            is_active = latest['gnss_status'] == "ACTIVE"
            status_html = '<span class="status-active">🟢 GNSS ACTIVE</span>' if is_active else '<span class="status-outage">🔴 OUTAGE DETECTED</span>'
            
            hud_col1.markdown(f'<div class="hud-card"><div class="metric-label">Signal State</div><div style="margin-top:8px;">{status_html}</div></div>', unsafe_allow_html=True)
            hud_col2.markdown(f'<div class="hud-card"><div class="metric-label">Drift Prevented</div><div class="metric-value" style="color:#00F3FF;">{latest["drift_error"]:.2f} m</div></div>', unsafe_allow_html=True)
            hud_col3.markdown(f'<div class="hud-card"><div class="metric-label">Local Pose (X, Y)</div><div class="metric-value">({latest["ekf_x"]:.1f}m, {latest["ekf_y"]:.1f}m)</div></div>', unsafe_allow_html=True)
            hud_col4.markdown(f'<div class="hud-card"><div class="metric-label">Vehicle Heading</div><div class="metric-value" style="color:#FFB800;">{heading:.0f}°</div></div>', unsafe_allow_html=True)

            # 3D MAP LAYERS
            
            # Layer 1: Glowing Radar Halo under vehicle
            halo_color = [0, 243, 255, 180] if is_active else [255, 0, 85, 220]
            halo_layer = pdk.Layer(
                "ScatterplotLayer",
                data=[{"lon": latest['ekf_lon'], "lat": latest['ekf_lat']}],
                get_position="[lon, lat]",
                get_radius=12,
                get_fill_color=halo_color,
                pickable=False,
            )

            # Layer 2: Filtered EKF Path (Cyan Neon Ribbon)
            ekf_path_layer = pdk.Layer(
                "PathLayer",
                data=[{"path": df[['ekf_lon', 'ekf_lat']].values.tolist()}],
                get_path="path",
                get_color=[0, 243, 255, 230],
                width_scale=3,
                width_min_pixels=4,
                joint_rounded=True,
                cap_rounded=True,
            )

            # Layer 3: Uncorrected Raw IMU Drift Path (Red Dashed Line)
            imu_path_layer = pdk.Layer(
                "PathLayer",
                data=[{"path": df[['imu_lon', 'imu_lat']].values.tolist()}],
                get_path="path",
                get_color=[255, 0, 85, 180],
                width_scale=2,
                width_min_pixels=3,
            )

            # Layer 4: Real 3D GLTF Vehicle Model with Heading Rotation
            car_3d_layer = pdk.Layer(
                "ScenegraphLayer",
                data=[{
                    "position": [latest['ekf_lon'], latest['ekf_lat'], 0],
                    "orientation": [0, 0, heading]
                }],
                get_position="position",
                get_orientation="orientation",
                scenegraph=CAR_MODEL_URL,
                size_scale=1.2,
                _lighting="pbr", # Physically-based rendering for metallic reflections
                pickable=True
            )

            # 3D Viewport Configuration (Tilted Drone View)
            view_state = pdk.ViewState(
                longitude=latest['ekf_lon'],
                latitude=latest['ekf_lat'],
                zoom=18.5,
                pitch=60, # Tilted 3D perspective
                bearing=30
            )

            # Render Pydeck Map
            deck = pdk.Deck(
                layers=[halo_layer, ekf_path_layer, imu_path_layer, car_3d_layer],
                initial_view_state=view_state,
                map_style="mapbox://styles/mapbox/dark-v10",
                tooltip={"text": "NaviFusion Autonomous Vehicle\nPose: ({ekf_x}, {ekf_y})"}
            )

            map_placeholder.pydeck_chart(deck)

    except Exception as e:
        st.error(f"Waiting for valid telemetry stream... ({e})")
else:
    st.warning("Waiting for live_telemetry.csv from backend...")

# Refresh Loop for Smooth Animation
time.sleep(0.5)
st.rerun()