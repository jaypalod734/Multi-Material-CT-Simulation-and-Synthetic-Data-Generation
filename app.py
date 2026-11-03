import os
import glob
import time
import json
import uuid
import sys
import math
import subprocess
import streamlit as st
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image

from config import (
    SimulationConfig,
    PartConfig,
    SourceConfig,
    DetectorConfig,
    AcquisitionConfig,
    FilterConfig
)
from utils import (
    load_materials_db,
    calculate_auto_framing,
    estimate_output_size,
    generate_kramers_spectrum,
    get_stl_bounds_cm
)


# Page Configuration 
st.set_page_config(
    page_title="Multi-Material CT Simulation and Data Generation",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Light-Blue Theme Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* App Background & Layout */
    .stApp {
        background-color: #F0F9FF;
        color: #0F172A;
    }

    .main .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2.5rem;
        max-width: 96%;
    }

    /* WebCT Styled Header Bar */
    .webct-header {
        background: #FFFFFF;
        border: 1px solid #BAE6FD;
        border-radius: 12px;
        padding: 1.2rem 1.6rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 10px rgba(14, 165, 233, 0.08);
    }

    .main-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0C4A6E;
        margin-bottom: 0.2rem;
        letter-spacing: -0.3px;
    }

    .sub-title {
        font-size: 0.95rem;
        color: #475569;
        font-weight: 400;
    }

    .header-badge {
        background: #E0F2FE;
        color: #0369A1;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 20px;
        border: 1px solid #BAE6FD;
        display: inline-block;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #EBF5FF !important;
        border-right: 1px solid #BAE6FD !important;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.2rem;
        padding-left: 1.1rem;
        padding-right: 1.1rem;
    }

    .sidebar-section-title {
        font-size: 0.92rem;
        font-weight: 700;
        color: #0369A1;
        border-left: 4px solid #0EA5E9;
        padding-left: 8px;
        margin-top: 1.1rem;
        margin-bottom: 0.5rem;
        letter-spacing: 0.3px;
        text-transform: uppercase;
    }

    /* WebCT Card Containers */
    .webct-card {
        background: #FFFFFF;
        border: 1px solid #BAE6FD;
        border-radius: 10px;
        padding: 1.1rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 1px 6px rgba(14, 165, 233, 0.05);
    }

    .webct-section-title {
        font-size: 1.18rem;
        font-weight: 700;
        color: #0369A1;
        border-left: 4px solid #0EA5E9;
        padding-left: 10px;
        margin-bottom: 0.8rem;
        margin-top: 0.3rem;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Material Category Badges */
    .badge-metal {
        background: #E0F2FE;
        color: #0369A1;
        padding: 3px 9px;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 600;
        border: 1px solid #BAE6FD;
        display: inline-block;
    }

    .badge-polymer {
        background: #DCFCE7;
        color: #15803D;
        padding: 3px 9px;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 600;
        border: 1px solid #BBF7D0;
        display: inline-block;
    }

    .badge-biological {
        background: #FFEDD5;
        color: #C2410C;
        padding: 3px 9px;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 600;
        border: 1px solid #FED7AA;
        display: inline-block;
    }

    .badge-gas {
        background: #F1F5F9;
        color: #475569;
        padding: 3px 9px;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 600;
        border: 1px solid #E2E8F0;
        display: inline-block;
    }

    /* Summary Card Before Run */
    .summary-card {
        background: #FFFFFF;
        border: 1px solid #7DD3FC;
        border-left: 4px solid #0EA5E9;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.08);
    }

    /* Metrics Styling */
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #BAE6FD;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 1px 4px rgba(14, 165, 233, 0.05);
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        color: #64748B !important;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        color: #0284C7 !important;
    }

    /* Buttons */
    div.stButton > button[kind="primary"], div.stButton > button:first-child[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #0EA5E9, #0284C7) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 6px rgba(14, 165, 233, 0.3) !important;
        transition: all 0.2s ease !important;
    }

    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0284C7, #0369A1) !important;
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.45) !important;
        transform: translateY(-1px);
    }

    div.stButton > button:not([kind="primary"]) {
        background: #FFFFFF;
        border: 1px solid #BAE6FD;
        color: #0369A1;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    div.stButton > button:not([kind="primary"]):hover {
        border-color: #0EA5E9;
        background: #F0F9FF;
        color: #0284C7;
    }

    /* Tabs Styling */
    button[data-baseweb="tab"] {
        font-size: 0.92rem !important;
        font-weight: 600 !important;
        color: #64748B !important;
        border-bottom: 3px solid transparent !important;
        padding: 0.5rem 1.1rem !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #0284C7 !important;
        border-bottom-color: #0EA5E9 !important;
        font-weight: 700 !important;
        background: transparent !important;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        background: #FFFFFF !important;
        border: 1px solid #BAE6FD !important;
        border-radius: 10px !important;
        margin-bottom: 0.75rem !important;
        box-shadow: 0 1px 4px rgba(14, 165, 233, 0.04) !important;
    }

    div[data-testid="stExpander"] summary {
        font-weight: 600 !important;
        color: #0F172A !important;
    }

    div[data-testid="stExpander"] summary:hover {
        color: #0284C7 !important;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_stls")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_stls")
os.makedirs(UPLOAD_DIR, exist_ok=True)
MATERIALS_DB = load_materials_db()

# --------------------------------------------------------------------------------
# Helper Functions

def get_material_badge_html(material_name: str) -> str:
    """Return styled HTML pill badge for the material category."""
    cat = MATERIALS_DB.get(material_name, {}).get("category", "Other")
    if cat == "Metal":
        return '<span class="badge-metal">Metal</span>'
    elif cat == "Polymer":
        return '<span class="badge-polymer">Polymer</span>'
    elif cat == "Biological":
        return '<span class="badge-biological">Biological</span>'
    elif cat == "Gas":
        return '<span class="badge-gas">Gas</span>'
    else:
        return f'<span class="badge-gas">{cat}</span>'

def estimate_run_time_sec(num_proj: int, beam_mode: str) -> float:
    # Calculate approximate execution time in seconds.
    time_per_frame = 0.08 if beam_mode == "Polychromatic" else 0.025
    return max(1.0, num_proj * time_per_frame)

def render_3d_scanner_geometry(
    sod_val: float,
    sdd_offset: float,
    sdd_val: float,
    det_rows: int,
    det_cols: int,
    pixel_size: float,
    stl_bounds: tuple = (2.0, 2.0, 1.5),
    source_type: str = "Point Source"
) -> go.Figure:
    """
    Renders an interactive 3D CT scanner geometry diagram using Plotly.
    Takes purely primitive values and returns a go.Figure.
    """
    dx, dy, dz = stl_bounds
    magnification = (sdd_val / sod_val) if sod_val > 0 else 1.0
    r_max = math.sqrt(dx**2 + dy**2)
    projected_footprint_cm = 2.0 * r_max * magnification
    detector_physical_width_cm = det_cols * pixel_size / 10.0
    detector_physical_height_cm = det_rows * pixel_size / 10.0
    object_fits = projected_footprint_cm <= detector_physical_width_cm

    w2 = detector_physical_width_cm / 2.0
    h2 = detector_physical_height_cm / 2.0

    fig = go.Figure()

    # 1. X-ray Point Source: single large red marker with label at (-sod_val, 0, 0)
    fig.add_trace(go.Scatter3d(
        x=[-sod_val],
        y=[0.0],
        z=[0.0],
        mode='markers+text',
        marker=dict(size=9, color='#EF4444', symbol='circle'),
        text=["X-ray Source"],
        textposition="top center",
        textfont=dict(color="#F87171", size=12),
        name="X-ray Source",
        hoverinfo="text"
    ))

    # 2. Detector Plane: semi-transparent blue rectangle (4 vertices, 2 triangles) at (+sdd_offset, 0, 0)
    x_det = [sdd_offset, sdd_offset, sdd_offset, sdd_offset]
    y_det = [-w2, w2, w2, -w2]
    z_det = [-h2, -h2, h2, h2]

    fig.add_trace(go.Mesh3d(
        x=x_det,
        y=y_det,
        z=z_det,
        i=[0, 0],
        j=[1, 2],
        k=[2, 3],
        color='#3B82F6',
        opacity=0.35,
        name="Detector Plane",
        hoverinfo="name"
    ))

    # Detector perimeter wireframe
    fig.add_trace(go.Scatter3d(
        x=[sdd_offset, sdd_offset, sdd_offset, sdd_offset, sdd_offset],
        y=[-w2, w2, w2, -w2, -w2],
        z=[-h2, -h2, h2, h2, -h2],
        mode='lines',
        line=dict(color='#60A5FA', width=3),
        name="Detector Boundary",
        showlegend=False
    ))

    # 3. Cone Beam Frustum: 4 yellow/orange lines from source point to the 4 detector rectangle corners
    frustum_x = [
        -sod_val, sdd_offset, None,
        -sod_val, sdd_offset, None,
        -sod_val, sdd_offset, None,
        -sod_val, sdd_offset, None
    ]
    frustum_y = [
        0.0, -w2, None,
        0.0,  w2, None,
        0.0,  w2, None,
        0.0, -w2, None
    ]
    frustum_z = [
        0.0, -h2, None,
        0.0, -h2, None,
        0.0,  h2, None,
        0.0,  h2, None
    ]
    fig.add_trace(go.Scatter3d(
        x=frustum_x,
        y=frustum_y,
        z=frustum_z,
        mode='lines',
        line=dict(color='#F59E0B', width=2),
        name="Cone Beam Frustum"
    ))

    # 4. Central Beam Axis: dashed white/gray line from source (-sod, 0, 0) through origin (0,0,0) to detector center (+odd, 0, 0)
    fig.add_trace(go.Scatter3d(
        x=[-sod_val, 0.0, sdd_offset],
        y=[0.0, 0.0, 0.0],
        z=[0.0, 0.0, 0.0],
        mode='lines',
        line=dict(color='#CBD5E1', width=3, dash='dash'),
        name="Central Beam Axis"
    ))

    # 5. Object Bounding Box: 12 wireframe edges [-dx, +dx] x [-dy, +dy] x [-dz, +dz]
    box_color = '#10B981' if object_fits else '#EF4444'
    box_x = [
        -dx,  dx,  dx, -dx, -dx, None,
        -dx,  dx,  dx, -dx, -dx, None,
        -dx, -dx, None,
         dx,  dx, None,
         dx,  dx, None,
        -dx, -dx, None
    ]
    box_y = [
        -dy, -dy,  dy,  dy, -dy, None,
        -dy, -dy,  dy,  dy, -dy, None,
        -dy, -dy, None,
        -dy, -dy, None,
         dy,  dy, None,
         dy,  dy, None
    ]
    box_z = [
        -dz, -dz, -dz, -dz, -dz, None,
         dz,  dz,  dz,  dz,  dz, None,
        -dz,  dz, None,
        -dz,  dz, None,
        -dz,  dz, None,
        -dz,  dz, None
    ]
    fig.add_trace(go.Scatter3d(
        x=box_x,
        y=box_y,
        z=box_z,
        mode='lines',
        line=dict(color=box_color, width=4),
        name=f"Object Box ({'Fits FOV' if object_fits else 'FOV Clipping Risk'})"
    ))

    # 6. Rotation Axis: dashed cyan vertical line along Z-axis from -max_extent*1.5 to +max_extent*1.5
    max_extent = max(dx, dy, dz, 1.0)
    fig.add_trace(go.Scatter3d(
        x=[0.0, 0.0],
        y=[0.0, 0.0],
        z=[-max_extent * 1.5, max_extent * 1.5],
        mode='lines',
        line=dict(color='#06B6D4', width=3, dash='dash'),
        name="Rotation Axis (Z)"
    ))

    # 7. Origin Marker: small white dot at (0,0,0) with label "Object Center"
    fig.add_trace(go.Scatter3d(
        x=[0.0],
        y=[0.0],
        z=[0.0],
        mode='markers+text',
        marker=dict(size=4, color='#FFFFFF'),
        text=["Object Center"],
        textposition="bottom center",
        textfont=dict(color="#E2E8F0", size=10),
        name="Object Center",
        hoverinfo="text"
    ))

    # Plotly Layout Settings
    fig.update_layout(
        scene=dict(
            xaxis_title="X — Beam Axis (cm)",
            yaxis_title="Y — Lateral (cm)",
            zaxis_title="Z — Vertical (cm)",
            aspectmode="data",
            camera=dict(
                eye=dict(x=0.0, y=-2.0, z=0.25),
                center=dict(x=0.0, y=0.0, z=0.0),
                up=dict(x=0.0, y=0.0, z=1.0)
            ),
            bgcolor="#0F172A",
            xaxis=dict(backgroundcolor="#1E293B", gridcolor="#334155", color="#94A3B8"),
            yaxis=dict(backgroundcolor="#1E293B", gridcolor="#334155", color="#94A3B8"),
            zaxis=dict(backgroundcolor="#1E293B", gridcolor="#334155", color="#94A3B8"),
        ),
        paper_bgcolor="#0F172A",
        font_color="#E2E8F0",
        margin=dict(l=0, r=0, t=30, b=0),
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color="#CBD5E1", size=11)
        )
    )

    return fig

# Filter physical attenuation database for spectral preview
FILTER_DATA = {
    "Al": {"name": "Aluminium (Al)", "density": 2.70, "a": 4200.0, "p": -2.85, "c": 0.04, "k_edge": None, "k_jump": 1.0},
    "Cu": {"name": "Copper (Cu)", "density": 8.96, "a": 38000.0, "p": -2.85, "c": 0.07, "k_edge": 8.98, "k_jump": 8.0},
    "Sn": {"name": "Tin (Sn)", "density": 7.31, "a": 25000.0, "p": -2.85, "c": 0.12, "k_edge": 29.2, "k_jump": 5.8},
    "Ti": {"name": "Titanium (Ti)", "density": 4.51, "a": 18000.0, "p": -2.85, "c": 0.06, "k_edge": 4.97, "k_jump": 8.5},
    "Mo": {"name": "Molybdenum (Mo)", "density": 10.2, "a": 19000.0, "p": -2.85, "c": 0.10, "k_edge": 20.0, "k_jump": 6.8},
    "Rh": {"name": "Rhodium (Rh)", "density": 12.4, "a": 21000.0, "p": -2.85, "c": 0.11, "k_edge": 23.2, "k_jump": 6.4},
    "Ag": {"name": "Silver (Ag)", "density": 10.49, "a": 23000.0, "p": -2.85, "c": 0.11, "k_edge": 25.5, "k_jump": 6.2},
    "Fe": {"name": "Iron (Fe)", "density": 7.87, "a": 28000.0, "p": -2.85, "c": 0.07, "k_edge": 7.11, "k_jump": 8.2},
    "Pb": {"name": "Lead (Pb)", "density": 11.34, "a": 55000.0, "p": -2.85, "c": 0.20, "k_edge": 88.0, "k_jump": 4.5},
}

def calc_filter_mu_mm(mat_code: str, energy_arr: np.ndarray) -> np.ndarray:
    """Calculate linear attenuation coefficient mu(E) in mm^-1 for a given filter material."""
    if mat_code not in FILTER_DATA:
        return np.zeros_like(energy_arr)
    d = FILTER_DATA[mat_code]
    photo = d["a"] * (energy_arr ** d["p"])
    if d["k_edge"] is not None:
        jump_mask = energy_arr >= d["k_edge"]
        photo = np.where(jump_mask, photo * d["k_jump"], photo)
    mu_cm = d["density"] * (photo + d["c"])
    return mu_cm / 10.0  # Convert cm^-1 to mm^-1

def plot_beam_spectrum(kvp: float, min_kev: float, num_bins: int, filter_mat: str = "None", filter_thick: float = 0.0, filters=None):
    # Generate Kramers law X-ray spectrum plot (unfiltered vs filtered) with multi-filter support
    fig, ax = plt.subplots(figsize=(4.8, 2.5))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F8FAFC")
    
    energies = np.linspace(min_kev, kvp, 120)
    brem = 74.0 * (kvp - energies)
    window_t = 1.0 / (1.0 + np.exp(-0.35 * (energies - min_kev * 1.25)))
    unfiltered = np.maximum(0.0, brem * window_t)
    
    # Characteristic lines for Tungsten anode (K-alpha ~59.3 keV, K-beta ~67.2 keV)
    if kvp > 69.5:
        k_alpha = np.exp(-0.5 * ((energies - 59.3) / 1.1) ** 2) * (brem.max() * 0.40)
        k_beta = np.exp(-0.5 * ((energies - 67.2) / 1.1) ** 2) * (brem.max() * 0.22)
        unfiltered += k_alpha + k_beta
        
    peak = unfiltered.max() if unfiltered.max() > 0 else 1.0
    unfiltered_norm = (unfiltered / peak) * 100.0
    
    ax.plot(energies, unfiltered_norm, color="#94A3B8", linestyle="--", lw=1.5, label="Raw Kramers")
    
    # Assemble active filters list
    active_flts = []
    if filters is not None:
        active_flts = filters
    elif filter_mat != "None" and filter_thick > 0.0:
        active_flts = [(filter_mat, filter_thick)]
        
    # Calculate cumulative filter transmission: T(E) = exp(-sum(mu_i * t_i))
    total_atten = np.zeros_like(energies)
    label_parts = []
    for f_mat, f_thk in active_flts:
        if f_mat != "None" and f_thk > 0.0:
            mu_mm = calc_filter_mu_mm(f_mat, energies)
            total_atten += mu_mm * f_thk
            label_parts.append(f"{f_thk:.2g}mm {f_mat}")
            
    if label_parts:
        trans = np.exp(-total_atten)
        filtered_norm = unfiltered_norm * trans
        filter_label = " + ".join(label_parts)
        ax.plot(energies, filtered_norm, color="#0284C7", lw=2.0, label=f"Filtered ({filter_label})")
        ax.fill_between(energies, filtered_norm, color="#0EA5E9", alpha=0.25)
    else:
        ax.fill_between(energies, unfiltered_norm, color="#94A3B8", alpha=0.15)
        
    # Mark discrete energy bins sampled
    discrete_bins = np.linspace(min_kev, kvp, num_bins)
    ax.scatter(discrete_bins, np.zeros_like(discrete_bins) + 3, color="#0284C7", s=10, zorder=4, label=f"{num_bins} Bins")
    
    ax.set_title(f"X-Ray Spectrum: {kvp:.0f} kVp (W Anode)", fontsize=8.5, fontweight="bold", color="#0F172A", pad=5)
    ax.set_xlabel("Photon Energy (keV)", fontsize=7.5, color="#475569")
    ax.set_ylabel("Relative Intensity (%)", fontsize=7.5, color="#475569")
    ax.tick_params(labelsize=7, colors="#475569")
    ax.grid(True, linestyle=":", alpha=0.6, color="#CBD5E1")
    ax.legend(fontsize=6.5, loc="upper right", framealpha=0.9)
    plt.tight_layout()
    return fig

# -------------------------------------------------------------------------------------------
# Session State Initialization (Guards against unwanted re-runs)

if "parts_config" not in st.session_state:
    st.session_state.parts_config = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "cached_projections" not in st.session_state:
    st.session_state.cached_projections = {}
if "optimal_pixel_size" not in st.session_state:
    st.session_state.optimal_pixel_size = 0.50
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

def clear_part_widget_keys():
    """Remove obsolete part widget keys from session state to prevent state leaking."""
    keys_to_del = [k for k in st.session_state.keys() if any(k.startswith(pfx) for pfx in ["mat_sel_", "density_", "center_", "scale_", "del_"])]
    for k in keys_to_del:
        del st.session_state[k]

def on_material_select(part_id: str):
    """Callback when user selects a new material in dropdown: auto-updates density input."""
    mat_key = f"mat_sel_{part_id}"
    dens_key = f"density_{part_id}"
    if mat_key in st.session_state:
        chosen_mat = st.session_state[mat_key]
        if chosen_mat in MATERIALS_DB:
            new_density = float(MATERIALS_DB[chosen_mat].get("density", 2.70))
            st.session_state[dens_key] = new_density
            for p in st.session_state.parts_config:
                if p["id"] == part_id:
                    p["material"] = chosen_mat
                    p["density"] = new_density
                    p["material_type"] = MATERIALS_DB[chosen_mat].get("type", "element")
                    p["formula"] = MATERIALS_DB[chosen_mat].get("formula", chosen_mat)
                    break

def load_preset(preset_name: str):
    """Load curated benchmark presets."""
    clear_part_widget_keys()
    # Reset previous simulation data so tabs 2, 3, 4 don't show stale results from old preset
    st.session_state.last_result = None
    st.session_state.cached_projections = {}
    if "uploader_key" in st.session_state:
        st.session_state.uploader_key += 1
    
    if preset_name == "Multi-Material Cylinder Phantom":
        st.session_state.parts_config = [
            {
                "id": str(uuid.uuid4())[:8],
                "label": "PMMA_Matrix",
                "stl_path": os.path.join(SAMPLE_DIR, "matrix_cylinder.stl"),
                "material": "PMMA (Acrylic)",
                "material_type": "compound",
                "formula": "C5H8O2",
                "density": 1.19,
                "center_mesh": True,
                "scale": 1.0,
                "translate": [0.0, 0.0, 0.0]
            },
            {
                "id": str(uuid.uuid4())[:8],
                "label": "Copper_Insert",
                "stl_path": os.path.join(SAMPLE_DIR, "insert_copper_rod.stl"),
                "material": "Copper",
                "material_type": "element",
                "formula": "Cu",
                "density": 8.96,
                "center_mesh": False,
                "scale": 1.0,
                "translate": [0.0, 0.0, 0.0]
            },
            {
                "id": str(uuid.uuid4())[:8],
                "label": "Aluminium_Insert",
                "stl_path": os.path.join(SAMPLE_DIR, "insert_aluminium_rod.stl"),
                "material": "Aluminium",
                "material_type": "element",
                "formula": "Al",
                "density": 2.70,
                "center_mesh": False,
                "scale": 1.0,
                "translate": [0.0, 0.0, 0.0]
            },
            {
                "id": str(uuid.uuid4())[:8],
                "label": "Titanium_Insert",
                "stl_path": os.path.join(SAMPLE_DIR, "insert_titanium_core.stl"),
                "material": "Titanium",
                "material_type": "element",
                "formula": "Ti",
                "density": 4.51,
                "center_mesh": False,
                "scale": 1.0,
                "translate": [0.0, 0.0, 0.0]
            }
        ]
        st.session_state.optimal_pixel_size = 0.40

    elif preset_name == "Cube with Sphere Core (Multi-Material)":
        cube_path = os.path.join(SAMPLE_DIR, "outer_cube_matrix.stl")
        sphere_path = os.path.join(SAMPLE_DIR, "inner_sphere_core.stl")
        satellite_path = os.path.join(SAMPLE_DIR, "satellite_sphere_defect.stl")
        st.session_state.parts_config = [
            {
                "id": str(uuid.uuid4())[:8],
                "label": "Cube_Container",
                "stl_path": cube_path,
                "material": "PMMA (Acrylic)",
                "material_type": "compound",
                "formula": "C5H8O2",
                "density": 1.19,
                "center_mesh": True,
                "scale": 1.0,
                "translate": [0.0, 0.0, 0.0]
            },
            {
                "id": str(uuid.uuid4())[:8],
                "label": "Sphere_Core",
                "stl_path": sphere_path,
                "material": "Copper",
                "material_type": "element",
                "formula": "Cu",
                "density": 8.96,
                "center_mesh": False,
                "scale": 1.0,
                "translate": [0.0, 0.0, 0.0]
            },
            {
                "id": str(uuid.uuid4())[:8],
                "label": "Satellite_Sphere",
                "stl_path": satellite_path,
                "material": "Titanium",
                "material_type": "element",
                "formula": "Ti",
                "density": 4.51,
                "center_mesh": False,
                "scale": 1.0,
                "translate": [0.0, 0.0, 0.0]
            }
        ]
        st.session_state.optimal_pixel_size = 0.25

    elif preset_name == "Aluminium Lizard (Single Material)":
        lizard_path = os.path.join(BASE_DIR, "lizard.stl")
        st.session_state.parts_config = [
            {
                "id": str(uuid.uuid4())[:8],
                "label": "Lizard_Aluminium",
                "stl_path": lizard_path,
                "material": "Aluminium",
                "material_type": "element",
                "formula": "Al",
                "density": 2.70,
                "center_mesh": True,
                "scale": 1.0,
                "translate": [0.0, 0.0, 0.0]
            }
        ]
        st.session_state.optimal_pixel_size = 1.38

    elif preset_name == "Welsh Dragon":
        from gvxrPython3 import gvxr
        dragon_path = os.path.join(os.path.dirname(gvxr.__file__), "welsh-dragon-small.stl")
        st.session_state.parts_config = [
            {
                "id": str(uuid.uuid4())[:8],
                "label": "Welsh_Dragon",
                "stl_path": dragon_path,
                "material": "PMMA (Acrylic)",
                "material_type": "compound",
                "formula": "C5H8O2",
                "density": 1.19,
                "center_mesh": True,
                "scale": 1.0,
                "translate": [0.0, 0.0, 0.0]
            }
        ]
        st.session_state.optimal_pixel_size = 0.85

    elif preset_name == "Custom Configuration":
        st.session_state.parts_config = []
        st.session_state.optimal_pixel_size = 0.50

# Default to Multi-Material Cylinder ONLY on initial app startup
if "app_initialized" not in st.session_state:
    st.session_state.app_initialized = True
    load_preset("Multi-Material Cylinder Phantom")

# -------------------------------------------------------------------------------------------------------
# SIDEBAR: CT SCANNER PARAMETER CONTROLS

with st.sidebar:
    st.markdown('<div class="sidebar-section-title">Benchmark Presets</div>', unsafe_allow_html=True)
    # Drop down list for benchmark presets
    selected_preset = st.selectbox(
        "Select CT Benchmark",
        [
            "Multi-Material Cylinder Phantom",
            "Cube with Sphere Core (Multi-Material)",
            "Aluminium Lizard (Single Material)",
            "Welsh Dragon",
            "Custom Configuration"
        ]
    )
    if st.button("Load Preset", use_container_width=True):
        load_preset(selected_preset)
        st.rerun()

    # 1. Source Setup
    st.markdown('<div class="sidebar-section-title">1. X-Ray Source</div>', unsafe_allow_html=True)
    sod_val = st.number_input("Source-to-Object Distance SOD (cm)", min_value=5.0, max_value=200.0, value=40.0, step=5.0)
    source_type = st.radio("Source Geometry", ["Point Source", "Parallel Beam"], horizontal=True)

    beam_mode = st.radio("Beam Spectrum", ["Monochromatic", "Polychromatic"], horizontal=True)

    active_filters = []
    if beam_mode == "Monochromatic":
        mono_energy = st.slider("Beam Energy (keV)", min_value=10.0, max_value=200.0, value=80.0, step=5.0)
        mono_photons = st.number_input("Incident Photons per Pixel", min_value=100, max_value=100000, value=1000, step=500)
        poly_kvp, poly_bins, poly_min = 100.0, 20, 20.0
    else:  # Polychromatic
        poly_kvp = st.slider("Tube Potential kVp (kV)", min_value=30.0, max_value=220.0, value=100.0, step=5.0)
        poly_min = st.slider("Lower Cutoff Energy (keV)", min_value=10.0, max_value=float(poly_kvp - 10.0), value=20.0, step=5.0)
        poly_bins = st.slider("Energy Discretization Bins", min_value=5, max_value=50, value=20, step=5)
        
        st.markdown("**Physical Beam Filtration**")
        filter_options = [
            "None",
            "Al (Aluminium - Standard)",
            "Cu (Copper - Beam Hardening)",
            "Sn (Tin - High-kV CT)",
            "Ti (Titanium - Light)",
            "Mo (Molybdenum - Soft Tissue)",
            "Rh (Rhodium - Dual Energy)",
            "Ag (Silver)",
            "Fe (Iron / Steel)",
            "Pb (Lead)"
        ]
        filter_code_map = {
            "None": "None",
            "Al (Aluminium - Standard)": "Al",
            "Cu (Copper - Beam Hardening)": "Cu",
            "Sn (Tin - High-kV CT)": "Sn",
            "Ti (Titanium - Light)": "Ti",
            "Mo (Molybdenum - Soft Tissue)": "Mo",
            "Rh (Rhodium - Dual Energy)": "Rh",
            "Ag (Silver)": "Ag",
            "Fe (Iron / Steel)": "Fe",
            "Pb (Lead)": "Pb"
        }
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            sel_f1 = st.selectbox("Primary Filter", filter_options, index=1)
            f1_mat = filter_code_map[sel_f1]
        with col_f2:
            default_f1_thk = 0.2 if f1_mat in ["Sn", "Pb"] else (0.5 if f1_mat in ["Cu", "Ag", "Mo", "Rh", "Fe"] else 1.0)
            f1_thick = st.number_input(
                "F1 Thick (mm)",
                min_value=0.0,
                max_value=15.0,
                value=default_f1_thk if f1_mat != "None" else 0.0,
                step=0.05 if f1_mat in ["Sn", "Cu", "Pb"] else 0.1
            )
        if f1_mat == "Sn":
            st.caption("💡 *Tin ($Z=50$) is very dense. Typical clinical CT thickness is 0.1 – 0.4 mm at ≥100 kVp.*")
        elif f1_mat == "Cu":
            st.caption("💡 *Copper ($Z=29$) is a strong hardener. Typical thickness is 0.1 – 0.5 mm.*")
            
        enable_f2 = st.checkbox("Add Secondary Filter (Stacked)", value=False)
        f2_mat, f2_thick = "None", 0.0
        if enable_f2:
            col_f3, col_f4 = st.columns(2)
            with col_f3:
                sel_f2 = st.selectbox("Secondary Filter", [opt for opt in filter_options if opt != "None"], index=1)
                f2_mat = filter_code_map[sel_f2]
            with col_f4:
                f2_thick = st.number_input("F2 Thick (mm)", min_value=0.01, max_value=10.0, value=0.1, step=0.05)
                
        if f1_mat != "None" and f1_thick > 0.0:
            active_filters.append((f1_mat, f1_thick))
        if enable_f2 and f2_mat != "None" and f2_thick > 0.0:
            active_filters.append((f2_mat, f2_thick))

        mono_energy, mono_photons = 80.0, 1000

        # Live Beam Spectrum Preview
        fig_spec = plot_beam_spectrum(poly_kvp, poly_min, poly_bins, filters=active_filters)
        st.pyplot(fig_spec)
        plt.close(fig_spec)

    # 2. Detector Setup
    st.markdown('<div class="sidebar-section-title">2. Flat-Panel Detector</div>', unsafe_allow_html=True)
    sdd_offset = st.number_input("Object-to-Detector Distance ODD (cm)", min_value=5.0, max_value=200.0, value=15.0, step=5.0)
    sdd_val = sod_val + sdd_offset
    st.caption(f"Total SDD: **{sdd_val:.1f} cm** | Magnification: **{sdd_val / sod_val:.2f}x**")

    # Flexible Detector Resolution: Any resolution supported
    res_presets = [
        "256 × 256 (Fast / Default)",
        "128 × 128 (Ultra-Fast Preview)",
        "384 × 384 (Balanced)",
        "512 × 512 (Standard Diagnostic)",
        "640 × 480 (VGA 4:3)",
        "768 × 768 (Detailed)",
        "1024 × 1024 (High-Resolution)",
        "Custom Resolution..."
    ]
    
    res_choice = st.selectbox("Detector Resolution", res_presets, index=0)
    
    if res_choice == "Custom Resolution...":
        lock_sq = st.checkbox("Square Dimensions (Cols = Rows)", value=True)
        col_c, col_r = st.columns(2)
        with col_c:
            det_cols = st.number_input("Columns (Width px)", min_value=16, max_value=2048, value=300, step=10)
        with col_r:
            if lock_sq:
                det_rows = det_cols
                st.number_input("Rows (Height px)", value=det_cols, disabled=True)
            else:
                det_rows = st.number_input("Rows (Height px)", min_value=16, max_value=2048, value=300, step=10)
    elif "×" in res_choice:
        dims = res_choice.split("(")[0].strip().split("×")
        det_cols = int(dims[0].strip())
        det_rows = int(dims[1].strip())
    else:
        det_cols, det_rows = 256, 256

    st.caption(f"Active Grid: **{det_cols} × {det_rows}** pixels ({det_cols * det_rows / 1000:.1f} kpx)")

    col_px, col_btn = st.columns([2, 1])
    with col_px:
        pixel_size = st.number_input(
            "Pixel Size (mm)",
            min_value=0.01,
            max_value=10.0,
            value=float(st.session_state.optimal_pixel_size),
            step=0.05
        )
    with col_btn:
        st.write("")
        st.write("")
        if st.button("Auto FOV"):
            if st.session_state.parts_config:
                primary_path = st.session_state.parts_config[0]["stl_path"]
                if os.path.exists(primary_path):
                    opt_px = calculate_auto_framing(
                        primary_path,
                        sod_cm=sod_val,
                        sdd_cm=sdd_val,
                        rows=int(det_rows),
                        cols=int(det_cols)
                    )
                    st.session_state.optimal_pixel_size = opt_px
                    st.rerun()

    # 3. Acquisition Setup
    st.markdown('<div class="sidebar-section-title">3. CT Acquisition Trajectory</div>', unsafe_allow_html=True)
    num_proj = st.slider("Number of Projections", min_value=15, max_value=360, value=90, step=15)
    rot_angle = st.radio("Total Rotation Range", [180.0, 360.0], index=1, horizontal=True)
    apply_noise = st.checkbox("Simulate Poisson Photon Noise", value=(beam_mode == "Polychromatic"))
    noise_scale = 30000.0
    if apply_noise:
        noise_scale = st.slider("Incident Photon Flux (Scale)", min_value=5000.0, max_value=100000.0, value=30000.0, step=5000.0)

    output_dir_name = st.text_input("Output Directory Name", value="streamlit_run")
    target_output_dir = os.path.join(BASE_DIR, "output", output_dir_name)

    # Pre-Flight Summary Card & Estimated Size/Time
    est_size_str = estimate_output_size(int(det_rows), int(det_cols), num_proj, "uint16")
    est_time_sec = estimate_run_time_sec(num_proj, beam_mode)
    
    st.markdown('<div class="sidebar-section-title">Pre-Flight Summary</div>', unsafe_allow_html=True)
    filter_desc = (" + ".join([f"{t:.2g}mm {m}" for m, t in active_filters])) if active_filters else "No Filter"
    beam_summary_str = f"{mono_energy} keV mono" if beam_mode == "Monochromatic" else f"{poly_kvp} kVp ({filter_desc})"
    parts_count = len(st.session_state.parts_config)
    parts_status = f"{parts_count} material mesh(es)" if parts_count > 0 else "<span style='color:#DC2626; font-weight:700;'>0 parts (No STL loaded)</span>"
    st.markdown(f"""
    <div class="summary-card">
        <div style="font-weight:700; color:#0369A1; font-size:0.85rem; margin-bottom:4px;">Ready to Simulate:</div>
        <div style="font-size:0.78rem; color:#334155; line-height:1.45;">
            • <b>Scan:</b> {num_proj} projections over {rot_angle:.0f}°<br/>
            • <b>Beam:</b> {beam_summary_str}<br/>
            • <b>Detector:</b> {det_cols}×{det_rows} px ({pixel_size} mm/px)<br/>
            • <b>Parts:</b> {parts_status}<br/>
            • <b>Est. Output:</b> <b>{est_size_str}</b> | <b>~{est_time_sec:.1f} s</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    run_clicked = st.button("Run CT Simulation", type="primary", use_container_width=True)
    if run_clicked and not st.session_state.parts_config:
        st.sidebar.error("No STL file uploaded or loaded! Please add at least one STL mesh part before running.")


# ---------------------------------------------------------------------------------------
# MAIN PAGE CONTENT
# ----------------------------------------------------------------------------------------
st.markdown("""
<div class="webct-header">
    <div class="main-title">Multi-material CT Simulation and Data Generation</div>
    <div class="sub-title">Interactive Synthetic CT Generator &middot; Multi-Material Attenuation &middot; gVXR-Based X-ray Simulation</div>
</div>
""", unsafe_allow_html=True)

tab_setup, tab_visualize, tab_projections, tab_metadata, tab_3d = st.tabs([
    "1. Multi-Material Sample Setup",
    "2. Preview & Sinogram Analysis",
    "3. Interactive Projection Viewer",
    "4. Metadata & Data Export",
    "5. 3D Scanner Geometry"
])

# ------------------------------------------------------------------------------
# TAB 1: 3D Object & Material Assignment
# ------------------------------------------------------------------------------
with tab_setup:
    st.markdown('<div class="webct-section-title">Multi-Material Sample Assembly</div>', unsafe_allow_html=True)
    st.caption("Configure each 3D mesh part, assign distinct elements/compounds from the materials database, or upload custom STL files.")

    # Upload custom STLs
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    if st.session_state.get("upload_success_msg"):
        st.success(st.session_state.pop("upload_success_msg"))

    with st.expander("Upload Custom STL Files", expanded=False):
        uploaded_files = st.file_uploader(
            "Upload STL Files for Assembly",
            type=["stl"],
            accept_multiple_files=True,
            key=f"stl_uploader_{st.session_state.uploader_key}"
        )
        if uploaded_files:
            newly_added = 0
            for uf in uploaded_files:
                dest_path = os.path.join(UPLOAD_DIR, uf.name)
                with open(dest_path, "wb") as f:
                    f.write(uf.getbuffer())
                
                if not any(p.get("stl_path") == dest_path for p in st.session_state.parts_config):
                    new_id = str(uuid.uuid4())[:8]
                    st.session_state.parts_config.append({
                        "id": new_id,
                        "label": os.path.splitext(uf.name)[0],
                        "stl_path": dest_path,
                        "material": "Aluminium",
                        "material_type": "element",
                        "formula": "Al",
                        "density": 2.70,
                        "center_mesh": True,
                        "scale": 1.0,
                        "translate": [0.0, 0.0, 0.0]
                    })
                    newly_added += 1

            # Invalidate stale simulation results so Tabs 2, 3, 4 reflect the new setup
            st.session_state.last_result = None
            st.session_state.cached_projections = {}

            if newly_added > 0:
                st.session_state.upload_success_msg = f"Added {newly_added} custom STL part(s)! Configured below."
            else:
                st.session_state.upload_success_msg = f"Updated {len(uploaded_files)} custom STL file(s) on disk."

            # Increment uploader key so that the uploader widget is cleared and doesn't re-add deleted files
            st.session_state.uploader_key += 1
            st.rerun()

    # Quick picking already available STLs on device and assigning aluminium as default
    available_ws_stls = sorted(glob.glob(os.path.join(BASE_DIR, "*.stl")) + glob.glob(os.path.join(SAMPLE_DIR, "*.stl")))
    ws_options = ["-- Select an STL to Add --"] + [os.path.relpath(p, BASE_DIR) for p in available_ws_stls]
    
    col_add1, col_add2, col_clear = st.columns([3, 1, 1])
    with col_add1:
        chosen_add = st.selectbox("Add STL from Workspace", ws_options)
    with col_add2:
        st.write("")
        st.write("")
        if st.button("Add Part") and chosen_add != ws_options[0]:
            full_path = os.path.join(BASE_DIR, chosen_add)
            part_label = os.path.splitext(os.path.basename(full_path))[0]
            new_id = str(uuid.uuid4())[:8]
            st.session_state.parts_config.append({
                "id": new_id,
                "label": part_label,
                "stl_path": full_path,
                "material": "Aluminium",
                "material_type": "element",
                "formula": "Al",
                "density": 2.70,
                "center_mesh": True,
                "scale": 1.0,
                "translate": [0.0, 0.0, 0.0]
            })
            st.session_state.last_result = None
            st.session_state.cached_projections = {}
            st.rerun()
    with col_clear:
        st.write("")
        st.write("")
        if st.button("Clear All"):
            clear_part_widget_keys()
            st.session_state.parts_config = []
            st.session_state.last_result = None
            st.session_state.cached_projections = {}
            if "uploader_key" in st.session_state:
                st.session_state.uploader_key += 1
            st.rerun()

    st.markdown("### Configured Mesh Parts & Materials")
    # if no STL file chosen
    if not st.session_state.parts_config:
        st.warning("No STL mesh parts currently loaded. Please upload a custom STL file above, choose an STL from the workspace, or select a preset from the sidebar.")

    # Track parts marked for removal and fetch available material presets
    parts_to_remove = []
    mat_options = list(MATERIALS_DB.keys())

    for idx, part in enumerate(st.session_state.parts_config):
        p_id = part["id"]
        mat_key = f"mat_sel_{p_id}"
        dens_key = f"density_{p_id}"
        cent_key = f"center_{p_id}"
        scale_key = f"scale_{p_id}"
    # Initialize session state defaults for this part if not already present
        if mat_key not in st.session_state:
            st.session_state[mat_key] = part["material"]
        if dens_key not in st.session_state:
            st.session_state[dens_key] = float(part["density"])
        if cent_key not in st.session_state:
            st.session_state[cent_key] = bool(part["center_mesh"])
        if scale_key not in st.session_state:
            st.session_state[scale_key] = float(part.get("scale", 1.0))

        # Retrieve current material details and badge display metadata
        cur_mat_name = st.session_state.get(mat_key, part["material"])
        cat_badge = get_material_badge_html(cur_mat_name)
        cat_text = MATERIALS_DB.get(cur_mat_name, {}).get("category", "General")

        # Render an expander card with a 4-column layout for editing part properties
        with st.expander(f"Part {idx + 1}: **{part['label']}** — {cur_mat_name} ({cat_text})", expanded=True):
            col1, col2, col3, col4 = st.columns([2.2, 1.8, 1.8, 1.0])

        # Column 1: Select material preset and update associated chemical metadata
            with col1:
                cur_mat = st.selectbox(
                    "Material Preset",
                    mat_options,
                    key=mat_key,
                    on_change=on_material_select,
                    args=(p_id,)
                )
                mat_meta = MATERIALS_DB.get(cur_mat, {})
                part["material"] = cur_mat
                part["material_type"] = mat_meta.get("type", "element")
                part["formula"] = mat_meta.get("formula", cur_mat)
                st.markdown(f"Category: {cat_badge} &nbsp;|&nbsp; Formula: <b>{part['formula']}</b>", unsafe_allow_html=True)

            # Column 2: Configure physical density 
            with col2:
                cur_dens = st.number_input(
                    "Density (g/cm³)",
                    min_value=0.0001,
                    max_value=25.0,
                    step=0.1,
                    key=dens_key
                )
                part["density"] = cur_dens
                st.caption(f"Physical Type: **{part['material_type'].capitalize()}**")

            # Column 3: Origin centreing and adjusting uniform scale
            with col3:
                cur_center = st.checkbox("Center on Origin (0,0,0)", key=cent_key)
                part["center_mesh"] = cur_center

                cur_scale = st.number_input("Uniform Scale", min_value=0.05, max_value=10.0, step=0.1, key=scale_key)
                part["scale"] = cur_scale

            # Column 4: Delete button
            with col4:
                st.write("")
                st.write("")
                if st.button("Delete", key=f"del_{p_id}"):
                    parts_to_remove.append(idx)

    # Purge widget state keys for deleted parts and refresh the UI
    if parts_to_remove:
        for idx in sorted(parts_to_remove, reverse=True):
            removed_part = st.session_state.parts_config[idx]
            removed_id = removed_part["id"]
            for pfx in ["mat_sel_", "density_", "center_", "scale_", "del_"]:
                st.session_state.pop(f"{pfx}{removed_id}", None)
            st.session_state.parts_config.pop(idx)
            # Remove uploaded file from disk if no other part is using it
            if removed_part.get("stl_path", "").startswith(UPLOAD_DIR):
                if not any(p.get("stl_path") == removed_part["stl_path"] for p in st.session_state.parts_config):
                    try:
                        if os.path.exists(removed_part["stl_path"]):
                            os.remove(removed_part["stl_path"])
                    except Exception:
                        pass
        st.session_state.last_result = None
        st.session_state.cached_projections = {}
        st.rerun()


# ------------------------------------------------------------------------------
# SIMULATION EXECUTION TRIGGER (Runs in dedicated subprocess for GLFW thread safety)
# ------------------------------------------------------------------------------
if run_clicked:
    if not st.session_state.parts_config:
        st.error("Cannot run simulation: No STL file uploaded or loaded! Please add at least one STL mesh part in 'Sample Setup' or load a preset.")
    else:
        # Configure optional X-ray beam filtration if selected
        filters = []
        for f_mat, f_thk in active_filters:
            if f_mat != "None" and f_thk > 0.0:
                filters.append(FilterConfig(material=f_mat, thickness=float(f_thk), unit="mm"))
        # Build PartConfig objects using active UI overriding
        sim_parts = []
        for p in st.session_state.parts_config:
            scale_val = float(st.session_state.get(f"scale_{p['id']}", p.get("scale", 1.0)))
            sim_parts.append(PartConfig(
                label=p["label"],
                stl_path=p["stl_path"],
                material_type=p["material_type"],
                material_identifier=p["formula"],
                density=float(st.session_state.get(f"density_{p['id']}", p["density"])),
                center_mesh=bool(st.session_state.get(f"center_{p['id']}", p["center_mesh"])),
                scale=[scale_val, scale_val, scale_val],
                translate=p.get("translate", [0.0, 0.0, 0.0])
            ))

        # Assemble the full SimulationConfig for source, detector, and scan acquisition
        config = SimulationConfig(
            output_dir=target_output_dir,
            parts=sim_parts,
            source=SourceConfig(
                position=[-float(sod_val), 0.0, 0.0],
                position_unit="cm",
                source_shape="parallel" if source_type == "Parallel Beam" else "point",
                beam_type="polychromatic" if beam_mode == "Polychromatic" else "monochromatic",
                mono_energy_kev=float(mono_energy),
                mono_photon_count=int(mono_photons),
                poly_tube_voltage_kv=float(poly_kvp),
                poly_min_energy_kev=float(poly_min),
                poly_energy_bins=int(poly_bins),
                poly_total_photons=40000,
                filters=filters
            ),

            detector=DetectorConfig(
                position=[float(sdd_offset), 0.0, 0.0],
                position_unit="cm",
                rows=int(det_rows),
                cols=int(det_cols),
                pixel_size_x=float(pixel_size),
                pixel_size_y=float(pixel_size),
                pixel_size_unit="mm"
            ),

            acquisition=AcquisitionConfig(
                num_projections=int(num_proj),
                total_angle_deg=float(rot_angle),
                apply_poisson_noise=bool(apply_noise),
                photon_noise_scale=float(noise_scale)
            )
        )

        # Set up Streamlit progress indicators and serialize config to a temporary JSON file
        st.markdown("### Simulation in Progress...")
        prog_bar = st.progress(0.0)
        status_box = st.empty()

        temp_config_path = os.path.join(BASE_DIR, "temp_run_config.json")
        config.to_json(temp_config_path)

        # Spawn simulation as a separate background process to avoid blocking the UI
        cmd = [sys.executable, os.path.join(BASE_DIR, "run_sim_task.py"), "--config", temp_config_path]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=BASE_DIR)

        sim_failed = False
        error_lines = []

        # Stream stdout in real time to update the UI progress bar and status message
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            line_str = line.strip()
            if line_str.startswith("PROGRESS:"):
                try:
                    parts = line_str.split(":", 2)
                    progress_info = parts[1].strip()
                    msg = parts[2].strip() if len(parts) > 2 else ""
                    cur_step, tot_step = map(int, progress_info.split("/"))
                    pct = cur_step / tot_step
                    prog_bar.progress(pct)
                    status_box.markdown(f"**Step {cur_step}/{tot_step}**: {msg}")
                except Exception:
                    pass
            elif "SIMULATION_FAILED:" in line_str:
                sim_failed = True
                error_lines.append(line_str)
            elif line_str:
                error_lines.append(line_str)

        # Wait for the subprocess to finish and capture the exit code
        rc = proc.wait()

        # On successful execution, parse summary metrics and cache the results in session state
        if rc == 0 and not sim_failed:
            meta_path = os.path.join(target_output_dir, "simulation_config.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)
                exec_sum = meta_data.get("execution_summary", {})
                duration = exec_sum.get("duration_seconds", 0.0)
                fps = exec_sum.get("projections_per_second", 0.0)
            else:
                duration, fps = 0.0, 0.0

        # Store result paths and metadata for previewing across other UI tabs
            st.session_state.last_result = {
                "output_dir": target_output_dir,
                "preview_path": os.path.join(target_output_dir, "preview.png"),
                "sinogram_path": os.path.join(target_output_dir, "sinogram.tif"),
                "metadata_path": meta_path,
                "num_projections": int(num_proj),
                "duration_sec": duration,
                "fps": fps
            }
            st.session_state.cached_projections = {}
            st.success(f"Simulation completed successfully in **{duration:.2f} s** ({fps:.1f} projections/sec)!")
        else:
            err_msg = "\n".join(error_lines[-5:]) if error_lines else "Unknown error occurred"
            st.error(f"Simulation failed:\n```\n{err_msg}\n```")


# ------------------------------------------------------------------------------
# TAB 2: Preview & Sinogram Analysis
# ------------------------------------------------------------------------------
with tab_visualize:
    # Option to load any saved run from disk
    output_root = os.path.join(BASE_DIR, "output")
    if os.path.exists(output_root):
        past_runs = [d for d in sorted(os.listdir(output_root)) if os.path.isdir(os.path.join(output_root, d)) and os.path.exists(os.path.join(output_root, d, "simulation_config.json"))]
        if past_runs:
            with st.expander("📂 Load Previous Saved Run from Disk", expanded=False):
                col_pr1, col_pr2 = st.columns([3, 1])
                with col_pr1:
                    chosen_run = st.selectbox("Select Saved Run Folder", past_runs, key="load_past_run_sel")
                with col_pr2:
                    st.write("")
                    st.write("")
                    if st.button("Load Saved Run", key="btn_load_past_run"):
                        run_dir = os.path.join(output_root, chosen_run)
                        meta_file = os.path.join(run_dir, "simulation_config.json")
                        prev_meta = {}
                        if os.path.exists(meta_file):
                            try:
                                with open(meta_file, "r", encoding="utf-8") as f:
                                    prev_meta = json.load(f)
                            except Exception:
                                pass
                        exec_s = prev_meta.get("execution_summary", {})
                        acq_s = prev_meta.get("acquisition", {})
                        st.session_state.last_result = {
                            "output_dir": run_dir,
                            "preview_path": os.path.join(run_dir, "preview.png"),
                            "sinogram_path": os.path.join(run_dir, "sinogram.tif"),
                            "metadata_path": meta_file,
                            "num_projections": int(acq_s.get("num_projections", len(glob.glob(os.path.join(run_dir, "proj_*.tif"))))),
                            "duration_sec": float(exec_s.get("duration_seconds", 0.0)),
                            "fps": float(exec_s.get("projections_per_second", 0.0))
                        }
                        st.session_state.cached_projections = {}
                        st.rerun()

    res = st.session_state.last_result
    if not res:
        st.info("No simulation run yet for this setup. Click 'Run CT Simulation' in the sidebar to generate projections and sinograms.")
    else:
        st.markdown('<div class="webct-section-title">Simulation Results & CT Inspection</div>', unsafe_allow_html=True)

        # Display key scan execution metrics across a 4-column summary bar
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Projections", f"{res['num_projections']} frames")
        m2.metric("Duration", f"{res['duration_sec']:.2f} s")
        m3.metric("Throughput", f"{res['fps']:.1f} FPS")
        m4.metric("Output Directory", os.path.basename(res['output_dir']))

        st.markdown("---")

        # Render the preview image (0° projection, sinogram, and profile)
        preview_img_path = res["preview_path"]
        if os.path.exists(preview_img_path):
            st.markdown('<div class="webct-card">', unsafe_allow_html=True)
            st.image(preview_img_path, caption="CT Comprehensive Inspection: 0° Projection, Sinogram, & Central Attenuation Profile", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        # Split layout into a sinogram visualization panel and a summary statistics card
        col_sino, col_stat = st.columns([2, 1])
        with col_sino:
            sino_path = res["sinogram_path"]
            if os.path.exists(sino_path):
                sino_arr = tifffile.imread(sino_path)
                st.caption(f"Sinogram Dimensions: **{sino_arr.shape[0]} angles × {sino_arr.shape[1]} detector columns**")

                # Plot the sinogram line integrals using the inferno colormap
                fig, ax = plt.subplots(figsize=(7.5, 3.4))
                fig.patch.set_facecolor("#FFFFFF")
                ax.set_facecolor("#0F172A")
                im = ax.imshow(sino_arr, cmap="inferno", aspect="auto", extent=[0, sino_arr.shape[1], 360, 0])
                ax.set_title("Sinogram (Attenuation Line Integrals)", fontsize=10, fontweight="bold", color="#0F172A")
                ax.set_xlabel("Detector Column (pixels)", fontsize=8.5, color="#475569")
                ax.set_ylabel("Rotation Angle (°)", fontsize=8.5, color="#475569")
                ax.tick_params(labelsize=8, colors="#475569")
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("Line Integral -ln(I/I0)", fontsize=8, color="#475569")
                cbar.ax.tick_params(labelsize=7, colors="#475569")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

         # Compute and display basic physical attenuation statistics from the sinogram array
        with col_stat:
            st.markdown('<div class="webct-card">', unsafe_allow_html=True)
            st.markdown("#### Physical Attenuation")
            if os.path.exists(res["sinogram_path"]):
                s_data = tifffile.imread(res["sinogram_path"])
                st.write(f"• **Peak Integral**: `{s_data.max():.2f}`")
                st.write(f"• **Mean Attenuation**: `{s_data.mean():.2f}`")
                st.write(f"• **Min Attenuation**: `{s_data.min():.2f}`")
                st.write(f"• **Dynamic Range**: `{s_data.max() - s_data.min():.2f}`")
            st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# TAB 3: Interactive Projection Scrubber
# ------------------------------------------------------------------------------
with tab_projections:
    res = st.session_state.last_result
    if not res:
        st.info("No projection data available for the current setup. Run a CT simulation first to interactively scrub through projection angles.")
    else:
        st.markdown('<div class="webct-section-title">Interactive CT Projection Viewer</div>', unsafe_allow_html=True)
        out_dir = res["output_dir"]
        # Find and sort all individual projection TIFF files in the output directory
        proj_files = sorted(glob.glob(os.path.join(out_dir, "proj_*.tif")))
        
        if not proj_files:
            st.warning("No projection TIFF files found in output directory.")
        else:
            total_frames = len(proj_files)
            slider_col, play_col = st.columns([3, 1])
            # Slider control for scrubbing turntable angles frame-by-frame
            with slider_col:
                frame_idx = st.slider(
                    "Scrub Turntable Projection Angle",
                    min_value=0,
                    max_value=total_frames - 1,
                    value=0,
                    step=1
                )

            # Calculate rotation angle corresponding to the selected frame index
            angle_per_frame = 360.0 / total_frames
            cur_angle = frame_idx * angle_per_frame
            with play_col:
                st.write("")
                st.markdown(f'<div style="margin-top:15px;"><span class="header-badge" style="font-size:0.85rem;">Frame {frame_idx:03d} &middot; {cur_angle:.1f}°</span></div>', unsafe_allow_html=True)

            # Cache loaded image in session state to enable smooth scrubbing without re-reading disk
            if frame_idx not in st.session_state.cached_projections:
                st.session_state.cached_projections[frame_idx] = tifffile.imread(proj_files[frame_idx])
            
            cur_proj = st.session_state.cached_projections[frame_idx]

            # Left column: 2D detector transmission image; Right column: 1D horizontal profile
            c_view1, c_view2 = st.columns([2, 1])
            with c_view1:
                col_m1, col_m2 = st.columns([2, 1])
                with col_m1:
                    disp_mode = st.radio(
                        "Display Contrast Mode",
                        ["Full 16-Bit Scale (0-65535)", "Auto-Contrast (Min-Max)", "Bone Radiograph (Inverted)"],
                        index=0,
                        horizontal=True
                    )
                with col_m2:
                    st.caption(f"Min: `{cur_proj.min()}` | Max: `{cur_proj.max()}`")

                fig_proj, ax_p = plt.subplots(figsize=(5.5, 5.5))
                fig_proj.patch.set_facecolor("#FFFFFF")
                
                if disp_mode == "Auto-Contrast (Min-Max)":
                    im_p = ax_p.imshow(cur_proj, cmap="gray")
                elif disp_mode == "Bone Radiograph (Inverted)":
                    im_p = ax_p.imshow(cur_proj, cmap="gray_r")
                else:
                    im_p = ax_p.imshow(cur_proj, cmap="gray", vmin=0, vmax=65535 if cur_proj.dtype == np.uint16 else None)

                ax_p.set_title(f"X-Ray Transmission at {cur_angle:.1f}° (Frame {frame_idx:03d})", fontsize=10, fontweight="bold", color="#0F172A")
                ax_p.set_xlabel("Detector Column (px)", fontsize=8.5, color="#475569")
                ax_p.set_ylabel("Detector Row (px)", fontsize=8.5, color="#475569")
                ax_p.tick_params(labelsize=8, colors="#475569")
                cbar_p = fig_proj.colorbar(im_p, ax=ax_p, fraction=0.046, pad=0.04)
                cbar_p.set_label("Transmission uint16", fontsize=8, color="#475569")
                cbar_p.ax.tick_params(labelsize=7, colors="#475569")
                plt.tight_layout()
                st.pyplot(fig_proj)
                plt.close(fig_proj)

            # Extract and plot a 1D transmission cross-section along a user-selected detector row
            with c_view2:
                st.markdown('<div class="webct-card">', unsafe_allow_html=True)
                st.markdown("#### 1D Intensity Cross-Section")
                slice_y = st.slider("Detector Row (Y Slice)", 0, cur_proj.shape[0] - 1, cur_proj.shape[0] // 2)
                row_profile = cur_proj[slice_y, :]

                # Render the 1D intensity profile line plot
                fig_prof, ax_prof = plt.subplots(figsize=(4.5, 3.2))
                fig_prof.patch.set_facecolor("#FFFFFF")
                ax_prof.set_facecolor("#F8FAFC")
                ax_prof.plot(row_profile, color="#0284C7", lw=1.8)
                ax_prof.fill_between(range(len(row_profile)), row_profile, color="#0EA5E9", alpha=0.15)
                ax_prof.set_title(f"Row {slice_y} Transmission Profile", fontsize=9, fontweight="bold", color="#0F172A")
                ax_prof.set_xlabel("Detector Column (px)", fontsize=8, color="#475569")
                ax_prof.set_ylabel("Pixel Value (ADU)", fontsize=8, color="#475569")
                ax_prof.tick_params(labelsize=7.5, colors="#475569")
                ax_prof.grid(True, linestyle=":", alpha=0.6, color="#CBD5E1")
                plt.tight_layout()
                st.pyplot(fig_prof)
                plt.close(fig_prof)
                st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# TAB 4: Metadata & Output Downloads
# ------------------------------------------------------------------------------
with tab_metadata:
    res = st.session_state.last_result
    st.markdown('<div class="webct-section-title">Metadata & Data Export</div>', unsafe_allow_html=True)

    # Load simulation configuration and summary metadata from output folder
    if res and os.path.exists(res.get("metadata_path", "")):
        with open(res["metadata_path"], "r", encoding="utf-8") as f:
            meta_json = json.load(f)

    # Download buttons for inspection images, sinograms, and configuration JSON
        st.markdown('<div class="webct-card">', unsafe_allow_html=True)
        st.markdown("#### Download Dataset Packages")
        d_col1, d_col2, d_col3 = st.columns(3)
        with d_col1:
            if os.path.exists(res["preview_path"]):
                with open(res["preview_path"], "rb") as f:
                    st.download_button("Download Preview Image", f.read(), file_name="preview.png", mime="image/png", use_container_width=True)
        with d_col2:
            if os.path.exists(res["sinogram_path"]):
                with open(res["sinogram_path"], "rb") as f:
                    st.download_button("Download Sinogram TIFF", f.read(), file_name="sinogram.tif", mime="image/tiff", use_container_width=True)
        with d_col3:
            with open(res["metadata_path"], "rb") as f:
                st.download_button("Download Metadata JSON", f.read(), file_name="simulation_config.json", mime="application/json", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Display full simulation configuration tree in an interactive JSON viewer
        st.markdown("#### Full Configuration JSON (`simulation_config.json`)")
        st.json(meta_json)
    else:
        st.info("No metadata available for the current setup. Run a CT simulation to generate and download metadata and datasets.")


# ------------------------------------------------------------------------------
# TAB 5: 3D Scanner Geometry & FOV Visualizer
# ------------------------------------------------------------------------------
with tab_3d:
    try:
        st.markdown('<div class="webct-section-title">3D CT Scanner Geometry & FOV Analysis</div>', unsafe_allow_html=True)

        stl_bounds = (2.0, 2.0, 1.5)  # default fallback
        if st.session_state.parts_config:
            first_stl = st.session_state.parts_config[0]["stl_path"]
            if os.path.exists(first_stl):
                stl_bounds = get_stl_bounds_cm(first_stl)

        fig_3d = render_3d_scanner_geometry(
            sod_val=float(sod_val),
            sdd_offset=float(sdd_offset),
            sdd_val=float(sdd_val),
            det_rows=int(det_rows),
            det_cols=int(det_cols),
            pixel_size=float(pixel_size),
            stl_bounds=stl_bounds,
            source_type=source_type
        )
        st.plotly_chart(fig_3d, use_container_width=True)

        # FOV & Geometry metrics calculation
        dx, dy, dz = stl_bounds
        magnification = (sdd_val / sod_val) if sod_val > 0 else 1.0
        r_max = math.sqrt(dx**2 + dy**2)
        projected_footprint_cm = 2.0 * r_max * magnification
        projected_footprint_mm = projected_footprint_cm * 10.0
        detector_physical_width_cm = det_cols * pixel_size / 10.0
        detector_physical_width_mm = det_cols * pixel_size
        detector_physical_height_mm = det_rows * pixel_size
        fov_coverage_pct = (projected_footprint_cm / detector_physical_width_cm * 100.0) if detector_physical_width_cm > 0 else 0.0
        object_fits = projected_footprint_cm <= detector_physical_width_cm

        st.markdown('<div class="webct-card">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Magnification", f"{magnification:.2f}×")
        with col2:
            st.metric("Detector Physical Size", f"{detector_physical_width_mm:.1f} × {detector_physical_height_mm:.1f} mm")
        with col3:
            st.metric("Object Projected Footprint", f"{projected_footprint_mm:.1f} mm")
        with col4:
            st.metric("FOV Coverage", f"{fov_coverage_pct:.1f}%")

        if object_fits:
            st.success("Object fits within detector FOV")
        else:
            st.warning("Object may be truncated — increase detector size or pixel size, or move object closer to detector")
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"3D Visualizer Error: {e}")

