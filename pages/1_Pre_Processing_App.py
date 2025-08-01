import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import io

# Suppress matplotlib warnings
import warnings
warnings.filterwarnings('ignore')

# Mario Theme Colors
MARIO_COLORS = {
    'red': '#E60012',           # Mario's hat red
    'blue': '#0066CC',          # Mario's overalls blue
    'yellow': '#FFD700',        # Coins/stars yellow
    'green': '#00A652',         # Luigi green
    'brown': '#8B4513',         # Blocks brown
    'orange': '#FF8C00',        # Fire flower orange
    'purple': '#8A2BE2',        # Poison mushroom purple
    'light_blue': '#87CEEB',    # Sky blue
    'dark_red': '#8B0000',      # Dark red
    'white': '#FFFFFF',         # Cloud white
    'black': '#000000'          # Outline black
}

# Set page config with simpler options
st.set_page_config(
    page_title="Mario Text Analysis",
    page_icon="🍄",
    layout="wide"
)

# Custom CSS for Mario theme
st.markdown("""
<style>
    /* Mario-themed background and main styling */
    .main .block-container {
        background: linear-gradient(135deg, #87CEEB 0%, #B0E0E6 50%, #87CEEB 100%);
        padding: 2rem 1rem;
        border-radius: 15px;
        border: 4px solid #8B4513;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
    }
    
    /* Main app background */
    .stApp {
        background: linear-gradient(45deg, #00A652 0%, #32CD32 25%, #00A652 50%, #228B22 75%, #00A652 100%);
        background-size: 400% 400%;
        animation: gradientShift 8s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #FFD700 0%, #FFA500 100%);
        border: 3px solid #8B4513;
        border-radius: 10px;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        color: #E60012;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.5);
        background: linear-gradient(45deg, #FFD700, #FFA500);
        padding: 20px;
        border-radius: 15px;
        border: 4px solid #8B4513;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    .sub-header {
        text-align: center;
        color: #0066CC;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        background: rgba(255, 255, 255, 0.8);
        padding: 10px;
        border-radius: 10px;
        border: 2px solid #0066CC;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    /* Success and info boxes */
    .success-box {
        background: linear-gradient(45deg, #00A652, #32CD32);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border: 3px solid #228B22;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    .info-box {
        background: linear-gradient(45deg, #FFD700, #FFA500);
        color: #8B4513;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border: 3px solid #8B4513;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        font-weight: bold;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #E60012, #FF4500);
        color: white;
        border: 3px solid #8B0000;
        border-radius: 10px;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #FF4500, #E60012);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.4);
    }
    
    /* Dataframe styling */
    .dataframe {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        border: 2px solid #8B4513;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Metric styling */
    .metric-container {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        border: 2px solid #8B4513;
        padding: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* File uploader styling */
    .uploadedFile {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        border: 2px solid #8B4513;
    }
    
    /* Sidebar elements */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 5px;
        border: 2px solid #8B4513;
    }
    
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 5px;
        border: 2px solid #8B4513;
    }
    
    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 5px;
        border: 2px solid #8B4513;
    }
    
    /* Add Mario coin animation */
    @keyframes coinFlip {
        0% { transform: rotateY(0deg); }
        100% { transform: rotateY(360deg); }
    }
    
    .coin {
        display: inline-block;
        animation: coinFlip 2s linear infinite;
    }
</style>
""", unsafe_allow_html=True)

def mario_style_plot():
    """Set Mario-themed plot style"""
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.weight': 'bold',
        'axes.facecolor': MARIO_COLORS['light_blue'],
        'figure.facecolor': MARIO_COLORS['white'],
        'axes.edgecolor': MARIO_COLORS['black'],
        'axes.linewidth': 3,
        'grid.color': MARIO_COLORS['white'],
        'grid.alpha': 0.7,
        'font.size': 10
    })
