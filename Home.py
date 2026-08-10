import streamlit as st

# Must be the very first Streamlit command called on the page
st.set_page_config(
    page_title="Spatial & Crowd Dynamics Toolkit",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Architecture & Crowd Dynamics Framework")
st.markdown("""
Welcome! This toolkit provides spatial analysis and computer vision tools to evaluate how spatial configuration impacts crowd movement.

### Available Modules:
1. **Visibility Graph Analysis (VGA):** Upload DXF floorplans to calculate visual integration, entropy, and isovist metrics.
2. **Urban Space Syntax:** idk what to put here yet let's KIV.
3. **Video Homography & Tracking:** Track human movement from site videos and project coordinates onto floorplans.
4. **Spatial vs. Crowd Data:** Upload analysis JSON files to run Pearson/Spearman spatial correlation tests.
5. **Aggregated Insights:** Explore crowd dynamic trends across multiple architectural case studies.
""")