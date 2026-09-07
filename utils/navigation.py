import base64
from pathlib import Path

import streamlit as st


LOGO_PATH = Path(__file__).resolve().parent.parent / "awanawam_logo.png"


def render_home_button(location="sidebar", width=180):
    logo_data = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    markup = f'''
        <a href="/" target="_self" aria-label="Home" style="display:block; text-align:center;">
            <img src="data:image/png;base64,{logo_data}" alt="Awanawam home" width="{width}" style="max-width:100%; height:auto;">
        </a>
    '''

    container = st.sidebar if location == "sidebar" else st
    container.markdown(markup, unsafe_allow_html=True)
