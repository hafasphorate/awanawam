import streamlit as st
import osmnx as ox
import networkx as nx
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import io

st.set_page_config(page_title="Urban Space Syntax", layout="wide")
st.title("Singapore Urban Space Syntax Analysis")

# 1. User Input Controls
col1, col2 = st.columns(2)
with col1:
    search_query = st.text_input("Location in Singapore", "Tiong Bahru, Singapore")
with col2:
    radius = st.slider("Radius (meters)", min_value=300, max_value=2000, value=600, step=100)

network_type = st.selectbox("Network Type", ["walk", "drive", "all"])

# Ensure State Initialization
if "has_run" not in st.session_state:
    st.session_state.has_run = False
if "axial_map" not in st.session_state:
    st.session_state.axial_map = None
if "dendrogram_fig" not in st.session_state:
    st.session_state.dendrogram_fig = None
if "export_fig" not in st.session_state:
    st.session_state.export_fig = None

# Trigger Run & Update Session State Directly
if st.button("Run Axial Analysis"):
    with st.spinner("Processing network, dendrogram, and high-contrast intersections..."):
        try:
            # Geocode location
            gdf_place = ox.geocode_to_gdf(search_query)
            center_lat = gdf_place.geometry.iloc[0].centroid.y
            center_lon = gdf_place.geometry.iloc[0].centroid.x
            
            # Fetch graph
            G = ox.graph_from_point((center_lat, center_lon), dist=radius, network_type=network_type)
            G_undirected = ox.convert.to_undirected(G)
            
            # Compute Centrality
            edge_centrality = nx.edge_betweenness_centrality(G_undirected)
            nx.set_edge_attributes(G_undirected, edge_centrality, "betweenness")
            
            gdf_nodes, gdf_edges = ox.convert.graph_to_gdfs(G_undirected)
            
            # Normalize edge values
            min_val = gdf_edges['betweenness'].min()
            max_val = gdf_edges['betweenness'].max()
            cmap = plt.get_cmap('turbo')
            
            def get_color_hex(val):
                norm = (val - min_val) / (max_val - min_val + 1e-6)
                return mcolors.to_hex(cmap(norm))
            
            gdf_edges['hex_color'] = gdf_edges['betweenness'].apply(get_color_hex)
            
            # Calculate intersection variance (Centrality contrast)
            node_variance = {}
            for node in G_undirected.nodes():
                incident_edges = G_undirected.edges(node, data=True)
                if len(incident_edges) > 1:
                    scores = [d.get('betweenness', 0) for _, _, d in incident_edges]
                    node_variance[node] = np.std(scores)
                else:
                    node_variance[node] = 0.0

            threshold = np.percentile(list(node_variance.values()), 90)
            
            # Build Folium Map
            m = folium.Map(location=[center_lat, center_lon], zoom_start=16, tiles="CartoDB dark_matter")
            
            # Render Edges
            for _, row in gdf_edges.iterrows():
                if row.geometry.geom_type == 'LineString':
                    coords = [[p[1], p[0]] for p in row.geometry.coords]
                    folium.PolyLine(
                        locations=coords,
                        color=row['hex_color'],
                        weight=3.5,
                        opacity=0.85,
                        tooltip=f"Centrality: {row['betweenness']:.4f}"
                    ).add_to(m)
            
            # Render Intersections (Dots)
            for node_id, row in gdf_nodes.iterrows():
                lat, lon = row.geometry.y, row.geometry.x
                is_high_contrast = node_variance.get(node_id, 0) >= threshold and node_variance.get(node_id, 0) > 0
                
                if is_high_contrast:
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=6,
                        color='#FF0055',
                        fill=True,
                        fill_color='#FF0055',
                        fill_opacity=0.9,
                        popup=f"High Contrast Shift Intersection"
                    ).add_to(m)
                else:
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=2.5,
                        color='#FFFFFF',
                        fill=True,
                        fill_color='#CCCCCC',
                        fill_opacity=0.6,
                        tooltip=f"Node ID: {node_id}"
                    ).add_to(m)
            
            # Map Legend Overlay
            legend_html = '''
            <div style="position: fixed; bottom: 30px; left: 30px; width: 220px; height: 125px; 
                        background-color: rgba(0,0,0,0.85); z-index:9999; font-size:12px; color: white;
                        padding: 10px; border-radius: 5px; font-family: sans-serif; border: 1px solid #555;">
                <b>Space Syntax Legend</b><br>
                <i style="background: red; width: 12px; height: 12px; display: inline-block;"></i> High Integration / Choice<br>
                <i style="background: blue; width: 12px; height: 12px; display: inline-block;"></i> Low Integration / Choice<br>
                <i style="background: #FF0055; width: 10px; height: 10px; border-radius: 50%; display: inline-block;"></i> High Contrast Intersection<br>
                <i style="background: #FFFFFF; width: 6px; height: 6px; border-radius: 50%; display: inline-block;"></i> Standard Intersection
            </div>
            '''
            m.get_root().html.add_child(folium.Element(legend_html))
            
            # Generate Dendrogram Plot
            feature_matrix = np.column_stack([gdf_edges['length'].values, gdf_edges['betweenness'].values])
            Z = linkage(feature_matrix, method='ward')
            
            fig_dend, ax_dend = plt.subplots(figsize=(10, 4))
            dendrogram(Z, ax=ax_dend, truncate_mode='lastp', p=30, leaf_rotation=90., leaf_font_size=8.)
            ax_dend.set_title("Pathway Hierarchical Cluster Dendrogram")
            ax_dend.set_ylabel("Height / Distance")
            plt.tight_layout()
            
            # Generate Export Figure
            fig_export, ax_export = plt.subplots(figsize=(8, 8))
            gdf_edges.plot(ax=ax_export, column='betweenness', cmap='turbo', linewidth=2)
            gdf_nodes.plot(ax=ax_export, color='white', markersize=4, alpha=0.7)
            ax_export.set_facecolor('black')
            fig_export.patch.set_facecolor('black')
            ax_export.axis('off')
            plt.tight_layout()

            # Assign to Session State
            st.session_state.axial_map = m
            st.session_state.dendrogram_fig = fig_dend
            st.session_state.export_fig = fig_export
            st.session_state.has_run = True

            # Force immediate UI rerun to flush out old cached states
            st.rerun()

        except Exception as e:
            st.error(f"Error generating analysis: {e}")

# -----------------------------------------------------------------------------
# Unconditional Render Section (Runs outside button scope based on session state)
# -----------------------------------------------------------------------------
if st.session_state.has_run and st.session_state.axial_map is not None:
    
    st.subheader("1. Interactive Space Syntax Map")
    st_folium(
        st.session_state.axial_map, 
        width=1000, 
        height=500, 
        key="space_syntax_map_v2",
        returned_objects=[]
    )
    
    with st.expander("Feature Explanations"):
        st.markdown("""
        * **Colored Pathways:** High integration (red) vs local pathways (blue).
        * **White Dots:** Intersections connecting pathways.
        * **Magenta Dots:** Nodes with steep shifts in centrality between connecting edges.
        """)

    st.subheader("2. Pathway Dendrogram")
    st.pyplot(st.session_state.dendrogram_fig)

    # Sidebar Exports
    st.sidebar.markdown("---")
    st.sidebar.subheader("Export Results")
    
    svg_io = io.BytesIO()
    st.session_state.export_fig.savefig(svg_io, format='svg', bbox_inches='tight', facecolor='black')
    st.sidebar.download_button(
        label="Download Vector (SVG)",
        data=svg_io.getvalue(),
        file_name="space_syntax.svg",
        mime="image/svg+xml"
    )
    
    png_io = io.BytesIO()
    st.session_state.export_fig.savefig(png_io, format='png', dpi=300, bbox_inches='tight', facecolor='black')
    st.sidebar.download_button(
        label="Download Image (PNG)",
        data=png_io.getvalue(),
        file_name="space_syntax.png",
        mime="image/png"
    )