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
import json

st.set_page_config(page_title="Urban Space Syntax Analysis", layout="wide")

# -----------------------------------------------------------------------------
# 1. Page Title & Definitions (Request 4)
# -----------------------------------------------------------------------------
st.title("Singapore Urban Space Syntax Analysis")

st.markdown("""
**Metric Definition:** **Betweenness Centrality** (used here as an axial proxy for *Space Syntax Choice / Integration*) measures the fraction of all shortest topological paths passing through a specific street segment within the network:

$$C_B(e) = \sum_{s \neq t \in V} \frac{\sigma_{st}(e)}{\sigma_{st}}$$

* **Units:** Dimensionless score bounded between **$0.0$** (completely isolated / low choice) and **$1.0$** (maximum spatial movement / high choice). Higher values highlight primary movement trunks in the urban layout.
""")
st.write("---")

# -----------------------------------------------------------------------------
# 2. User Input Controls & Toggles (Request 2)
# -----------------------------------------------------------------------------
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    search_query = st.text_input("Location in Singapore", "Tiong Bahru, Singapore")
with col2:
    radius = st.slider("Analysis Radius (meters)", min_value=300, max_value=2000, value=700, step=100)
with col3:
    network_type = st.selectbox("Network Type", ["walk", "drive", "all"])

# Sidebar display options (Request 2)
st.sidebar.header("Map Display Options")
show_standard_nodes = st.sidebar.checkbox("Show Standard Intersections (White)", value=True)
show_contrast_nodes = st.sidebar.checkbox("Show High-Contrast Intersections (Red)", value=True)

# Session State Initialization
if "has_run" not in st.session_state:
    st.session_state.has_run = False
if "axial_map" not in st.session_state:
    st.session_state.axial_map = None
if "dendrogram_fig" not in st.session_state:
    st.session_state.dendrogram_fig = None
if "export_fig" not in st.session_state:
    st.session_state.export_fig = None
if "gdf_edges_export" not in st.session_state:
    st.session_state.gdf_edges_export = None

# -----------------------------------------------------------------------------
# 3. Main Analytical Workflow
# -----------------------------------------------------------------------------
if st.button("Run Axial Analysis"):
    with st.spinner("Processing network, centralities, dendrogram, and exports..."):
        try:
            # Geocode Location
            gdf_place = ox.geocode_to_gdf(search_query)
            center_lat = gdf_place.geometry.iloc[0].centroid.y
            center_lon = gdf_place.geometry.iloc[0].centroid.x
            
            # Fetch Street Graph
            G = ox.graph_from_point((center_lat, center_lon), dist=radius, network_type=network_type)
            G_undirected = ox.convert.to_undirected(G)
            
            # Calculate Betweenness Centrality
            edge_centrality = nx.edge_betweenness_centrality(G_undirected)
            nx.set_edge_attributes(G_undirected, edge_centrality, "betweenness")
            
            gdf_nodes, gdf_edges = ox.convert.graph_to_gdfs(G_undirected)
            
            # Clean up street names for exports/dendrogram
            def clean_name(val):
                if isinstance(val, list):
                    return ", ".join(str(v) for v in val if v)
                return str(val) if val and str(val) != 'nan' else 'Unnamed Segment'

            gdf_edges['street_name'] = gdf_edges['name'].apply(clean_name)
            
            # Normalize Centrality
            min_val = gdf_edges['betweenness'].min()
            max_val = gdf_edges['betweenness'].max()
            cmap = plt.get_cmap('turbo')
            
            def get_color_hex(val):
                norm = (val - min_val) / (max_val - min_val + 1e-6)
                return mcolors.to_hex(cmap(norm))
            
            gdf_edges['hex_color'] = gdf_edges['betweenness'].apply(get_color_hex)
            
            # -----------------------------------------------------------------
            # Refined High-Contrast Intersection Calculation (Request 5)
            # -----------------------------------------------------------------
            # High contrast = Node where high centrality (>=75th percentile) directly intersects low centrality (<=25th percentile)
            q_high = gdf_edges['betweenness'].quantile(0.75)
            q_low = gdf_edges['betweenness'].quantile(0.25)
            
            high_contrast_nodes = set()
            for node in G_undirected.nodes():
                incident_edges = G_undirected.edges(node, data=True)
                if len(incident_edges) > 1:
                    scores = [d.get('betweenness', 0) for _, _, d in incident_edges]
                    has_high = any(s >= q_high for s in scores)
                    has_low = any(s <= q_low for s in scores)
                    if has_high and has_low:
                        high_contrast_nodes.add(node)

            # -----------------------------------------------------------------
            # Build Interactive Folium Map (Requests 2 & 3)
            # -----------------------------------------------------------------
            m = folium.Map(location=[center_lat, center_lon], zoom_start=16, tiles="CartoDB dark_matter")
            
            # Draw Segments
            for _, row in gdf_edges.iterrows():
                if row.geometry.geom_type == 'LineString':
                    coords = [[p[1], p[0]] for p in row.geometry.coords]
                    folium.PolyLine(
                        locations=coords,
                        color=row['hex_color'],
                        weight=3.5,
                        opacity=0.85,
                        tooltip=f"Street: {row['street_name']} | Centrality: {row['betweenness']:.4f}"
                    ).add_to(m)
            
            # Draw Intersections with 60% Transparency / 0.4 Opacity (Request 2)
            for node_id, row in gdf_nodes.iterrows():
                lat, lon = row.geometry.y, row.geometry.x
                is_contrast = node_id in high_contrast_nodes
                
                if is_contrast and show_contrast_nodes:
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=6,
                        color='#FF0055',
                        fill=True,
                        fill_color='#FF0055',
                        fill_opacity=0.4,  # 60% opacity / 40% fill
                        opacity=0.4,
                        popup=f"High-Contrast Intersection<br>High/Low Centrality Boundary"
                    ).add_to(m)
                elif not is_contrast and show_standard_nodes:
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=2.5,
                        color='#FFFFFF',
                        fill=True,
                        fill_color='#FFFFFF',
                        fill_opacity=0.4,  # 60% opacity
                        opacity=0.4,
                        tooltip=f"Intersection ID: {node_id}"
                    ).add_to(m)
            
            # Gradient Legend Overlay (Request 3)
            legend_html = '''
            <div style="position: fixed; bottom: 30px; left: 30px; width: 250px; 
                        background-color: rgba(0,0,0,0.85); z-index:9999; font-size:12px; color: white;
                        padding: 12px; border-radius: 6px; font-family: sans-serif; border: 1px solid #444;">
                <b>Space Syntax Integration</b><br>
                <div style="display: flex; justify-content: space-between; margin-top: 5px; font-size: 10px;">
                    <span>Low (0.0)</span>
                    <span>High (1.0)</span>
                </div>
                <div style="height: 12px; width: 100%; background: linear-gradient(to right, #300060, #0000FF, #00FFFF, #00FF00, #FFFF00, #FF0000); border-radius: 2px;"></div>
                <hr style="border: 0.5px solid #444; margin: 8px 0;">
                <i style="background: rgba(255,0,85,0.4); width: 10px; height: 10px; border-radius: 50%; display: inline-block; border: 1px solid #FF0055;"></i> High-Contrast Intersection<br>
                <i style="background: rgba(255,255,255,0.4); width: 8px; height: 8px; border-radius: 50%; display: inline-block;"></i> Standard Intersection
            </div>
            '''
            m.get_root().html.add_child(folium.Element(legend_html))
            
            # -----------------------------------------------------------------
            # Enhanced Dendrogram with Road Labels (Request 7)
            # -----------------------------------------------------------------
            # Group street segments and sample prominent road names
            feature_matrix = np.column_stack([
                gdf_edges['length'].values,
                gdf_edges['betweenness'].values
            ])
            
            Z = linkage(feature_matrix, method='ward')
            
            # Extract names for dendrogram leaves
            leaf_names = gdf_edges['street_name'].tolist()
            
            fig_dend, ax_dend = plt.subplots(figsize=(12, 5), dpi=200)
            dend_res = dendrogram(
                Z, 
                ax=ax_dend,
                truncate_mode='lastp',
                p=25,
                leaf_rotation=90.,
                leaf_font_size=9.,
                labels=leaf_names[:len(Z)+1] if len(leaf_names) > len(Z) else None,
                show_contracted=True
            )
            
            ax_dend.set_title("Hierarchical Cluster Dendrogram of Street Pathways", fontsize=12, fontweight='bold')
            ax_dend.set_ylabel("Ward Linkage Distance (Topology + Geometry Deviation)", fontsize=10)
            ax_dend.set_xlabel("Street Clusters (Major Street Segments Labeled)", fontsize=10)
            plt.tight_layout()

            # -----------------------------------------------------------------
            # Prepare Exportable Static Figure (Request 1)
            # -----------------------------------------------------------------
            fig_export, ax_export = plt.subplots(figsize=(8, 8), dpi=300)
            gdf_edges.plot(ax=ax_export, column='betweenness', cmap='turbo', linewidth=2)
            if show_standard_nodes:
                gdf_nodes[~gdf_nodes.index.isin(high_contrast_nodes)].plot(ax=ax_export, color='white', markersize=3, alpha=0.4)
            if show_contrast_nodes:
                gdf_nodes[gdf_nodes.index.isin(high_contrast_nodes)].plot(ax=ax_export, color='#FF0055', markersize=25, alpha=0.4)
            
            ax_export.set_facecolor('#111111')
            fig_export.patch.set_facecolor('#111111')
            ax_export.axis('off')
            plt.tight_layout()

            # Assign to Session State
            st.session_state.axial_map = m
            st.session_state.dendrogram_fig = fig_dend
            st.session_state.export_fig = fig_export
            st.session_state.gdf_edges_export = gdf_edges[['street_name', 'betweenness', 'length', 'geometry']]
            st.session_state.has_run = True

            st.rerun()

        except Exception as e:
            st.error(f"Error generating analysis: {e}")

# -----------------------------------------------------------------------------
# 4. Results Section (Requests 1, 6, 7)
# -----------------------------------------------------------------------------
if st.session_state.has_run and st.session_state.axial_map is not None:
    
    st.subheader("1. Interactive Axial & Segment Syntax Map")
    st_folium(
        st.session_state.axial_map, 
        width=1000, 
        height=520, 
        key="space_syntax_map_v3",
        returned_objects=[]
    )
    
    # Detailed Feature Explanations (Request 6)
    st.markdown("### Feature Explanations & Urban Implications")
    st.markdown("""
    * **Axial Segments (Rainbow Scale):** Calculated by computing topological shortest paths across the street network graph. 
      * *Implication:* **Red/Orange segments** represent primary movement arteries that accumulate high pedestrian or vehicular throughput. **Blue/Purple segments** indicate segregated, quiet streets suitable for residential zones or low-traffic public spaces.
    * **High-Contrast Intersections (Magenta Dots):** Calculated by identifying nodes where a high-centrality street ($\ge 75\text{th percentile}$) directly connects to a low-centrality street ($\le 25\text{th percentile}$).
      * *Implication:* These represent critical urban decision points or transitional boundaries (e.g., exiting a major arterial road directly into a quiet alleyway or pedestrianized precinct). They are key zones for urban design interventions, traffic calming, or safety monitoring.
    * **Standard Intersections (White Dots):** Calculated as topological graph vertices where street lines join or split.
      * *Implication:* Represents physical connectivity density. Highly dense node clusters highlight fine-grained street grids with high walkability potential.
    """)

    st.write("---")
    st.subheader("2. Pathway Hierarchical Dendrogram")
    st.pyplot(st.session_state.dendrogram_fig)
    
    # Dendrogram Explanations (Request 7)
    st.markdown("""
    **Dendrogram Axis & Structural Explanation:**
    * **Vertical Axis (Ward Linkage Distance):** Represents the mathematical variance between street clusters based on segment length and topological centrality. Higher branch points indicate major structural divergences in the urban grid layout.
    * **Horizontal Axis (Street Clusters):** Groups street segments with similar spatial behaviors. Major road names are labeled along the base branches to show which physical streets form unified movement corridors.
    """)

    # -------------------------------------------------------------------------
    # Sidebar QGIS & Image Exports (Request 1)
    # -------------------------------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("Export Results for QGIS / CAD")
    
    # GeoJSON Export (Native GIS Format)
    geojson_str = st.session_state.gdf_edges_export.to_json()
    st.sidebar.download_button(
        label="Download GIS Vector (.geojson)",
        data=geojson_str,
        file_name="space_syntax_singapore.geojson",
        mime="application/geo+json",
        help="Import directly into QGIS, ArcGIS, or Rhino/Grasshopper."
    )
    
    # PNG Image Export
    png_io = io.BytesIO()
    st.session_state.export_fig.savefig(png_io, format='png', dpi=300, bbox_inches='tight', facecolor='#111111')
    st.sidebar.download_button(
        label="Download High-Res Map (.png)",
        data=png_io.getvalue(),
        file_name="space_syntax_map.png",
        mime="image/png"
    )