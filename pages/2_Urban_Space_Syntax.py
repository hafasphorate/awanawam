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

# -----------------------------------------------------------------------------
# 1. User Input Controls
# -----------------------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    search_query = st.text_input("Location in Singapore", "Tiong Bahru, Singapore")
with col2:
    radius = st.slider("Radius (meters)", min_value=300, max_value=2000, value=600, step=100)

network_type = st.selectbox("Network Type", ["walk", "drive", "all"])

# Initialize session state variables
if "axial_map" not in st.session_state:
    st.session_state.axial_map = None
if "dendrogram_fig" not in st.session_state:
    st.session_state.dendrogram_fig = None
if "export_fig" not in st.session_state:
    st.session_state.export_fig = None

# -----------------------------------------------------------------------------
# 2. Main Processing Workflow
# -----------------------------------------------------------------------------
if st.button("Run Axial Analysis"):
    with st.spinner("Fetching network, calculating centralities, and building hierarchical clusters..."):
        try:
            # Geocode location
            gdf_place = ox.geocode_to_gdf(search_query)
            center_lat = gdf_place.geometry.iloc[0].centroid.y
            center_lon = gdf_place.geometry.iloc[0].centroid.x
            
            # Fetch graph
            G = ox.graph_from_point((center_lat, center_lon), dist=radius, network_type=network_type)
            G_undirected = ox.convert.to_undirected(G)
            
            # Compute Betweenness Centrality (Space Syntax Integration/Choice Proxy)
            edge_centrality = nx.edge_betweenness_centrality(G_undirected)
            nx.set_edge_attributes(G_undirected, edge_centrality, "betweenness")
            
            gdf_nodes, gdf_edges = ox.convert.graph_to_gdfs(G_undirected)
            
            # Normalize edge values for colormap
            min_val = gdf_edges['betweenness'].min()
            max_val = gdf_edges['betweenness'].max()
            cmap = plt.get_cmap('turbo')
            
            def get_color_hex(val):
                norm = (val - min_val) / (max_val - min_val + 1e-6)
                return mcolors.to_hex(cmap(norm))
            
            gdf_edges['hex_color'] = gdf_edges['betweenness'].apply(get_color_hex)
            
            # -----------------------------------------------------------------
            # Highlight High Centrality Contrast Intersections (Feature 5)
            # -----------------------------------------------------------------
            # Detect intersections where connected edges have high standard deviation in centrality
            node_variance = {}
            for node in G_undirected.nodes():
                incident_edges = G_undirected.edges(node, data=True)
                if len(incident_edges) > 1:
                    scores = [d.get('betweenness', 0) for _, _, d in incident_edges]
                    node_variance[node] = np.std(scores)
                else:
                    node_variance[node] = 0.0

            threshold = np.percentile(list(node_variance.values()), 90) # Top 10% highest contrast
            
            # -----------------------------------------------------------------
            # Build Interactive Folium Map (Features 1, 4, 5)
            # -----------------------------------------------------------------
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
                        tooltip=f"Edge Centrality: {row['betweenness']:.4f}"
                    ).add_to(m)
            
            # Render Nodes / Intersections (Feature 1 & 5)
            for node_id, row in gdf_nodes.iterrows():
                lat, lon = row.geometry.y, row.geometry.x
                is_high_contrast = node_variance.get(node_id, 0) >= threshold and node_variance.get(node_id, 0) > 0
                
                # High contrast intersections get highlighted with red circles
                if is_high_contrast:
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=6,
                        color='#FF0055',
                        fill=True,
                        fill_color='#FF0055',
                        fill_opacity=0.9,
                        popup=f"High Centrality Shift Intersection<br>Std Dev: {node_variance[node_id]:.4f}"
                    ).add_to(m)
                else:
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=2.5,
                        color='#FFFFFF',
                        fill=True,
                        fill_color='#CCCCCC',
                        fill_opacity=0.6,
                        tooltip=f"Intersection ID: {node_id}"
                    ).add_to(m)
            
            # Add Legend Overlay (Feature 4)
            legend_html = '''
            <div style="position: fixed; bottom: 30px; left: 30px; width: 220px; height: 130px; 
                        background-color: rgba(0,0,0,0.7); z-index:9999; font-size:12px; color: white;
                        padding: 10px; border-radius: 5px; font-family: sans-serif;">
                <b>Space Syntax Legend</b><br>
                <i style="background: red; width: 12px; height: 12px; display: inline-block;"></i> High Integration / Choice<br>
                <i style="background: blue; width: 12px; height: 12px; display: inline-block;"></i> Low Integration / Choice<br>
                <i style="background: #FF0055; width: 10px; height: 10px; border-radius: 50%; display: inline-block;"></i> High Contrast Intersection<br>
                <i style="background: #FFFFFF; width: 6px; height: 6px; border-radius: 50%; display: inline-block;"></i> Standard Intersection
            </div>
            '''
            m.get_root().html.add_child(folium.Element(legend_html))
            
            st.session_state.axial_map = m

            # -----------------------------------------------------------------
            # Generate Dendrogram (Feature 2)
            # -----------------------------------------------------------------
            # Extract feature vectors for edges (length & centrality) for hierarchical clustering
            feature_matrix = np.column_stack([
                gdf_edges['length'].values,
                gdf_edges['betweenness'].values
            ])
            
            # Perform Linkage
            Z = linkage(feature_matrix, method='ward')
            
            fig_dend, ax_dend = plt.subplots(figsize=(12, 4))
            dendrogram(
                Z, 
                ax=ax_dend,
                truncate_mode='lastp',
                p=30,
                leaf_rotation=90.,
                leaf_font_size=8.,
                show_contracted=True
            )
            ax_dend.set_title("Pathway Hierarchical Cluster Dendrogram")
            ax_dend.set_ylabel("Distance Threshold / Height")
            ax_dend.set_xlabel("Street Segment Clusters")
            plt.tight_layout()
            
            st.session_state.dendrogram_fig = fig_dend

            # -----------------------------------------------------------------
            # Exportable Static Map Figure (Feature 3)
            # -----------------------------------------------------------------
            fig_export, ax_export = plt.subplots(figsize=(10, 10))
            gdf_edges.plot(ax=ax_export, column='betweenness', cmap='turbo', linewidth=2)
            gdf_nodes.plot(ax=ax_export, color='white', markersize=5, alpha=0.7)
            
            # Highlight high variance nodes on export map
            high_var_nodes = gdf_nodes[gdf_nodes.index.isin([k for k, v in node_variance.items() if v >= threshold])]
            high_var_nodes.plot(ax=ax_export, color='#FF0055', markersize=30)
            
            ax_export.set_facecolor('black')
            fig_export.patch.set_facecolor('black')
            ax_export.axis('off')
            ax_export.set_title(f"Axial Centrality - {search_query}", color='white', fontsize=14)
            plt.tight_layout()
            
            st.session_state.export_fig = fig_export

        except Exception as e:
            st.error(f"Error generating analysis: {e}")

# -----------------------------------------------------------------------------
# 3. Output Displays & Layout
# -----------------------------------------------------------------------------
if st.session_state.axial_map is not None:
    st.subheader("1. Interactive Space Syntax Map")
    st_folium(
        st.session_state.axial_map, 
        width=1000, 
        height=550, 
        key="space_syntax_map",
        returned_objects=[]
    )
    
    # Feature Explanations (Feature 4)
    with st.expander("Understanding the Features & Space Syntax Metrics"):
        st.markdown("""
        * **Line Segments (Axial/Segment Network):** Colored using the classic Space Syntax spectrum. **Red lines** denote high Choice/Betweenness (frequently traversed axial pathways), while **Blue lines** indicate segregated or local pathways.
        * **White Dots (Intersections):** Represent street nodes where pathways cross.
        * **Magenta / Pink Dots (High Contrast Intersections):** Represent urban decision points where connecting streets undergo significant shifts in centrality (e.g., exiting a major arterial road directly into a quiet alley).
        """)

    st.subheader("2. Pathway Hierarchical Dendrogram")
    st.pyplot(st.session_state.dendrogram_fig)

# -----------------------------------------------------------------------------
# 4. Export Vector / Image Options Sidebar (Feature 3)
# -----------------------------------------------------------------------------
if st.session_state.export_fig is not None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Export Results")
    
    # SVG (Vector) Export
    svg_io = io.BytesIO()
    st.session_state.export_fig.savefig(svg_io, format='svg', bbox_inches='tight', facecolor='black')
    st.sidebar.download_button(
        label="Download Map as Vector (SVG)",
        data=svg_io.getvalue(),
        file_name="space_syntax_analysis.svg",
        mime="image/svg+xml"
    )
    
    # PNG Export
    png_io = io.BytesIO()
    st.session_state.export_fig.savefig(png_io, format='png', dpi=300, bbox_inches='tight', facecolor='black')
    st.sidebar.download_button(
        label="Download Map as Image (PNG)",
        data=png_io.getvalue(),
        file_name="space_syntax_analysis.png",
        mime="image/png"
    )