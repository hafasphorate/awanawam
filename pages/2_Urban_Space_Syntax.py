import streamlit as st
import osmnx as ox
import networkx as nx
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import io

st.set_page_config(page_title="Urban Space Syntax Analysis", layout="wide")

# -----------------------------------------------------------------------------
# 1. Page Title & Corrected Definitions (Fixes Issue 1 & 3)
# -----------------------------------------------------------------------------
st.title("Singapore Urban Space Syntax Analysis")

st.markdown("""
**Metric Definition:** **Betweenness Centrality** (used here as an axial proxy for *Space Syntax Choice / Integration*) measures the fraction of all shortest topological paths passing through a specific street segment within the network:
""")

# Clean LaTeX rendering
st.latex(r"C_B(e) = \sum_{s \neq t \in V} \frac{\sigma_{st}(e)}{\sigma_{st}}")

st.markdown("""
* **Units:** Bounded dimensionless score from **0.0** (completely isolated / low choice) to **1.0** (maximum spatial movement / high choice). Higher values highlight primary movement trunks in the urban layout.
""")
st.write("---")

# -----------------------------------------------------------------------------
# 2. Controls & Instant Toggle Configuration (Fixes Issue 2)
# -----------------------------------------------------------------------------
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    search_query = st.text_input("Location in Singapore", "Tiong Bahru, Singapore")
with col2:
    radius = st.slider("Analysis Radius (meters)", min_value=300, max_value=2000, value=600, step=100)
with col3:
    network_type = st.selectbox("Network Type", ["walk", "drive", "all"])

# Sidebar Toggles (Instant rerender without triggering heavy OSmnx pipeline)
st.sidebar.header("Map Display Options")
show_standard_nodes = st.sidebar.checkbox("Show Standard Intersections (White)", value=True)
show_contrast_nodes = st.sidebar.checkbox("Show High-Contrast Intersections (Red)", value=True)

# Session State Initialization
if "graph_data" not in st.session_state:
    st.session_state.graph_data = None

# -----------------------------------------------------------------------------
# 3. Main Data Processing Workflow
# -----------------------------------------------------------------------------
if st.button("Run Axial Analysis"):
    with st.spinner("Processing urban network and calculating spatial depth..."):
        try:
            # Geocode location
            gdf_place = ox.geocode_to_gdf(search_query)
            center_lat = gdf_place.geometry.iloc[0].centroid.y
            center_lon = gdf_place.geometry.iloc[0].centroid.x
            
            # Fetch Network
            G = ox.graph_from_point((center_lat, center_lon), dist=radius, network_type=network_type)
            G_undirected = ox.convert.to_undirected(G)
            
            # Compute Centrality
            edge_centrality = nx.edge_betweenness_centrality(G_undirected)
            nx.set_edge_attributes(G_undirected, edge_centrality, "betweenness")
            
            gdf_nodes, gdf_edges = ox.convert.graph_to_gdfs(G_undirected)
            
            def clean_name(val):
                if isinstance(val, list):
                    return ", ".join(str(v) for v in val if v)
                return str(val) if val and str(val) != 'nan' else 'Unnamed Segment'

            gdf_edges['street_name'] = gdf_edges['name'].apply(clean_name)
            
            # Color Mapping
            min_val = gdf_edges['betweenness'].min()
            max_val = gdf_edges['betweenness'].max()
            cmap = plt.get_cmap('turbo')
            
            def get_color_hex(val):
                norm = (val - min_val) / (max_val - min_val + 1e-6)
                return mcolors.to_hex(cmap(norm))
            
            gdf_edges['hex_color'] = gdf_edges['betweenness'].apply(get_color_hex)
            
            # Identify High Contrast Intersections (75th percentile vs 25th percentile)
            q_high = gdf_edges['betweenness'].quantile(0.75)
            q_low = gdf_edges['betweenness'].quantile(0.25)
            
            high_contrast_nodes = set()
            for node in G_undirected.nodes():
                incident_edges = G_undirected.edges(node, data=True)
                if len(incident_edges) > 1:
                    scores = [d.get('betweenness', 0) for _, _, d in incident_edges]
                    if any(s >= q_high for s in scores) and any(s <= q_low for s in scores):
                        high_contrast_nodes.add(node)

            # Store processed structural data in session state
            st.session_state.graph_data = {
                "G_undirected": G_undirected,
                "gdf_nodes": gdf_nodes,
                "gdf_edges": gdf_edges,
                "center_lat": center_lat,
                "center_lon": center_lon,
                "high_contrast_nodes": high_contrast_nodes,
                "search_query": search_query
            }

        except Exception as e:
            st.error(f"Error generating analysis: {e}")

# -----------------------------------------------------------------------------
# 4. Dynamic Render Section (Runs instantly when toggling checkboxes)
# -----------------------------------------------------------------------------
if st.session_state.graph_data is not None:
    data = st.session_state.graph_data
    gdf_edges = data["gdf_edges"]
    gdf_nodes = data["gdf_nodes"]
    high_contrast_nodes = data["high_contrast_nodes"]
    G_undirected = data["G_undirected"]
    
    # -------------------------------------------------------------------------
    # Map Reconstruction (Fixes Issue 2)
    # -------------------------------------------------------------------------
    m = folium.Map(location=[data["center_lat"], data["center_lon"]], zoom_start=16, tiles="CartoDB dark_matter")
    
    # Draw Edges
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
    
    # Draw Nodes with 0.4 Opacity (60% Transparent)
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
                fill_opacity=0.4,
                opacity=0.4,
                popup="High-Contrast Intersection (Boundary Junction)"
            ).add_to(m)
        elif not is_contrast and show_standard_nodes:
            folium.CircleMarker(
                location=[lat, lon],
                radius=2.5,
                color='#FFFFFF',
                fill=True,
                fill_color='#FFFFFF',
                fill_opacity=0.4,
                opacity=0.4,
                tooltip=f"Intersection ID: {node_id}"
            ).add_to(m)

    # Legend Overlay
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

    st.subheader("1. Interactive Space Syntax Map")
    st_folium(m, width=1000, height=520, key=f"map_{show_standard_nodes}_{show_contrast_nodes}", returned_objects=[])

    # Feature Explanations (Fixes Issue 3 formatting)
    st.markdown("### Feature Explanations & Urban Implications")
    st.markdown("""
    * **Axial Segments (Rainbow Scale):** Calculated by computing topological shortest paths across the street network graph.
      * *Implication:* Red/Orange segments represent primary movement arteries that accumulate high pedestrian or vehicular throughput. Blue/Purple segments indicate segregated, quiet streets suitable for residential zones.
    * **High-Contrast Intersections (Magenta Dots):** Calculated by identifying nodes where a high-centrality street (at or above the 75th percentile) directly connects to a low-centrality street (at or below the 25th percentile).
      * *Implication:* These represent critical urban decision points or transitional boundaries (e.g., exiting a major arterial road directly into a quiet alleyway or pedestrianized precinct).
    * **Standard Intersections (White Dots):** Calculated as topological graph vertices where street lines join or split.
      * *Implication:* Represents physical connectivity density. Highly dense node clusters highlight fine-grained street grids with high walkability potential.
    """)

    st.write("---")

    # -------------------------------------------------------------------------
    # 5. Justified Topological Graph / J-Graph Generation (Fixes Issue 4)
    # -------------------------------------------------------------------------
    st.subheader("2. Justified Topological Graph (J-Graph)")
    
    # Pick root carrier node (highest degree/centrality vertex as root space 0)
    root_node = max(G_undirected.nodes(), key=lambda n: G_undirected.degree(n))
    
    # Calculate step depth from root node
    depths = nx.single_source_shortest_path_length(G_undirected, root_node)
    
    # Limit max depth display for rendering clarity
    max_depth = min(max(depths.values()), 10)
    
    # Organize nodes by horizontal depth levels
    level_nodes = {d: [] for d in range(max_depth + 1)}
    for node, depth in depths.items():
        if depth <= max_depth:
            level_nodes[depth].append(node)
            
    fig_jgraph, ax_jg = plt.subplots(figsize=(12, 8), dpi=200)
    
    # Compute coordinates for J-Graph layout
    pos = {}
    for depth, nodes in level_nodes.items():
        n_nodes = len(nodes)
        for idx, n in enumerate(nodes):
            # Spread horizontally across x-axis, stack vertically by step depth y-axis
            x = (idx + 1) / (n_nodes + 1) if n_nodes > 0 else 0.5
            y = depth
            pos[n] = (x, y)

    # Draw horizontal step-depth guide lines
    for d in range(max_depth + 1):
        ax_jg.axhline(y=d, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax_jg.text(-0.02, d, f"Depth {d}", va='center', ha='right', fontsize=9, fontweight='bold', color='#333333')

    # Draw topological connections
    sub_nodes = [n for depth in range(max_depth + 1) for n in level_nodes[depth]]
    sub_G = G_undirected.subgraph(sub_nodes)
    
    for u, v in sub_G.edges():
        if u in pos and v in pos:
            x_vals = [pos[u][0], pos[v][0]]
            y_vals = [pos[u][1], pos[v][1]]
            ax_jg.plot(x_vals, y_vals, color='#666666', linewidth=0.7, zorder=1)

    # Color nodes based on space syntax classification
    node_colors = []
    for n in sub_nodes:
        if n == root_node:
            node_colors.append('#0000FF')  # Primary Root Carrier Space
        elif n in high_contrast_nodes:
            node_colors.append('#FF0055')  # High-Contrast Transition Boundary
        elif G_undirected.degree(n) >= 4:
            node_colors.append('#FFA500')  # Major Junction / Distributed Space
        else:
            node_colors.append('#FFFF00')  # Standard Linear Pathway Node

    x_coords = [pos[n][0] for n in sub_nodes]
    y_coords = [pos[n][1] for n in sub_nodes]
    
    ax_jg.scatter(x_coords, y_coords, s=140, c=node_colors, edgecolors='black', linewidths=0.8, zorder=2)
    
    # Annotate Root space
    ax_jg.text(pos[root_node][0], pos[root_node][1] - 0.15, "0 (Root Carrier)", ha='center', fontsize=8, fontweight='bold', color='blue')

    ax_jg.set_ylim(-0.5, max_depth + 0.8)
    ax_jg.set_xlim(-0.05, 1.05)
    ax_jg.axis('off')
    ax_jg.set_title(f"Justified Step-Depth Graph (J-Graph) from Root Junction #{root_node}", fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    st.pyplot(fig_jgraph)
    
    st.markdown("""
    **J-Graph Structural & Topological Explanation:**
    * **Vertical Levels (Step Depth 0 to N):** Represents the number of spatial direction changes or topological steps required to reach any urban space from the primary Root Carrier space ($0$).
    * **Tree Ring/Branch Spread:** Deeper levels (higher step depth) indicate highly private, segregated, or non-distributed spaces. Shallower trees indicate highly permeable, integrated urban grids.
    * **Node Color Classification:**
      * **Blue Circle (0):** Carrier space / Primary urban origin junction.
      * **Magenta Circles:** High-contrast decision boundaries intersecting high/low centrality streets.
      * **Orange Circles:** Major street intersections connecting $\ge 4$ directions.
      * **Yellow Circles:** Standard pathway spaces / local street segments.
    """)

    # -------------------------------------------------------------------------
    # Sidebar QGIS & Export Section
    # -------------------------------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("Export Results for QGIS / CAD")
    
    # GeoJSON Export
    gdf_export = gdf_edges[['street_name', 'betweenness', 'length', 'geometry']]
    geojson_str = gdf_export.to_json()
    st.sidebar.download_button(
        label="Download GIS Vector (.geojson)",
        data=geojson_str,
        file_name="space_syntax_singapore.geojson",
        mime="application/geo+json"
    )
    
    # Static PNG Export
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

    png_io = io.BytesIO()
    fig_export.savefig(png_io, format='png', dpi=300, bbox_inches='tight', facecolor='#111111')
    st.sidebar.download_button(
        label="Download High-Res Map (.png)",
        data=png_io.getvalue(),
        file_name="space_syntax_map.png",
        mime="image/png"
    )