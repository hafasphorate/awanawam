import streamlit as st
import osmnx as ox
import networkx as nx
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import numpy as np
import io
import json
from shapely.geometry import LineString, mapping, shape
from folium.plugins import Draw

st.set_page_config(page_title="Urban Space Syntax Analysis", layout="wide")

# -----------------------------------------------------------------------------
# 1. Page Title & Definitions
# -----------------------------------------------------------------------------
st.title("Singapore Urban Space Syntax Analysis")

st.markdown("""
**Metric Definition:** **Betweenness Centrality** (used here as an axial proxy for *Space Syntax Choice / Integration*) measures the fraction of all shortest topological paths passing through a specific street segment within the network:
""")

st.latex(r"C_B(e) = \sum_{s \neq t \in V} \frac{\sigma_{st}(e)}{\sigma_{st}}")

st.markdown("""
* **Units:** Bounded dimensionless score from **0.0** (completely isolated / low choice) to **1.0** (maximum spatial movement / high choice). Higher values highlight primary movement trunks in the urban layout.
""")
st.write("---")

# -----------------------------------------------------------------------------
# 2. Controls & Instant Toggle Configuration
# -----------------------------------------------------------------------------
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    search_query = st.text_input("Location in Singapore", "Tiong Bahru, Singapore")
with col2:
    radius = st.slider("Analysis Radius (meters)", min_value=300, max_value=2000, value=600, step=100)
with col3:
    network_type = st.selectbox("Network Type", ["walk", "drive", "all"])

st.sidebar.header("Map Display Options")
show_standard_nodes = st.sidebar.checkbox("Show Standard Intersections (White)", value=True)
show_contrast_nodes = st.sidebar.checkbox("Show High-Contrast Intersections (Red)", value=True)

if "graph_data" not in st.session_state:
    st.session_state.graph_data = None

if "plan_data" not in st.session_state:
    st.session_state.plan_data = None

if "desire_paths" not in st.session_state:
    st.session_state.desire_paths = []

if "selected_node" not in st.session_state:
    st.session_state.selected_node = None

# -----------------------------------------------------------------------------
# 3. Plan Preview and Analysis Workflow
# -----------------------------------------------------------------------------
def clean_name(val):
    if isinstance(val, list):
        return ", ".join(str(v) for v in val if v)
    return str(val) if val and str(val) != 'nan' else 'Unnamed Segment'


def load_plan(location, distance, kind):
    gdf_place = ox.geocode_to_gdf(location)
    center_lat = gdf_place.geometry.iloc[0].centroid.y
    center_lon = gdf_place.geometry.iloc[0].centroid.x
    graph = ox.convert.to_undirected(
        ox.graph_from_point((center_lat, center_lon), dist=distance, network_type=kind)
    )
    nodes, edges = ox.convert.graph_to_gdfs(graph)
    edges['street_name'] = edges['name'].apply(clean_name)
    return {
        "G_undirected": graph, "gdf_nodes": nodes, "gdf_edges": edges,
        "center_lat": center_lat, "center_lon": center_lon, "search_query": location,
    }


def desire_path_features():
    return [{
        "type": "Feature", "properties": {"path_id": index + 1, "length": round(path.length, 2)},
        "geometry": mapping(path),
    } for index, path in enumerate(st.session_state.desire_paths)]


uploaded_json = st.file_uploader("Import Urban Space Syntax JSON", type=["json"])
if uploaded_json is not None:
    try:
        imported = json.load(uploaded_json)
        imported_edges = gpd.GeoDataFrame.from_features(imported["plan"]["edges"]["features"])
        imported_nodes = gpd.GeoDataFrame.from_features(imported["plan"]["nodes"]["features"])
        imported_edges.set_index(["u", "v", "key"], inplace=True)
        imported_nodes.set_index("node_id", inplace=True)
        st.session_state.plan_data = {
            "gdf_edges": imported_edges, "gdf_nodes": imported_nodes,
            "center_lat": imported["plan"]["center_lat"], "center_lon": imported["plan"]["center_lon"],
            "search_query": imported["plan"].get("search_query", "Imported plan"), "G_undirected": None,
        }
        st.session_state.graph_data = None
        st.session_state.desire_paths = [shape(feature["geometry"]) for feature in imported.get("desire_paths", [])]
        st.success("Urban plan imported. The map and desire paths are ready to view.")
    except (KeyError, ValueError, TypeError) as error:
        st.error(f"Could not import this JSON package: {error}")


if st.button("Load Street Plan"):
    with st.spinner("Loading street network for preview..."):
        try:
            st.session_state.plan_data = load_plan(search_query, radius, network_type)
            st.session_state.graph_data = None
        except Exception as e:
            st.error(f"Error loading street plan: {e}")

if st.session_state.plan_data is not None and st.button("Run Axial Analysis"):
    with st.spinner("Calculating spatial depth and centrality..."):
        try:
            plan = st.session_state.plan_data
            G_undirected = plan["G_undirected"]
            if G_undirected is None:
                raise ValueError("Imported display data cannot be rerun; load the street plan to calculate new metrics.")
            edge_centrality = nx.edge_betweenness_centrality(G_undirected)
            nx.set_edge_attributes(G_undirected, edge_centrality, "betweenness")
            
            gdf_nodes, gdf_edges = ox.convert.graph_to_gdfs(G_undirected)
            
            gdf_edges['street_name'] = gdf_edges['name'].apply(clean_name)
            
            min_val = gdf_edges['betweenness'].min()
            max_val = gdf_edges['betweenness'].max()
            cmap = plt.get_cmap('turbo')
            
            def get_color_hex(val):
                norm = (val - min_val) / (max_val - min_val + 1e-6)
                return mcolors.to_hex(cmap(norm))
            
            gdf_edges['hex_color'] = gdf_edges['betweenness'].apply(get_color_hex)
            
            q_high = gdf_edges['betweenness'].quantile(0.75)
            q_low = gdf_edges['betweenness'].quantile(0.25)
            
            high_contrast_nodes = set()
            for node in G_undirected.nodes():
                incident_edges = G_undirected.edges(node, data=True)
                if len(incident_edges) > 1:
                    scores = [d.get('betweenness', 0) for _, _, d in incident_edges]
                    if any(s >= q_high for s in scores) and any(s <= q_low for s in scores):
                        high_contrast_nodes.add(node)

            st.session_state.graph_data = {
                "G_undirected": G_undirected,
                "gdf_nodes": gdf_nodes,
                "gdf_edges": gdf_edges,
                "center_lat": plan["center_lat"],
                "center_lon": plan["center_lon"],
                "high_contrast_nodes": high_contrast_nodes,
                "search_query": plan["search_query"]
            }
            st.session_state.selected_node = None

        except Exception as e:
            st.error(f"Error generating analysis: {e}")

# -----------------------------------------------------------------------------
# 4. Plan Preview and Interactive Rendering
# -----------------------------------------------------------------------------
if st.session_state.plan_data is not None and st.session_state.graph_data is None:
    plan = st.session_state.plan_data
    st.subheader("Street Plan Preview")
    st.caption("Draw desire paths on the plan, then run the axial analysis. Double-click a path to remove it.")
    preview_map = folium.Map(
        location=[plan["center_lat"], plan["center_lon"]], zoom_start=16, tiles="CartoDB dark_matter"
    )
    for _, row in plan["gdf_edges"].iterrows():
        if row.geometry.geom_type == "LineString":
            folium.PolyLine(
                locations=[[point[1], point[0]] for point in row.geometry.coords],
                color="#35B7FF", weight=3, opacity=0.8,
                tooltip=f"Street: {row['street_name']}"
            ).add_to(preview_map)
    for index, path in enumerate(st.session_state.desire_paths):
        folium.PolyLine(
            locations=[[point[1], point[0]] for point in path.coords],
            color="#FFD166", weight=5, opacity=0.95,
            tooltip=f"Desire path {index + 1} ({path.length:.1f} map units)"
        ).add_to(preview_map)
    Draw(
        export=False,
        draw_options={"polyline": {"shapeOptions": {"color": "#FFD166", "weight": 5}}, "polygon": False, "rectangle": False, "circle": False, "marker": False, "circlemarker": False},
        edit_options={"edit": True, "remove": True},
    ).add_to(preview_map)
    preview_result = st_folium(preview_map, width=1000, height=520, key="plan_preview", returned_objects=["all_drawings"])
    drawings = (preview_result or {}).get("all_drawings") or []
    drawn_paths = []
    for feature in drawings:
        if feature.get("geometry", {}).get("type") == "LineString":
            coords = feature["geometry"]["coordinates"]
            if len(coords) >= 2:
                drawn_paths.append(LineString([(point[0], point[1]) for point in coords]))
    if drawings:
        st.session_state.desire_paths = drawn_paths
    st.info(f"{len(st.session_state.desire_paths)} desire path(s) staged for analysis.")


# 5. Map & Interactive Rendering
# -----------------------------------------------------------------------------
if st.session_state.graph_data is not None:
    data = st.session_state.graph_data
    gdf_edges = data["gdf_edges"]
    gdf_nodes = data["gdf_nodes"]
    high_contrast_nodes = data["high_contrast_nodes"]
    G_undirected = data["G_undirected"]
    
    selected_node = st.session_state.selected_node

    # Map Rendering
    m = folium.Map(location=[data["center_lat"], data["center_lon"]], zoom_start=16, tiles="CartoDB dark_matter")

    for index, path in enumerate(st.session_state.desire_paths):
        folium.PolyLine(
            locations=[[point[1], point[0]] for point in path.coords],
            color="#FFD166", weight=5, opacity=0.95,
            tooltip=f"Desire path {index + 1} ({path.length:.1f} map units)"
        ).add_to(m)
    
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
    
    # Draw Nodes with standard / highlight styling
    for node_id, row in gdf_nodes.iterrows():
        lat, lon = row.geometry.y, row.geometry.x
        is_contrast = node_id in high_contrast_nodes
        is_selected = (node_id == selected_node)
        
        if is_selected:
            folium.CircleMarker(
                location=[lat, lon],
                radius=12,
                color='#00FFFF',
                fill=True,
                fill_color='#00FFFF',
                fill_opacity=0.9,
                popup=f"Selected Node"
            ).add_to(m)
        elif is_contrast and show_contrast_nodes:
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color='#FF0055',
                fill=True,
                fill_color='#FF0055',
                fill_opacity=0.4,
                popup="High-Contrast Intersection"
            ).add_to(m)
        elif not is_contrast and show_standard_nodes:
            folium.CircleMarker(
                location=[lat, lon],
                radius=2.5,
                color='#FFFFFF',
                fill=True,
                fill_color='#FFFFFF',
                fill_opacity=0.4,
                tooltip=f"Intersection"
            ).add_to(m)

    # Legend Overlay for Map
    legend_html = '''
    <div style="position: fixed; bottom: 30px; left: 30px; width: 250px; 
                background-color: rgba(17,17,17,0.9); z-index:9999; font-size:12px; color: white;
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
    st_folium(m, width=1000, height=520, key="main_map", returned_objects=[])

    # Feature Explanations for Map
    st.markdown("### Feature Explanations & Urban Implications")
    st.markdown("""
    * **Axial Segments (Rainbow Scale):** Calculated by computing topological shortest paths across the street network graph.
      * *Implication:* Red/Orange segments represent primary movement arteries that accumulate high pedestrian or vehicular throughput. Blue/Purple segments indicate segregated, quiet streets suitable for residential zones.
    * **High-Contrast Intersections (Magenta Dots):** Calculated by identifying nodes where a high-centrality street ($\ge$ 75th percentile) directly connects to a low-centrality street ($\le$ 25th percentile).
      * *Implication:* These represent critical urban decision points or transitional boundaries (e.g., exiting a major arterial road directly into a quiet alleyway or pedestrianized precinct).
    * **Standard Intersections (White Dots):** Calculated as topological graph vertices where street lines join or split.
      * *Implication:* Represents physical connectivity density. Highly dense node clusters highlight fine-grained street grids with high walkability potential.
    """)

    st.write("---")

    # -------------------------------------------------------------------------
    # 5. Dark-Themed Plotly J-Graph with Road Names on Edges
    # -------------------------------------------------------------------------
    st.subheader("2. Interactive Justified Topological Graph (J-Graph)")
    st.caption("Hover over nodes or connection lines to inspect street names and depth relationships. Click a node to select and highlight it on the map above.")

    root_node = max(G_undirected.nodes(), key=lambda n: G_undirected.degree(n))
    depths = nx.single_source_shortest_path_length(G_undirected, root_node)
    max_depth = min(max(depths.values()), 10)
    
    level_nodes = {d: [] for d in range(max_depth + 1)}
    for node, depth in depths.items():
        if depth <= max_depth:
            level_nodes[depth].append(node)
            
    pos = {}
    for depth, nodes in level_nodes.items():
        n_nodes = len(nodes)
        for idx, n in enumerate(nodes):
            x = (idx + 1) / (n_nodes + 1) if n_nodes > 0 else 0.5
            y = depth
            pos[n] = (x, y)

    sub_nodes = [n for depth in range(max_depth + 1) for n in level_nodes[depth]]
    sub_G = G_undirected.subgraph(sub_nodes)

    # Build Connection Lines and Midpoint Road Labels
    edge_x, edge_y, edge_hover = [], [], []
    mid_x, mid_y, mid_labels = [], [], []

    for u, v, d in sub_G.edges(data=True):
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            street_name = d.get('street_name', d.get('name', 'Unnamed Street'))
            edge_hover.extend([f"Street: {street_name}", f"Street: {street_name}", None])
            
            # Compute line midpoint for road label placement
            mid_x.append((x0 + x1) / 2.0)
            mid_y.append((y0 + y1) / 2.0)
            mid_labels.append(street_name)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.2, color='#AAAAAA'),
        hoverinfo='text',
        text=edge_hover,
        mode='lines'
    )

    # Road Name Text Traces positioned at line midpoints
    edge_label_trace = go.Scatter(
        x=mid_x, y=mid_y,
        mode='text',
        text=mid_labels,
        textposition='middle center',
        hoverinfo='none',
        textfont=dict(color='#00FFFF', size=9)
    )

    # Node Traces (No raw numerical IDs on top)
    node_x, node_y, node_hover, node_color, custom_data = [], [], [], [], []
    
    for n in sub_nodes:
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        custom_data.append(n)
        
        incident = sub_G.edges(n, data=True)
        streets = list(set([d.get('street_name', 'Unnamed') for _, _, d in incident if 'street_name' in d]))
        streets_str = ", ".join(streets) if streets else "Local Segment"
        
        lat = gdf_nodes.loc[n].geometry.y if n in gdf_nodes.index else 0
        lon = gdf_nodes.loc[n].geometry.x if n in gdf_nodes.index else 0
        
        node_hover.append(
            f"<b>Step Depth:</b> {depths[n]}<br>"
            f"<b>Adjacent Streets:</b> {streets_str}<br>"
            f"<b>Coordinates:</b> ({lat:.4f}, {lon:.4f})"
        )
        
        if n == root_node:
            node_color.append('#3388FF')  # Root Carrier (Blue)
        elif n in high_contrast_nodes:
            node_color.append('#FF0055')  # High-Contrast Transition (Magenta/Red)
        elif G_undirected.degree(n) >= 4:
            node_color.append('#FFA500')  # Major Junction (Orange)
        else:
            node_color.append('#FFFF00')  # Local Street Node (Yellow)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        hovertext=node_hover,
        marker=dict(
            size=16,
            color=node_color,
            line=dict(width=1.5, color='#FFFFFF')
        ),
        customdata=custom_data
    )

    # Assemble Plotly Figure with Dark Theme (#111111)
    fig_jg = go.Figure(data=[edge_trace, edge_label_trace, node_trace])
    fig_jg.update_layout(
        title=dict(
            text=f"Justified Step-Depth Graph (J-Graph) from Root Carrier Space",
            font=dict(color='#FFFFFF', size=14)
        ),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=40, r=40, t=50),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(
            title=dict(text="Step Depth Level", font=dict(color='#FFFFFF')),
            tickmode='linear',
            dtick=1,
            showgrid=True,
            gridcolor='#333333',
            tickfont=dict(color='#FFFFFF')
        ),
        height=620,
        paper_bgcolor='#111111',
        plot_bgcolor='#1E1E1E'
    )

    event = st.plotly_chart(fig_jg, use_container_width=True, on_select="rerun", key="jgraph_chart")
    
    if event and "selection" in event and event["selection"]["points"]:
        point = event["selection"]["points"][0]
        if "customdata" in point:
            clicked_node = point["customdata"]
            if st.session_state.selected_node != clicked_node:
                st.session_state.selected_node = clicked_node
                st.rerun()

    # J-Graph Legend & Structural Explanatory Text
    st.markdown("""
    **J-Graph Legend & Structural Breakdown:**
    * **Vertical Axis (Step Depth 0 to N):** Represents the topological step distance (number of direction changes/turns) required to reach any street segment starting from the primary Root Carrier space (Depth 0).
    * **Cyan Street Labels:** Displayed directly on connection edges to identify the corresponding road name between topological intersections.
    * **Node Space Classification Legend:**
      * <span style="color:#3388FF; font-weight:bold;">● Blue Node:</span> Primary Root Carrier Space (origin space with highest network degree).
      * <span style="color:#FF0055; font-weight:bold;">● Magenta Node:</span> High-Contrast Transition Boundary (connects high/low choice streets).
      * <span style="color:#FFA500; font-weight:bold;">● Orange Node:</span> Major Junction / Distributed Intersection ($\ge$ 4 connected pathways).
      * <span style="color:#FFFF00; font-weight:bold;">● Yellow Node:</span> Standard Local Street Segment / Non-distributed Pathway.
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Sidebar QGIS & Export Section
    # -------------------------------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("Export Results for QGIS / CAD")

    export_edges = gdf_edges.reset_index()
    export_nodes = gdf_nodes.reset_index().rename(columns={"osmid": "node_id"})
    export_edges = export_edges.drop(columns=["geometry"], errors="ignore").join(gdf_edges.geometry.rename("geometry"))
    export_nodes = export_nodes.drop(columns=["geometry"], errors="ignore").join(gdf_nodes.geometry.rename("geometry"))
    json_package = {
        "format": "awanawam.urban_space_syntax.v1",
        "plan": {
            "search_query": data["search_query"],
            "center_lat": data["center_lat"],
            "center_lon": data["center_lon"],
            "edges": json.loads(export_edges.to_json()),
            "nodes": json.loads(export_nodes.to_json()),
        },
        "desire_paths": desire_path_features(),
        "analysis": {
            "metric": "edge_betweenness_centrality",
            "edge_count": len(gdf_edges),
            "desire_path_count": len(st.session_state.desire_paths),
        },
    }
    st.sidebar.download_button(
        label="Download Complete Analysis (.json)",
        data=json.dumps(json_package, indent=2),
        file_name="urban_space_syntax_analysis.json",
        mime="application/json",
    )
    
    gdf_export = gdf_edges[['street_name', 'betweenness', 'length', 'geometry']]
    geojson_str = gdf_export.to_json()
    st.sidebar.download_button(
        label="Download GIS Vector (.geojson)",
        data=geojson_str,
        file_name="space_syntax_singapore.geojson",
        mime="application/geo+json"
    )
    
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