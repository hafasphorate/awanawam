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
from shapely.ops import linemerge, substring
from folium.plugins import Draw
from geopy.distance import geodesic

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

if "street_plan_paths" not in st.session_state:
    st.session_state.street_plan_paths = []

if "desire_path_analysis" not in st.session_state:
    st.session_state.desire_path_analysis = []

if "path_map_revision" not in st.session_state:
    st.session_state.path_map_revision = 0

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
    overpass_endpoints = [
        "https://overpass-api.de/api",
        "https://overpass.kumi.systems/api",
        "https://overpass.private.coffee/api",
    ]
    original_endpoint = ox.settings.overpass_url
    graph = None
    errors = []
    try:
        for endpoint in overpass_endpoints:
            ox.settings.overpass_url = endpoint
            try:
                graph = ox.convert.to_undirected(
                    ox.graph_from_point(
                        (center_lat, center_lon), dist=distance, network_type=kind
                    )
                )
                break
            except Exception as error:
                errors.append(f"{endpoint}: {error}")
        if graph is None:
            raise ConnectionError(
                "All Overpass services were unavailable. "
                "Please try again shortly or reduce the analysis radius.\n"
                + "\n".join(errors)
            )
    finally:
        ox.settings.overpass_url = original_endpoint
    nodes, edges = ox.convert.graph_to_gdfs(graph)
    edges['street_name'] = edges['name'].apply(clean_name)
    return {
        "G_undirected": graph, "gdf_nodes": nodes, "gdf_edges": edges,
        "center_lat": center_lat, "center_lon": center_lon, "search_query": location,
    }


def street_paths_from_edges(edges):
    return [row.geometry for _, row in edges.iterrows()
            if row.geometry.geom_type == "LineString"]


if st.session_state.plan_data is not None and not st.session_state.street_plan_paths:
    st.session_state.street_plan_paths = street_paths_from_edges(
        st.session_state.plan_data["gdf_edges"]
    )


def desire_path_features():
    return [{
        "type": "Feature", "properties": {"path_id": index + 1, "length_m": round(path_length_m(path), 2)},
        "geometry": mapping(path),
    } for index, path in enumerate(st.session_state.desire_paths)]


def split_path(path, percentage):
    cut_distance = path.length * percentage / 100.0
    return (
        substring(path, 0.0, cut_distance),
        substring(path, cut_distance, path.length),
    )


def path_length_m(path):
    return sum(
        geodesic((start[1], start[0]), (end[1], end[0])).meters
        for start, end in zip(path.coords, path.coords[1:])
    )


def json_safe(value):
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError, OverflowError):
        return str(value)


def geojson_feature_collection(gdf):
    features = []
    for _, row in gdf.iterrows():
        properties = {
            column: json_safe(value)
            for column, value in row.items()
            if column != "geometry"
        }
        features.append({
            "type": "Feature",
            "properties": properties,
            "geometry": mapping(row.geometry) if row.geometry is not None else None,
        })
    return {"type": "FeatureCollection", "features": features}


def add_cut_preview(map_object, path, percentage):
    if path is None or path.length == 0:
        return
    cut_distance = path.length * percentage / 100.0
    cut_point = path.interpolate(cut_distance)
    tangent_distance = min(path.length * 0.01, path.length / 2.0)
    before = path.interpolate(max(0.0, cut_distance - tangent_distance))
    after = path.interpolate(min(path.length, cut_distance + tangent_distance))
    tangent_x = after.x - before.x
    tangent_y = after.y - before.y
    tangent_length = (tangent_x ** 2 + tangent_y ** 2) ** 0.5
    if tangent_length > 0:
        half_length = min(path.length * 0.04, path.length / 2.0)
        normal_x = -tangent_y / tangent_length * half_length
        normal_y = tangent_x / tangent_length * half_length
        preview_line = LineString([
            (cut_point.x - normal_x, cut_point.y - normal_y),
            (cut_point.x + normal_x, cut_point.y + normal_y),
        ])
        folium.PolyLine(
            locations=[[point[1], point[0]] for point in preview_line.coords],
            color="#FF3366", weight=8, opacity=1.0,
            tooltip=f"Perpendicular cut preview at {percentage}%"
        ).add_to(map_object)
    folium.CircleMarker(
        location=[cut_point.y, cut_point.x], radius=6,
        color="#FFFFFF", weight=2, fill=True, fill_color="#FF3366", fill_opacity=1.0,
        tooltip=f"Cut point: {percentage}%"
    ).add_to(map_object)


def update_desire_paths_from_map(drawings):
    """Merge newly drawn lines without clearing paths when Folium has no event."""
    if not drawings:
        return False
    incoming_paths = []
    for feature in drawings:
        if feature.get("geometry", {}).get("type") != "LineString":
            continue
        coordinates = feature["geometry"].get("coordinates", [])
        if len(coordinates) >= 2:
            incoming_paths.append(LineString([(point[0], point[1]) for point in coordinates]))
    paths = list(st.session_state.desire_paths)
    for path in incoming_paths:
        if not any(path.equals(existing) for existing in paths):
            paths.append(path)
    if incoming_paths:
        st.session_state.desire_paths = paths
        return True
    return False


def render_path_editor(key_prefix):
    """Provide deterministic remove and cut operations for plan and desire paths."""
    path_options = [
        (f"street:{index}:{path.wkb_hex}", "street", index, path)
        for index, path in enumerate(st.session_state.street_plan_paths)
    ] + [
        (f"desire:{index}:{path.wkb_hex}", "desire", index, path)
        for index, path in enumerate(st.session_state.desire_paths)
    ]
    if not path_options:
        return None, 50

    st.markdown("#### Path Editor")
    option_ids = [option[0] for option in path_options]
    option_lookup = {option[0]: option for option in path_options}
    selection_key = f"{key_prefix}_path_select"
    if st.session_state.get(selection_key) not in option_lookup:
        st.session_state[selection_key] = option_ids[0]
    st.selectbox(
        "Path to edit", option_ids,
        format_func=lambda option_id: (
            f"Street path {option_lookup[option_id][2] + 1}"
            if option_lookup[option_id][1] == "street"
            else f"Desire path {option_lookup[option_id][2] + 1}"
        ), key=selection_key
    )
    selected_id = st.session_state[selection_key]
    selected_path = option_lookup[selected_id][3]
    edit_col, remove_col = st.columns(2)
    with edit_col:
        cut_position = st.slider(
            "Cut position (%)", 1, 99, 50, key=f"{key_prefix}_cut_position"
        )
        if st.button("Cut selected path", key=f"{key_prefix}_cut_button"):
            selected_id = st.session_state[selection_key]
            _, path_type, path_index, path = option_lookup[selected_id]
            first_path, second_path = split_path(path, cut_position)
            target_paths = (st.session_state.street_plan_paths
                            if path_type == "street" else st.session_state.desire_paths)
            target_paths[path_index:path_index + 1] = [first_path, second_path]
            st.session_state.desire_path_analysis = []
            st.session_state.path_map_revision += 1
            st.rerun()
    with remove_col:
        if st.button("Remove selected path", key=f"{key_prefix}_remove_button"):
            selected_id = st.session_state[selection_key]
            _, path_type, path_index, _ = option_lookup[selected_id]
            target_paths = (st.session_state.street_plan_paths
                            if path_type == "street" else st.session_state.desire_paths)
            target_paths.pop(path_index)
            st.session_state.desire_path_analysis = []
            st.session_state.path_map_revision += 1
            st.rerun()
    return selected_path, cut_position


def analyze_desire_paths(graph, paths, edge_centrality):
    """Snap each drawn path to the network and summarize its shortest route."""
    def edge_score(u, v, edge_key):
        return edge_centrality.get(
            (u, v, edge_key),
            edge_centrality.get(
                (v, u, edge_key),
                edge_centrality.get((u, v), edge_centrality.get((v, u), 0.0)),
            ),
        )

    results = []
    for index, drawn_path in enumerate(paths):
        try:
            start = min(graph.nodes, key=lambda node: (
                (graph.nodes[node]["x"] - drawn_path.coords[0][0]) ** 2
                + (graph.nodes[node]["y"] - drawn_path.coords[0][1]) ** 2
            ))
            end = min(graph.nodes, key=lambda node: (
                (graph.nodes[node]["x"] - drawn_path.coords[-1][0]) ** 2
                + (graph.nodes[node]["y"] - drawn_path.coords[-1][1]) ** 2
            ))
            route = nx.shortest_path(graph, start, end, weight="length")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            results.append({
                "path_id": index + 1, "drawn_geometry": mapping(drawn_path),
                "drawn_length_m": round(path_length_m(drawn_path), 2),
                "route_geometry": None,
                "route_length_m": 0.0,
                "centrality": 0.0,
                "mean_betweenness": 0.0,
                "max_betweenness": 0.0,
                "network_edge_count": 0,
                "error": "No connected street route found",
            })
            continue
        route_edges = list(zip(route[:-1], route[1:]))
        scores = []
        route_lines = []
        route_length = 0.0
        for u, v in route_edges:
            edge_data = graph.get_edge_data(u, v)
            edge_key, attributes = min(edge_data.items(), key=lambda item: item[1].get("length", float("inf")))
            scores.append(edge_score(u, v, edge_key))
            route_length += float(attributes.get("length", 0.0))
            geometry = attributes.get("geometry")
            route_lines.append(geometry if geometry is not None else LineString([
                (graph.nodes[u]["x"], graph.nodes[u]["y"]),
                (graph.nodes[v]["x"], graph.nodes[v]["y"]),
            ]))
        if not route_edges:
            edge_records = (
                graph.edges(keys=True, data=True)
                if graph.is_multigraph() else
                ((u, v, None, attributes) for u, v, attributes in graph.edges(data=True))
            )
            nearest_edge = min(
                edge_records,
                key=lambda record: (
                    (record[3].get("geometry") or LineString([
                        (graph.nodes[record[0]]["x"], graph.nodes[record[0]]["y"]),
                        (graph.nodes[record[1]]["x"], graph.nodes[record[1]]["y"]),
                    ])).distance(drawn_path)
                ),
            )
            u, v, edge_key, attributes = nearest_edge
            geometry = attributes.get("geometry") or LineString([
                (graph.nodes[u]["x"], graph.nodes[u]["y"]),
                (graph.nodes[v]["x"], graph.nodes[v]["y"]),
            ])
            route_lines.append(geometry)
            route_length = float(attributes.get("length", path_length_m(geometry)))
            scores.append(edge_score(u, v, edge_key))
            route_edges = [(u, v)]
        merged_route = linemerge(route_lines)
        results.append({
            "path_id": index + 1,
            "drawn_geometry": mapping(drawn_path),
            "drawn_length_m": round(path_length_m(drawn_path), 2),
            "route_geometry": mapping(merged_route),
            "start_node": start,
            "end_node": end,
            "route_length_m": round(route_length, 2),
            "centrality": round(float(np.mean(scores)), 6) if scores else 0.0,
            "mean_betweenness": round(float(np.mean(scores)), 6) if scores else 0.0,
            "max_betweenness": round(float(np.max(scores)), 6) if scores else 0.0,
            "network_edge_count": len(route_edges),
        })
    return results


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
        imported_street_paths = imported.get("street_plan_paths")
        st.session_state.street_plan_paths = (
            [shape(feature["geometry"]) for feature in imported_street_paths]
            if imported_street_paths is not None
            else street_paths_from_edges(imported_edges)
        )
        st.session_state.desire_paths = [shape(feature["geometry"]) for feature in imported.get("desire_paths", [])]
        st.session_state.desire_path_analysis = imported.get("desire_path_analysis", [])
        for result in st.session_state.desire_path_analysis:
            result.setdefault("drawn_length_m", 0.0)
            result.setdefault("route_length_m", 0.0)
            result.setdefault("centrality", result.get("mean_betweenness", 0.0))
            result.setdefault("mean_betweenness", 0.0)
            result.setdefault("max_betweenness", 0.0)
            result.setdefault("network_edge_count", 0)
        st.success("Urban plan imported. The map and desire paths are ready to view.")
    except (KeyError, ValueError, TypeError) as error:
        st.error(f"Could not import this JSON package: {error}")


if st.button("Load Street Plan"):
    with st.spinner("Loading street network for preview..."):
        try:
            st.session_state.plan_data = load_plan(search_query, radius, network_type)
            st.session_state.graph_data = None
            st.session_state.street_plan_paths = street_paths_from_edges(
                st.session_state.plan_data["gdf_edges"]
            )
        except Exception as e:
            st.error(f"Error loading street plan: {e}")

# -----------------------------------------------------------------------------
# 4. Plan Preview and Interactive Rendering
# -----------------------------------------------------------------------------
if st.session_state.plan_data is not None and st.session_state.graph_data is None:
    plan = st.session_state.plan_data
    st.subheader("Street Plan Preview")
    st.caption("Draw desire paths on the plan, edit them below, then run the axial analysis.")
    selected_preview_path, preview_cut_position = render_path_editor("preview")
    preview_map = folium.Map(
        location=[plan["center_lat"], plan["center_lon"]], zoom_start=16, tiles="CartoDB dark_matter"
    )
    for index, path in enumerate(st.session_state.street_plan_paths):
        folium.PolyLine(
            locations=[[point[1], point[0]] for point in path.coords],
            color="#35B7FF", weight=3, opacity=0.8,
            tooltip=f"Street path {index + 1}"
        ).add_to(preview_map)
    for index, path in enumerate(st.session_state.desire_paths):
        folium.PolyLine(
            locations=[[point[1], point[0]] for point in path.coords],
            color="#FFD166", weight=5, opacity=0.95,
            tooltip=f"Desire path {index + 1} ({path_length_m(path):.1f} m)"
        ).add_to(preview_map)
    add_cut_preview(preview_map, selected_preview_path, preview_cut_position)
    Draw(
        export=False,
        draw_options={"polyline": {"shapeOptions": {"color": "#FFD166", "weight": 5}}, "polygon": False, "rectangle": False, "circle": False, "marker": False, "circlemarker": False},
        edit_options={"edit": True, "remove": True},
    ).add_to(preview_map)
    preview_result = st_folium(
        preview_map, width=1000, height=520,
        key=f"plan_preview_{st.session_state.path_map_revision}", returned_objects=["all_drawings"]
    )
    drawings = (preview_result or {}).get("all_drawings") or []
    update_desire_paths_from_map(drawings)
    st.info(f"{len(st.session_state.desire_paths)} desire path(s) staged for analysis.")

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
            st.session_state.desire_path_analysis = analyze_desire_paths(
                G_undirected, st.session_state.desire_paths, edge_centrality
            )
            st.session_state.selected_node = None

        except Exception as e:
            st.error(f"Error generating analysis: {e}")


# 5. Map & Interactive Rendering
# -----------------------------------------------------------------------------
if st.session_state.graph_data is not None:
    data = st.session_state.graph_data
    gdf_edges = data["gdf_edges"]
    gdf_nodes = data["gdf_nodes"]
    high_contrast_nodes = data["high_contrast_nodes"]
    G_undirected = data["G_undirected"]
    
    selected_node = st.session_state.selected_node
    selected_preview_path, preview_cut_position = render_path_editor("analysis")

    # Map Rendering
    m = folium.Map(location=[data["center_lat"], data["center_lon"]], zoom_start=16, tiles="CartoDB dark_matter")

    for index, path in enumerate(st.session_state.desire_paths):
        folium.PolyLine(
            locations=[[point[1], point[0]] for point in path.coords],
            color="#FFD166", weight=5, opacity=0.95,
            tooltip=f"Desire path {index + 1} ({path_length_m(path):.1f} m)"
        ).add_to(m)
    for result in st.session_state.desire_path_analysis:
        if result.get("route_geometry") is None:
            continue
        folium.GeoJson(
            result["route_geometry"],
            name=f"Analyzed route {result['path_id']}",
            style_function=lambda feature: {"color": "#FFFFFF", "weight": 3, "opacity": 0.9},
            tooltip=f"Desire path {result['path_id']} | Centrality: {result['centrality']:.6f}",
        ).add_to(m)
        for node_label in ("start_node", "end_node"):
            node_id = result.get(node_label)
            if node_id in gdf_nodes.index:
                node = gdf_nodes.loc[node_id].geometry
                folium.CircleMarker(
                    location=[node.y, node.x], radius=5,
                    color="#FF3366", fill=True, fill_color="#FF3366", fill_opacity=0.95,
                    tooltip=f"Desire path {result['path_id']} {node_label.replace('_', ' ')}",
                ).add_to(m)
    
    # Draw the analyzed network using the edited street geometry.
    street_rows = [row for _, row in gdf_edges.iterrows()
                   if row.geometry.geom_type == "LineString"]
    for index, path in enumerate(st.session_state.street_plan_paths):
        row = street_rows[min(index, len(street_rows) - 1)]
        folium.PolyLine(
            locations=[[point[1], point[0]] for point in path.coords],
            color=row["hex_color"],
            weight=3.5,
            opacity=0.85,
            tooltip=f"Street: {row['street_name']} | Centrality: {row['betweenness']:.4f}"
        ).add_to(m)
    add_cut_preview(m, selected_preview_path, preview_cut_position)
    
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
    if st.session_state.desire_path_analysis:
        st.subheader("Desire Path Axial Results")
        st.dataframe(st.session_state.desire_path_analysis, use_container_width=True)

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

    export_edges = gdf_edges.reset_index().drop(columns=["geometry"], errors="ignore")
    export_edges["geometry"] = gdf_edges.geometry.to_numpy()
    export_nodes = gdf_nodes.reset_index().rename(columns={"osmid": "node_id"})
    export_nodes = export_nodes.drop(columns=["geometry"], errors="ignore")
    export_nodes["geometry"] = gdf_nodes.geometry.to_numpy()
    json_package = {
        "format": "awanawam.urban_space_syntax.v1",
        "plan": {
            "search_query": data["search_query"],
            "center_lat": data["center_lat"],
            "center_lon": data["center_lon"],
            "edges": geojson_feature_collection(export_edges),
            "nodes": geojson_feature_collection(export_nodes),
        },
        "desire_paths": desire_path_features(),
        "street_plan_paths": [{
            "type": "Feature", "properties": {"path_id": index + 1},
            "geometry": mapping(path),
        } for index, path in enumerate(st.session_state.street_plan_paths)],
        "desire_path_analysis": st.session_state.desire_path_analysis,
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
    geojson_str = json.dumps(geojson_feature_collection(gdf_export))
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