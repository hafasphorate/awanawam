import streamlit as st
import networkx as nx
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import io
import json
import os
import time
from shapely.geometry import LineString, mapping, shape
from shapely.ops import linemerge, substring
from folium.plugins import Draw
from geopy.distance import geodesic
from utils.navigation import render_home_button
from utils.urban_paths import (
    entrance_barrier_nodes_within_radius,
    entrance_barrier_paths,
)

st.set_page_config(page_title="Urban Spatial Interaction", layout="wide")
render_home_button()


def dark_map(location, zoom_start=16):
    """Folium map with a keyless dark basemap (Carto's dark_matter now requires an API key)."""
    return folium.Map(
        location=location, zoom_start=zoom_start, tiles=None, max_zoom=19
    ).add_child(folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ",
        name="Dark basemap", max_native_zoom=16, max_zoom=19,
    ))

# -----------------------------------------------------------------------------
# 1. Page Title & Definitions
# -----------------------------------------------------------------------------
st.title("Module 1: Urban Spatial Interaction")

st.markdown("""
**Metric Definition:** **Betweenness Centrality** (used here as an axial proxy for *Space Syntax Choice / Integration*) measures the fraction of all shortest topological paths passing through a specific street segment within the network:
""")

st.latex(r"C_B(e) = \sum_{s \neq t \in V} \frac{\sigma_{st}(e)}{\sigma_{st}}")

st.markdown("""
* **Units:** Bounded dimensionless score from **0.0** (completely isolated / low choice) to **1.0** (maximum spatial movement / high choice). Higher values highlight primary movement trunks in the urban layout.
""")
st.write("---")

tab_interaction, tab_gravity = st.tabs(["Urban Spatial Interaction", "Gravity Model Density"])
# Tab 1 spans the rest of the page; it is closed explicitly before Tab 2 at the end of the file.
tab_interaction.__enter__()

# -----------------------------------------------------------------------------
# 2. Controls & Instant Toggle Configuration
# -----------------------------------------------------------------------------
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    search_query = st.text_input("Location in Singapore", "Tiong Bahru, Singapore")
with col2:
    radius = int(st.number_input(
        "Analysis Radius (meters)", min_value=100, max_value=5000, value=600, step=1, format="%d"
    ))
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

if "entrance_barrier_paths" not in st.session_state:
    st.session_state.entrance_barrier_paths = []

if "street_plan_paths" not in st.session_state:
    st.session_state.street_plan_paths = []

if "desire_path_analysis" not in st.session_state:
    st.session_state.desire_path_analysis = []

if "entrance_barrier_analysis" not in st.session_state:
    st.session_state.entrance_barrier_analysis = []

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


@st.cache_data(show_spinner=False, ttl=3600, max_entries=8)
def load_plan(location, distance, kind):
    import osmnx as ox

    gdf_place = ox.geocode_to_gdf(location)
    center_lat = gdf_place.geometry.iloc[0].centroid.y
    center_lon = gdf_place.geometry.iloc[0].centroid.x
    overpass_endpoints = [
        "https://overpass-api.de/api",
        "https://overpass.private.coffee/api",
        "https://overpass.kumi.systems/api",
        "https://overpass.osm.jp/api",
        "https://overpass.openstreetmap.fr/api",
        "https://maps.mail.ru/osm/tools/overpass/api",
    ]
    original_endpoint = ox.settings.overpass_url
    original_timeout = ox.settings.requests_timeout
    graph = None
    errors = []
    try:
        # Public Overpass servers are often overloaded; fail over quickly and make a second pass.
        ox.settings.requests_timeout = 60
        for attempt in range(2):
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
                    errors.append(f"{endpoint}: {str(error)[:150]}")
            if graph is not None:
                break
            time.sleep(5)
        if graph is None:
            raise ConnectionError(
                "All Overpass services were unavailable. "
                "Please try again shortly or reduce the analysis radius.\n"
                + "\n".join(errors)
            )
    finally:
        ox.settings.overpass_url = original_endpoint
        ox.settings.requests_timeout = original_timeout
    nodes, edges = ox.convert.graph_to_gdfs(graph)
    edges['street_name'] = edges['name'].apply(clean_name)
    return {
        "G_undirected": graph, "gdf_nodes": nodes, "gdf_edges": edges,
        "center_lat": center_lat, "center_lon": center_lon, "search_query": location,
        "analysis_radius_m": distance,
    }


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ENTRANCE_BARRIER_FILE = os.path.join(DATA_DIR, "entrances and barriers.geojson")


@st.cache_data(show_spinner=False, ttl=3600, max_entries=4)
def load_entrance_barrier_data():
    return gpd.read_file(ENTRANCE_BARRIER_FILE)


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


def centrality_color(value, minimum, maximum):
    normalized = max(0.0, min(1.0, (value - minimum) / (maximum - minimum + 1e-6)))
    return mcolors.to_hex(plt.get_cmap("turbo")(normalized))


def assign_edge_colors(gdf_edges):
    if gdf_edges.empty or "betweenness" not in gdf_edges.columns:
        return gdf_edges
    minimum = float(gdf_edges["betweenness"].min())
    maximum = float(gdf_edges["betweenness"].max())
    gdf_edges["hex_color"] = gdf_edges["betweenness"].apply(
        lambda value: centrality_color(float(value), minimum, maximum)
    )
    return gdf_edges


def geometry_intersection_points(first, second):
    intersection = first.intersection(second)
    if intersection.is_empty:
        return []
    if intersection.geom_type == "Point":
        return [intersection]
    if intersection.geom_type in ("MultiPoint", "GeometryCollection"):
        return [geometry for geometry in intersection.geoms if geometry.geom_type == "Point"]
    if intersection.geom_type == "MultiLineString":
        return [line.interpolate(line.length / 2.0) for line in intersection.geoms if line.length]
    if intersection.geom_type == "LineString":
        return [intersection.interpolate(intersection.length / 2.0)]
    return []


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


def build_graph_from_gdfs(gdf_nodes, gdf_edges):
    graph = nx.Graph()
    for node_id, row in gdf_nodes.iterrows():
        graph.add_node(
            node_id,
            x=float(row.geometry.x),
            y=float(row.geometry.y),
            osmid=row.get("osmid"),
        )
    edge_df = gdf_edges.reset_index()
    for _, row in edge_df.iterrows():
        u = row.get("u")
        v = row.get("v")
        if u is None or v is None:
            continue
        geometry = row.get("geometry")
        if geometry is None:
            geometry = LineString([
                (graph.nodes[u]["x"], graph.nodes[u]["y"]),
                (graph.nodes[v]["x"], graph.nodes[v]["y"]),
            ])
        attrs = {
            column: row[column]
            for column in edge_df.columns
            if column not in {"u", "v", "key", "geometry"}
        }
        attrs["geometry"] = geometry
        attrs["length"] = float(row.get("length", geometry.length))
        attrs["betweenness"] = float(row.get("betweenness", 0.0))
        graph.add_edge(u, v, **attrs)
    return graph


def high_contrast_nodes_for_edges(gdf_nodes, gdf_edges):
    if gdf_edges.empty:
        return set()
    edge_df = gdf_edges.reset_index()
    if "betweenness" not in edge_df.columns:
        return set()
    betweenness = pd.to_numeric(edge_df["betweenness"], errors="coerce").fillna(0.0)
    q_high = betweenness.quantile(0.75)
    q_low = betweenness.quantile(0.25)
    high_contrast = set()
    for node_id in gdf_nodes.index:
        incident_edges = edge_df[
            edge_df[["u", "v"]].apply(lambda row: node_id in (row["u"], row["v"]), axis=1)
        ]
        if len(incident_edges) <= 1:
            continue
        scores = pd.to_numeric(incident_edges["betweenness"], errors="coerce").fillna(0.0).tolist()
        if any(score >= q_high for score in scores) and any(score <= q_low for score in scores):
            high_contrast.add(node_id)
    return high_contrast


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
    ] + [
        (f"entrance:{index}:{path.wkb_hex}", "entrance", index, path)
        for index, path in enumerate(st.session_state.entrance_barrier_paths)
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
            else (
                f"Desire path {option_lookup[option_id][2] + 1}"
                if option_lookup[option_id][1] == "desire"
                else f"Entrance/barrier path {option_lookup[option_id][2] + 1}"
            )
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
            target_paths = {
                "street": st.session_state.street_plan_paths,
                "desire": st.session_state.desire_paths,
                "entrance": st.session_state.entrance_barrier_paths,
            }[path_type]
            target_paths[path_index:path_index + 1] = [first_path, second_path]
            st.session_state.desire_path_analysis = []
            st.session_state.entrance_barrier_analysis = []
            st.session_state.path_map_revision += 1
            st.rerun()
    with remove_col:
        if st.button("Remove selected path", key=f"{key_prefix}_remove_button"):
            selected_id = st.session_state[selection_key]
            _, path_type, path_index, _ = option_lookup[selected_id]
            target_paths = {
                "street": st.session_state.street_plan_paths,
                "desire": st.session_state.desire_paths,
                "entrance": st.session_state.entrance_barrier_paths,
            }[path_type]
            target_paths.pop(path_index)
            st.session_state.desire_path_analysis = []
            st.session_state.entrance_barrier_analysis = []
            st.session_state.path_map_revision += 1
            st.rerun()
    return selected_path, cut_position


def analyze_desire_paths(graph, paths, edge_centrality, path_type="Desire path"):
    """Snap each drawn path to the network and summarize its shortest route."""
    def edge_score(u, v, edge_key):
        return edge_centrality.get(
            (u, v, edge_key),
            edge_centrality.get(
                (v, u, edge_key),
                edge_centrality.get(
                    (u, v),
                    edge_centrality.get(
                        (v, u),
                        float(graph.get_edge_data(u, v)[edge_key].get("betweenness", 0.0))
                        if graph.is_multigraph() and edge_key in graph.get_edge_data(u, v, default={}) else 0.0,
                    ),
                ),
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
                "path_type": path_type, "path_id": index + 1,
                "drawn_geometry": mapping(drawn_path),
                "drawn_length_m": round(path_length_m(drawn_path), 2),
                "start_node": start, "end_node": end,
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
            if not graph.is_multigraph():
                # Simple graphs (e.g. imported plans) return one attribute dict, not {key: attributes}.
                edge_data = {0: edge_data}
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
            "path_type": path_type,
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


uploaded_json = st.file_uploader("Import Urban Spatial Interaction JSON", type=["json"])
import_signature = (
    (uploaded_json.name, uploaded_json.size) if uploaded_json is not None else None
)
if uploaded_json is None:
    st.session_state.pop("urban_import_signature", None)
# Import once per upload; the uploader keeps its file across reruns and would
# otherwise overwrite the loaded state (and edits) on every interaction.
elif st.session_state.get("urban_import_signature") != import_signature:
    try:
        imported = json.loads(uploaded_json.getvalue())
        imported_edges = gpd.GeoDataFrame.from_features(imported["plan"]["edges"]["features"])
        imported_nodes = gpd.GeoDataFrame.from_features(imported["plan"]["nodes"]["features"])
        if "u" in imported_edges.columns and "v" in imported_edges.columns and "key" in imported_edges.columns:
            imported_edges.set_index(["u", "v", "key"], inplace=True)
        if "node_id" in imported_nodes.columns:
            imported_nodes.set_index("node_id", inplace=True)
        if "betweenness" in imported_edges.columns:
            imported_edges["betweenness"] = pd.to_numeric(imported_edges["betweenness"], errors="coerce").fillna(0.0)
        else:
            imported_edges["betweenness"] = 0.0
        imported_graph = build_graph_from_gdfs(imported_nodes, imported_edges)
        edge_centrality = {}
        if imported_graph.number_of_edges() > 0:
            edge_centrality = nx.edge_betweenness_centrality(imported_graph)
            nx.set_edge_attributes(imported_graph, edge_centrality, "betweenness")
            for u, v, attrs in imported_graph.edges(data=True):
                if "betweenness" in imported_edges.columns:
                    imported_edges.loc[(imported_edges.index.get_level_values(0) == u) & (imported_edges.index.get_level_values(1) == v), "betweenness"] = attrs.get("betweenness", 0.0)
        imported_edges = assign_edge_colors(imported_edges)
        plan = {
            "gdf_edges": imported_edges, "gdf_nodes": imported_nodes,
            "center_lat": imported["plan"]["center_lat"], "center_lon": imported["plan"]["center_lon"],
            "search_query": imported["plan"].get("search_query", "Imported plan"), "G_undirected": imported_graph,
            "analysis_radius_m": imported["plan"].get("analysis_radius_m", radius),
        }
        st.session_state.plan_data = plan
        st.session_state.graph_data = {
            "G_undirected": imported_graph,
            "gdf_nodes": imported_nodes,
            "gdf_edges": imported_edges,
            "center_lat": imported["plan"]["center_lat"],
            "center_lon": imported["plan"]["center_lon"],
            "high_contrast_nodes": high_contrast_nodes_for_edges(imported_nodes, imported_edges),
            "search_query": imported["plan"].get("search_query", "Imported plan"),
            "analysis_radius_m": imported["plan"].get("analysis_radius_m", radius),
        }
        imported_street_paths = imported.get("street_plan_paths")
        st.session_state.street_plan_paths = (
            [shape(feature["geometry"]) for feature in imported_street_paths]
            if imported_street_paths is not None
            else street_paths_from_edges(imported_edges)
        )
        st.session_state.desire_paths = [shape(feature["geometry"]) for feature in imported.get("desire_paths", [])]
        st.session_state.desire_path_analysis = imported.get("desire_path_analysis", [])
        imported_entrance_paths = imported.get("entrance_barrier_paths")
        if imported_entrance_paths is not None:
            st.session_state.entrance_barrier_paths = [
                shape(feature["geometry"]) for feature in imported_entrance_paths
            ]
        else:
            entrance_nodes = entrance_barrier_nodes_within_radius(
                load_entrance_barrier_data(),
                imported["plan"]["center_lat"],
                imported["plan"]["center_lon"],
                imported["plan"].get("analysis_radius_m", radius),
            )
            st.session_state.entrance_barrier_paths = entrance_barrier_paths(
                entrance_nodes, st.session_state.street_plan_paths
            )
        imported_entrance_analysis = imported.get("entrance_barrier_analysis")
        st.session_state.entrance_barrier_analysis = (
            imported_entrance_analysis
            if imported_entrance_analysis is not None
            else analyze_desire_paths(
                imported_graph,
                st.session_state.entrance_barrier_paths,
                edge_centrality,
                path_type="Entrance/barrier path",
            )
        )
        for result in st.session_state.desire_path_analysis:
            result.setdefault("drawn_length_m", 0.0)
            result.setdefault("route_length_m", 0.0)
            result.setdefault("centrality", result.get("mean_betweenness", 0.0))
            result.setdefault("mean_betweenness", 0.0)
            result.setdefault("max_betweenness", 0.0)
            result.setdefault("network_edge_count", 0)
        st.session_state.selected_node = None
        st.session_state.path_map_revision += 1
        st.session_state.urban_import_signature = import_signature
        st.success("Urban plan imported. The map and pathways are ready to view.")
    except Exception as error:
        st.error(f"Could not import this JSON package: {error!r}")


if st.button("Load Street Plan"):
    with st.spinner("Loading street network for preview..."):
        try:
            st.session_state.plan_data = load_plan(search_query, radius, network_type)
            st.session_state.graph_data = None
            st.session_state.street_plan_paths = street_paths_from_edges(
                st.session_state.plan_data["gdf_edges"]
            )
            entrance_nodes = entrance_barrier_nodes_within_radius(
                load_entrance_barrier_data(),
                st.session_state.plan_data["center_lat"],
                st.session_state.plan_data["center_lon"],
                radius,
            )
            st.session_state.entrance_barrier_paths = entrance_barrier_paths(
                entrance_nodes, st.session_state.street_plan_paths
            )
            st.session_state.desire_paths = []
            st.session_state.desire_path_analysis = []
            st.session_state.entrance_barrier_analysis = []
        except Exception as e:
            st.error(f"Error loading street plan: {e}")

# -----------------------------------------------------------------------------
# 4. Plan Preview and Interactive Rendering
# -----------------------------------------------------------------------------
if st.session_state.plan_data is not None and st.session_state.graph_data is None:
    plan = st.session_state.plan_data
    st.subheader("Street Plan Preview")
    st.caption("Entrance and barrier connectors are generated from nearby tagged nodes. Draw desire paths or edit any pathway before analysis.")
    selected_preview_path, preview_cut_position = render_path_editor("preview")
    preview_map = dark_map([plan["center_lat"], plan["center_lon"]])
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
    for index, path in enumerate(st.session_state.entrance_barrier_paths):
        folium.PolyLine(
            locations=[[point[1], point[0]] for point in path.coords],
            color="#B388FF", weight=4, opacity=0.9,
            tooltip=f"Entrance/barrier path {index + 1} ({path_length_m(path):.1f} m)"
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
    st.info(
        f"{len(st.session_state.desire_paths)} desire path(s) and "
        f"{len(st.session_state.entrance_barrier_paths)} entrance/barrier path(s) staged for analysis."
    )

if st.session_state.plan_data is not None and st.button("Run Axial Analysis"):
    with st.spinner("Calculating spatial depth and centrality..."):
        try:
            plan = st.session_state.plan_data
            G_undirected = plan["G_undirected"]
            if G_undirected is None:
                raise ValueError("Imported display data cannot be rerun; load the street plan to calculate new metrics.")
            edge_centrality = nx.edge_betweenness_centrality(G_undirected)
            nx.set_edge_attributes(G_undirected, edge_centrality, "betweenness")

            import osmnx as ox

            gdf_nodes, gdf_edges = ox.convert.graph_to_gdfs(G_undirected)
            gdf_edges['street_name'] = gdf_edges['name'].apply(clean_name)

            gdf_edges = assign_edge_colors(gdf_edges)

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
                "analysis_radius_m": plan["analysis_radius_m"],
                "high_contrast_nodes": high_contrast_nodes,
                "search_query": plan["search_query"]
            }
            st.session_state.desire_path_analysis = analyze_desire_paths(
                G_undirected, st.session_state.desire_paths, edge_centrality,
                path_type="Desire path",
            )
            st.session_state.entrance_barrier_analysis = analyze_desire_paths(
                G_undirected, st.session_state.entrance_barrier_paths, edge_centrality,
                path_type="Entrance/barrier path",
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
    m = dark_map([data["center_lat"], data["center_lon"]])
    centrality_min = float(gdf_edges["betweenness"].min())
    centrality_max = float(gdf_edges["betweenness"].max())
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
    # All dots go into one layer that is added after every path so they render on top.
    node_layer = folium.FeatureGroup(name="Nodes")
    for node_id, row in gdf_nodes.iterrows():
        lat, lon = row.geometry.y, row.geometry.x
        is_contrast = node_id in high_contrast_nodes
        is_selected = (node_id == selected_node)
        
        if is_selected:
            folium.CircleMarker(
                location=[lat, lon],
                radius=18,
                color='#00FFFF',
                weight=3,
                fill=True,
                fill_color='#00FFFF',
                fill_opacity=0.9,
                popup=f"Selected Node"
            ).add_to(node_layer)
        elif is_contrast and show_contrast_nodes:
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color='#FF0055',
                fill=True,
                fill_color='#FF0055',
                fill_opacity=0.4,
                popup="High-Contrast Intersection"
            ).add_to(node_layer)
        elif not is_contrast and show_standard_nodes:
            folium.CircleMarker(
                location=[lat, lon],
                radius=2.5,
                color='#FFFFFF',
                fill=True,
                fill_color='#FFFFFF',
                fill_opacity=0.4,
                tooltip=f"Intersection"
            ).add_to(node_layer)

    # Keep user-drawn and entrance/barrier routes above the network layers.
    visible_intersections = set()
    # Imported or edited sessions can lack results for some paths; compute them from the
    # graph's stored edge betweenness so every route is coloured on the centrality scale.
    if G_undirected is not None:
        if len(st.session_state.desire_path_analysis) != len(st.session_state.desire_paths):
            st.session_state.desire_path_analysis = analyze_desire_paths(
                G_undirected, st.session_state.desire_paths, {}, path_type="Desire path"
            )
        if len(st.session_state.entrance_barrier_analysis) != len(st.session_state.entrance_barrier_paths):
            st.session_state.entrance_barrier_analysis = analyze_desire_paths(
                G_undirected, st.session_state.entrance_barrier_paths, {},
                path_type="Entrance/barrier path",
            )
    path_groups = [
        ("Desire path", st.session_state.desire_paths, st.session_state.desire_path_analysis),
        (
            "Entrance/barrier path",
            st.session_state.entrance_barrier_paths,
            st.session_state.entrance_barrier_analysis,
        ),
    ]
    for path_label, paths, results in path_groups:
        result_by_path_id = {result["path_id"]: result for result in results}
        fallback_color = "#FFD166" if path_label == "Desire path" else "#B388FF"
        for path_id, drawn_path in enumerate(paths, start=1):
            result = result_by_path_id.get(path_id)
            centrality = (
                result.get("centrality", result.get("mean_betweenness", 0.0))
                if result else None
            )
            route_color = (
                centrality_color(centrality, centrality_min, centrality_max)
                if centrality is not None else fallback_color
            )
            route_tooltip = f"{path_label} {path_id}"
            if result:
                route_tooltip += (
                    f" | Centrality: {centrality:.6f}"
                    f" | Mean: {result.get('mean_betweenness', centrality):.6f}"
                    f" | Max: {result.get('max_betweenness', centrality):.6f}"
                    f" | Connector: {result.get('drawn_length_m', 0.0):.1f} m"
                    f" | Network route: {result.get('route_length_m', 0.0):.1f} m"
                )
            folium.PolyLine(
                locations=[[point[1], point[0]] for point in drawn_path.coords],
                color=route_color, weight=6, opacity=0.95,
                tooltip=route_tooltip,
            ).add_to(m)
            if result and result.get("route_geometry") is not None:
                route_shape = shape(result["route_geometry"])
                route_parts = route_shape.geoms if hasattr(route_shape, "geoms") else [route_shape]
                for route_part in route_parts:
                    folium.PolyLine(
                        locations=[[point[1], point[0]] for point in route_part.coords],
                        color=route_color, weight=8, opacity=1.0,
                        tooltip=route_tooltip,
                    ).add_to(m)
            if result:
                for node_label in ("start_node", "end_node"):
                    node_id = result.get(node_label)
                    if node_id in gdf_nodes.index:
                        node = gdf_nodes.loc[node_id].geometry
                        folium.CircleMarker(
                            location=[node.y, node.x], radius=7,
                            color="#FFFFFF", weight=2, fill=True,
                            fill_color="#FF3366", fill_opacity=1.0,
                            tooltip=f"{path_label} {path_id} {node_label.replace('_', ' ')}",
                        ).add_to(node_layer)
            for street_path in st.session_state.street_plan_paths:
                for point in geometry_intersection_points(drawn_path, street_path):
                    point_key = (round(point.x, 7), round(point.y, 7))
                    if point_key in visible_intersections:
                        continue
                    visible_intersections.add(point_key)
                    folium.CircleMarker(
                        location=[point.y, point.x], radius=5,
                        color="#FFFFFF", weight=2, fill=True,
                        fill_color=route_color, fill_opacity=1.0,
                        tooltip=(
                            f"{path_label} {path_id} street intersection"
                            + (f" | Centrality: {centrality:.6f}" if centrality is not None else "")
                        ),
                    ).add_to(node_layer)
            # Path end points (e.g. building entrances) are intersections too: white, or
            # red when the network node they attach to is a high-contrast intersection.
            for end_label, end_coord in (("start_node", drawn_path.coords[0]), ("end_node", drawn_path.coords[-1])):
                end_key = (round(end_coord[0], 7), round(end_coord[1], 7))
                if end_key in visible_intersections:
                    continue
                visible_intersections.add(end_key)
                end_is_contrast = bool(result) and result.get(end_label) in high_contrast_nodes
                end_color = "#FF0055" if end_is_contrast else "#FFFFFF"
                folium.CircleMarker(
                    location=[end_coord[1], end_coord[0]], radius=5 if end_is_contrast else 4,
                    color=end_color, weight=2, fill=True,
                    fill_color=end_color, fill_opacity=0.9,
                    tooltip=f"{path_label} {path_id} end point"
                    + (" | High-Contrast Intersection" if end_is_contrast else ""),
                ).add_to(node_layer)
    add_cut_preview(m, selected_preview_path, preview_cut_position)
    node_layer.add_to(m)

    # Legend Overlay for Map
    legend_html = '''
    <div style="position: fixed; bottom: 30px; left: 30px; width: 250px; 
                background-color: rgba(17,17,17,0.9); z-index:9999; font-size:12px; color: white;
                padding: 12px; border-radius: 6px; font-family: sans-serif; border: 1px solid #444;">
        <b>Urban Spatial Interaction</b><br>
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

    st.subheader("1. Interactive Urban Spatial Interaction Map")
    st_folium(m, width=1000, height=520, key="main_map", returned_objects=[])
    path_analysis = (
        st.session_state.desire_path_analysis
        + st.session_state.entrance_barrier_analysis
    )
    if path_analysis:
        st.subheader("Pathway Centrality Results")
        st.dataframe(path_analysis, width="stretch")

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
            
            street_name = clean_name(d.get('street_name', d.get('name')))
            edge_hover.extend([
                f"Street: {street_name}" if not street_name.startswith("Unnamed") else "Street segment",
                f"Street: {street_name}" if not street_name.startswith("Unnamed") else "Street segment",
                None,
            ])

            # Only named streets receive visible midpoint labels.
            if not street_name.startswith("Unnamed"):
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
        streets = sorted({
            clean_name(d.get('street_name', d.get('name')))
            for _, _, d in incident
            if not clean_name(d.get('street_name', d.get('name'))).startswith("Unnamed")
        })
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

    event = st.plotly_chart(fig_jg, width="stretch", on_select="rerun", key="jgraph_chart")
    
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
            "analysis_radius_m": data.get("analysis_radius_m", radius),
            "edges": geojson_feature_collection(export_edges),
            "nodes": geojson_feature_collection(export_nodes),
        },
        "desire_paths": desire_path_features(),
        "street_plan_paths": [{
            "type": "Feature", "properties": {"path_id": index + 1},
            "geometry": mapping(path),
        } for index, path in enumerate(st.session_state.street_plan_paths)],
        "entrance_barrier_paths": [{
            "type": "Feature",
            "properties": {"path_id": index + 1, "source": "entrance_or_barrier"},
            "geometry": mapping(path),
        } for index, path in enumerate(st.session_state.entrance_barrier_paths)],
        "desire_path_analysis": st.session_state.desire_path_analysis,
        "entrance_barrier_analysis": st.session_state.entrance_barrier_analysis,
        "analysis": {
            "metric": "edge_betweenness_centrality",
            "edge_count": len(gdf_edges),
            "desire_path_count": len(st.session_state.desire_paths),
            "entrance_barrier_path_count": len(st.session_state.entrance_barrier_paths),
        },
    }
    st.sidebar.download_button(
        label="Download Complete Analysis (.json)",
        data=json.dumps(json_package, indent=2),
        file_name="urban_spatial_interaction_analysis.json",
        mime="application/json",
        key="urban_spatial_interaction_json_export",
    )
    
    gdf_export = gdf_edges[['street_name', 'betweenness', 'length', 'geometry']]
    geojson_str = json.dumps(geojson_feature_collection(gdf_export))
    st.sidebar.download_button(
        label="Download GIS Vector (.geojson)",
        data=geojson_str,
        file_name="urban_spatial_interaction.geojson",
        mime="application/geo+json",
        key="urban_spatial_interaction_geojson_export",
    )
    
    fig_export, ax_export = plt.subplots(figsize=(8, 8), dpi=300)
    gdf_edges.plot(ax=ax_export, column='betweenness', cmap='turbo', linewidth=2, zorder=1)
    for paths, results in (
        (st.session_state.desire_paths, st.session_state.desire_path_analysis),
        (
            st.session_state.entrance_barrier_paths,
            st.session_state.entrance_barrier_analysis,
        ),
    ):
        for result in results:
            centrality = result.get("centrality", result.get("mean_betweenness", 0.0))
            route_color = centrality_color(centrality, centrality_min, centrality_max)
            drawn_geometry = paths[result["path_id"] - 1]
            gpd.GeoSeries([drawn_geometry], crs=gdf_edges.crs).plot(
                ax=ax_export, color=route_color, linewidth=3.5, zorder=4
            )
            route_geometry = result.get("route_geometry")
            if route_geometry is not None:
                route_shape = shape(route_geometry)
                gpd.GeoSeries([route_shape], crs=gdf_edges.crs).plot(
                    ax=ax_export, color=route_color, linewidth=5, alpha=0.9, zorder=3
                )
    if show_standard_nodes:
        gdf_nodes[~gdf_nodes.index.isin(high_contrast_nodes)].plot(ax=ax_export, color='white', markersize=3, alpha=0.8, zorder=2)
    if show_contrast_nodes:
        gdf_nodes[gdf_nodes.index.isin(high_contrast_nodes)].plot(ax=ax_export, color='#FF0055', markersize=25, alpha=0.8, zorder=3)
    
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
        mime="image/png",
        key="urban_spatial_interaction_png_export",
    )

tab_interaction.__exit__(None, None, None)


# -----------------------------------------------------------------------------
# Tab 2: Gravity model of pedestrian density along the Tab 1 desire paths
# -----------------------------------------------------------------------------
GEOJSON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_mass_points(source, weight_field):
    """Read point features and return a GeoDataFrame with a numeric 'mass' column."""
    gdf = gpd.read_file(source)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gdf = gdf.to_crs(4326)
    gdf = gdf.explode(index_parts=False, ignore_index=True)
    gdf = gdf[gdf.geometry.notna() & (gdf.geometry.geom_type == "Point")].copy()
    column = next((c for c in gdf.columns if c.upper() == weight_field.upper()), None)
    if column is None:
        raise KeyError(f"Field '{weight_field}' not found. Available: {', '.join(map(str, gdf.columns))}")
    gdf["mass"] = pd.to_numeric(gdf[column], errors="coerce")
    gdf = gdf[gdf["mass"].notna() & (gdf["mass"] > 0)].reset_index(drop=True)
    return gdf


def gravity_path_density(mass_gdf, paths, beta, spacing_m, people_per_m2, min_distance_m):
    """Production-constrained gravity model.

    Each mass i emits P_i = MAX_GFA_i * people_per_m2 people. They are distributed over all
    sample points s on the paths in proportion to d_is^-beta, so
    people_s = sum_i P_i * d_is^-beta / sum_s' d_is'^-beta.
    """
    utm = mass_gdf.estimate_utm_crs()
    projected_masses = mass_gdf.to_crs(utm)
    masses_xy = np.column_stack([
        projected_masses.geometry.x.to_numpy(), projected_masses.geometry.y.to_numpy()
    ])
    projected_paths = gpd.GeoSeries(paths, crs=4326).to_crs(utm)

    rows = []
    for path_id, (path, projected) in enumerate(zip(paths, projected_paths), start=1):
        length = float(projected.length)
        if length <= 0:
            continue
        count = max(1, int(np.ceil(length / spacing_m)))
        step = length / count
        for k in range(count):
            fraction = (k + 0.5) / count
            lonlat = path.interpolate(fraction, normalized=True)
            xy = projected.interpolate(fraction, normalized=True)
            rows.append({
                "path_id": path_id, "offset_m": fraction * length, "segment_length_m": step,
                "x": xy.x, "y": xy.y, "lon": lonlat.x, "lat": lonlat.y,
            })
    samples = pd.DataFrame(rows)
    if samples.empty:
        return samples, pd.DataFrame()

    sample_xy = samples[["x", "y"]].to_numpy()
    generated = mass_gdf["mass"].to_numpy() * people_per_m2
    allocated_people = np.zeros(len(samples), dtype=float)
    for mass_xy, people in zip(masses_xy, generated):
        distance = np.linalg.norm(sample_xy - mass_xy, axis=1)
        attraction = np.maximum(distance, min_distance_m) ** (-beta)
        attraction_total = attraction.sum()
        if attraction_total > 0:
            allocated_people += people * attraction / attraction_total
    samples["people"] = allocated_people
    samples["density_people_per_m"] = samples["people"] / samples["segment_length_m"]

    summary = samples.groupby("path_id").agg(
        length_m=("segment_length_m", "sum"),
        total_people=("people", "sum"),
        peak_density_people_per_m=("density_people_per_m", "max"),
    ).reset_index()
    summary["mean_density_people_per_m"] = summary["total_people"] / summary["length_m"]
    summary["length_m"] = summary["length_m"].round(1)
    return samples, summary


EVENT_COLUMNS = ["label", "lat", "lon", "radius_m", "factor"]


def empty_events():
    return pd.DataFrame({
        "label": pd.Series(dtype="object"), "lat": pd.Series(dtype="float"),
        "lon": pd.Series(dtype="float"), "radius_m": pd.Series(dtype="float"),
        "factor": pd.Series(dtype="float"),
    })


def clean_events(events):
    events = events.dropna(subset=["lat", "lon", "radius_m", "factor"]).copy()
    events = events[(events["radius_m"] > 0) & (events["factor"] > 0)]
    events["label"] = events["label"].fillna("").astype(str)
    events.loc[events["label"] == "", "label"] = [str(i + 1) for i in range(int((events["label"] == "").sum()))]
    return events.reset_index(drop=True)


def apply_event_factors(mass_gdf, events):
    """Multiply each building's GFA by the factor of every event whose radius covers it."""
    adjusted = mass_gdf.copy()
    adjusted["base_mass"] = adjusted["mass"]
    adjusted["event_factor"] = 1.0
    if not events.empty:
        utm = adjusted.estimate_utm_crs()
        mass_xy = adjusted.to_crs(utm).geometry
        event_xy = gpd.GeoSeries(
            gpd.points_from_xy(events["lon"], events["lat"]), crs=4326
        ).to_crs(utm)
        for (_, event), event_point in zip(events.iterrows(), event_xy):
            inside = mass_xy.distance(event_point).to_numpy() <= float(event["radius_m"])
            adjusted.loc[inside, "event_factor"] *= float(event["factor"])
    adjusted["mass"] = adjusted["base_mass"] * adjusted["event_factor"]
    return adjusted


with tab_gravity:
    st.subheader("Gravity Model: People Density Along All Street Paths")
    st.markdown(
        "Each point is a **mass** weighted by `MAX_GFA`. Its people are distributed over every street "
        "path from the Tab 1 network by the gravity model (spatial interaction theory):"
    )
    st.latex(r"T_{is} = P_i \cdot \frac{d_{is}^{-\beta}}{\sum_{s'} d_{is'}^{-\beta}}, \qquad P_i = \text{MAX\_GFA}_i \cdot \rho")
    st.caption(
        f"Place your GeoJSON in `{GEOJSON_DIR}` (it will appear in the list below), or upload it directly."
    )

    os.makedirs(GEOJSON_DIR, exist_ok=True)
    available_files = sorted(
        f for f in os.listdir(GEOJSON_DIR) if f.lower().endswith((".geojson", ".json"))
    )
    source_col, upload_col = st.columns(2)
    with source_col:
        chosen_file = st.selectbox("GeoJSON in data folder", ["-- none --"] + available_files)
    with upload_col:
        uploaded_masses = st.file_uploader("...or upload GeoJSON", type=["geojson", "json"], key="gravity_geojson")

    param_cols = st.columns(4)
    beta = param_cols[0].number_input("Distance decay β", 0.1, 5.0, 2.0, 0.1)
    people_per_m2 = param_cols[1].number_input("People per m² GFA (ρ)", 0.001, 1.0, 0.05, 0.005, format="%.3f")
    spacing_m = param_cols[2].number_input("Sample spacing (m)", 1.0, 100.0, 10.0, 1.0)
    min_distance_m = param_cols[3].number_input("Min. distance (m)", 1.0, 100.0, 10.0, 1.0)

    mass_source = uploaded_masses if uploaded_masses is not None else (
        os.path.join(GEOJSON_DIR, chosen_file) if chosen_file != "-- none --" else None
    )
    mass_gdf = None
    if mass_source is not None:
        try:
            mass_gdf = load_mass_points(mass_source, "MAX_GFA")
            if mass_gdf.empty:
                st.warning("No point features with a positive MAX_GFA were found.")
                mass_gdf = None
        except Exception as error:
            st.error(f"Could not read GeoJSON: {error}")

    graph_t2 = st.session_state.graph_data
    if "gravity_events_base" not in st.session_state:
        st.session_state.gravity_events_base = empty_events()
        st.session_state.gravity_events_applied = empty_events()
        st.session_state.gravity_events_rev = 0
    event_df = st.session_state.gravity_events_applied
    clicked_point = None
    if mass_gdf is not None:
        mass_gdf = apply_event_factors(mass_gdf, event_df)
    street_paths_t2 = list(st.session_state.street_plan_paths)
    if not street_paths_t2 and graph_t2 is not None:
        street_paths_t2 = street_paths_from_edges(graph_t2["gdf_edges"])
    if mass_gdf is None:
        st.info("Select or upload a GeoJSON of points with a `MAX_GFA` property.")
    if not street_paths_t2:
        st.info("Load a street plan in the Urban Spatial Interaction tab first.")
    if graph_t2 is None:
        st.info("Run the axial analysis in the Urban Spatial Interaction tab to show the red high-contrast nodes here.")

    samples = summary = None
    if mass_gdf is not None and street_paths_t2:
        samples, summary = gravity_path_density(
            mass_gdf, street_paths_t2, beta, spacing_m, people_per_m2, min_distance_m
        )

    if mass_gdf is not None or street_paths_t2 or graph_t2 is not None:
        if graph_t2 is not None:
            center = [graph_t2["center_lat"], graph_t2["center_lon"]]
        elif mass_gdf is not None:
            center = [mass_gdf.geometry.y.mean(), mass_gdf.geometry.x.mean()]
        else:
            center = [street_paths_t2[0].coords[0][1], street_paths_t2[0].coords[0][0]]
        gravity_map = dark_map(center)

        if graph_t2 is not None:
            for _, street in graph_t2["gdf_edges"].iterrows():
                if street.geometry.geom_type == "LineString":
                    folium.PolyLine(
                        locations=[[p[1], p[0]] for p in street.geometry.coords],
                        color="#35B7FF", weight=2, opacity=0.35,
                    ).add_to(gravity_map)

        if samples is not None and not samples.empty:
            d_min = float(samples["density_people_per_m"].min())
            d_max = float(samples["density_people_per_m"].max())
            for path_id, group in samples.groupby("path_id"):
                points = group.sort_values("offset_m")
                for (_, a), (_, b) in zip(points.iterrows(), points.iloc[1:].iterrows()):
                    folium.PolyLine(
                        locations=[[a["lat"], a["lon"]], [b["lat"], b["lon"]]],
                        color=centrality_color(a["density_people_per_m"], d_min, d_max),
                        weight=7, opacity=0.95,
                        tooltip=f"Path {path_id} | {a['density_people_per_m']:.3f} people/m",
                    ).add_to(gravity_map)
        else:
            for index, path in enumerate(street_paths_t2):
                folium.PolyLine(
                    locations=[[p[1], p[0]] for p in path.coords],
                    color="#FFD166", weight=5, opacity=0.95, tooltip=f"Street path {index + 1}",
                ).add_to(gravity_map)

        # Dots are added last so they sit above every path.
        if graph_t2 is not None:
            for node_id in graph_t2["high_contrast_nodes"]:
                if node_id in graph_t2["gdf_nodes"].index:
                    point = graph_t2["gdf_nodes"].loc[node_id].geometry
                    folium.CircleMarker(
                        location=[point.y, point.x], radius=6, color="#FF0055",
                        fill=True, fill_color="#FF0055", fill_opacity=0.4,
                        popup="High-Contrast Intersection",
                    ).add_to(gravity_map)

        if mass_gdf is not None:
            max_mass = float(mass_gdf["mass"].max())
            min_mass = float(mass_gdf["mass"].min())
            for _, mass_row in mass_gdf.iterrows():
                mass_color = mcolors.to_hex(plt.get_cmap("RdYlGn_r")(
                    (mass_row["mass"] - min_mass) / (max_mass - min_mass + 1e-9)
                ))
                folium.CircleMarker(
                    location=[mass_row.geometry.y, mass_row.geometry.x],
                    radius=3 + 9 * (mass_row["mass"] / max_mass) ** 0.5,
                    color=mass_color, weight=1, fill=True, fill_color=mass_color, fill_opacity=0.7,
                    tooltip=(
                        f"GFA: {mass_row['base_mass']:,.0f} m² × event factor {mass_row['event_factor']:.2f}"
                        f" = {mass_row['mass']:,.0f} m²"
                    ),
                ).add_to(gravity_map)

        for _, event in event_df.iterrows():
            folium.Circle(
                location=[event["lat"], event["lon"]], radius=float(event["radius_m"]),
                color="#B266FF", weight=2, fill=True, fill_opacity=0.08,
                tooltip=f"Event {event['label']} | ×{event['factor']:.2f} within {event['radius_m']:.0f} m",
            ).add_to(gravity_map)
            folium.CircleMarker(
                location=[event["lat"], event["lon"]], radius=7, color="#FFFFFF", weight=2,
                fill=True, fill_color="#B266FF", fill_opacity=1.0,
                tooltip=f"Event {event['label']} | ×{event['factor']:.2f}",
            ).add_to(gravity_map)

        map_state = st_folium(
            gravity_map, width=1000, height=520, key="gravity_map", returned_objects=["last_clicked"]
        )
        clicked_point = (map_state or {}).get("last_clicked")
        st.caption(
            "Mass circles: buildings (size ∝ √adjusted GFA; green = lighter, red = heavier). Red dots: high-contrast intersections from Tab 1. "
            "Purple: event points and their radius. Path colour: people density (blue = low, red = high)."
        )

    if summary is not None and not summary.empty:
        st.subheader("Density Along All Street Paths")
        st.dataframe(summary, width="stretch")
        profile = go.Figure()
        for path_id, group in samples.groupby("path_id"):
            profile.add_trace(go.Scatter(
                x=group["offset_m"], y=group["density_people_per_m"],
                mode="lines", name=f"Path {path_id}",
            ))
        profile.update_layout(
            template="plotly_dark", xaxis_title="Distance along path (m)",
            yaxis_title="Density (people / m)", height=380,
        )
        st.plotly_chart(profile, width="stretch")
        st.download_button(
            "Download sample densities (.csv)",
            data=samples.drop(columns=["x", "y"]).to_csv(index=False),
            file_name="gravity_path_density.csv", mime="text/csv", key="gravity_csv_export",
        )

    st.write("---")
    st.subheader("Step 2: Event Factors on GFA")
    st.caption(
        "Add one or more event points. Every building within an event's radius has its GFA multiplied by "
        "that event's factor (overlapping events multiply). Edit the table, then click the button to recompute."
    )
    if clicked_point:
        st.write(f"Last map click: {clicked_point['lat']:.6f}, {clicked_point['lng']:.6f}")
        add_cols = st.columns(3)
        new_radius = add_cols[0].number_input("Radius (m)", 5.0, 5000.0, 100.0, 5.0, key="new_event_radius")
        new_factor = add_cols[1].number_input("Factor", 0.1, 50.0, 2.0, 0.1, key="new_event_factor")
        if add_cols[2].button("Add event at clicked point", key="add_clicked_event"):
            new_row = pd.DataFrame([{
                "label": f"Event {len(st.session_state.gravity_events_base) + 1}", "lat": clicked_point["lat"],
                "lon": clicked_point["lng"], "radius_m": new_radius, "factor": new_factor,
            }])
            st.session_state.gravity_events_base = pd.concat(
                [st.session_state.gravity_events_base, new_row], ignore_index=True
            )
            st.session_state.gravity_events_rev += 1
            st.rerun()
    else:
        st.caption("Tip: click the map above to pick a location, then add it as an event here.")

    edited_events = st.data_editor(
        st.session_state.gravity_events_base,
        num_rows="dynamic", width="stretch",
        key=f"gravity_events_editor_{st.session_state.gravity_events_rev}",
        column_config={
            "label": st.column_config.TextColumn("Event"),
            "lat": st.column_config.NumberColumn("Latitude", format="%.6f"),
            "lon": st.column_config.NumberColumn("Longitude", format="%.6f"),
            "radius_m": st.column_config.NumberColumn("Radius (m)", min_value=1.0, default=100.0),
            "factor": st.column_config.NumberColumn("GFA factor", min_value=0.01, default=2.0),
        },
    )
    if st.button("Apply events and recompute", type="primary", key="apply_gravity_events"):
        st.session_state.gravity_events_applied = clean_events(edited_events)
        st.rerun()
    if clean_events(edited_events).to_dict("records") != event_df.to_dict("records"):
        st.warning("Event table has changes that are not applied yet. Click the button to recompute.")
    if mass_gdf is not None and not event_df.empty:
        affected = int((mass_gdf["event_factor"] != 1.0).sum())
        st.info(
            f"{affected} of {len(mass_gdf)} buildings affected. Total GFA "
            f"{mass_gdf['base_mass'].sum():,.0f} → {mass_gdf['mass'].sum():,.0f} m²."
        )
