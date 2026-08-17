import streamlit as st
import osmnx as ox
import networkx as nx
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

st.set_page_config(page_title="Urban Space Syntax", layout="wide")
st.title("Singapore Urban Space Syntax Analysis")

# 1. User Input Controls
col1, col2 = st.columns(2)
with col1:
    search_query = st.text_input("Location in Singapore", "Tiong Bahru, Singapore")
with col2:
    radius = st.slider("Radius (meters)", min_value=300, max_value=2000, value=600, step=100)

network_type = st.selectbox("Network Type", ["walk", "drive", "all"])

# Initialize session state for map storage
if "axial_map" not in st.session_state:
    st.session_state.axial_map = None

if st.button("Run Axial Analysis"):
    with st.spinner("Fetching street network and calculating centrality..."):
        try:
            # 2. Geocode & Fetch Street Graph from OpenStreetMap
            gdf_place = ox.geocode_to_gdf(search_query)
            center_lat = gdf_place.geometry.iloc[0].centroid.y
            center_lon = gdf_place.geometry.iloc[0].centroid.x
            
            # Fetch network graph
            G = ox.graph_from_point((center_lat, center_lon), dist=radius, network_type=network_type)
            
            # 3. Calculate Betweenness Centrality
            G_undirected = ox.convert.to_undirected(G)
            edge_centrality = nx.edge_betweenness_centrality(G_undirected)
            nx.set_edge_attributes(G_undirected, edge_centrality, "betweenness")
            
            # Convert graph edges to GeoDataFrame
            gdf_edges = ox.convert.graph_to_gdfs(G_undirected, nodes=False)
            
            # 4. Normalize & Color Code
            min_val = gdf_edges['betweenness'].min()
            max_val = gdf_edges['betweenness'].max()
            
            cmap = plt.get_cmap('turbo')
            
            def get_color_hex(val):
                norm = (val - min_val) / (max_val - min_val + 1e-6)
                return mcolors.to_hex(cmap(norm))
            
            gdf_edges['hex_color'] = gdf_edges['betweenness'].apply(get_color_hex)
            
            # 5. Create Folium Map
            m = folium.Map(location=[center_lat, center_lon], zoom_start=16, tiles="CartoDB dark_matter")
            
            for _, row in gdf_edges.iterrows():
                if row.geometry.geom_type == 'LineString':
                    coords = [[p[1], p[0]] for p in row.geometry.coords]
                    folium.PolyLine(
                        locations=coords,
                        color=row['hex_color'],
                        weight=4,
                        opacity=0.85,
                        tooltip=f"Centrality: {row['betweenness']:.4f}"
                    ).add_to(m)
            
            # Store map object in session state
            st.session_state.axial_map = m

        except Exception as e:
            st.error(f"Error generating analysis: {e}")

# 6. Render Map persistently outside the button conditional
if st.session_state.axial_map is not None:
    st_folium(
        st.session_state.axial_map, 
        width=1000, 
        height=600, 
        key="space_syntax_map",
        returned_objects=[] # Prevents rerun trigger when interacting/clicking on the map
    )