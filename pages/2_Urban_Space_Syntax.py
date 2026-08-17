import streamlit as st
import osmnx as ox
import networkx as nx
import geopandas as gpd
import pydeck as pdk
import matplotlib.cm as cm
import matplotlib.colors as mcolors

st.set_page_config(page_title="Urban Space Syntax", layout="wide")
st.title("Singapore Urban Space Syntax Analysis")

# 1. User Input Controls
col1, col2 = st.sidebar.columns(2)
with col1:
    search_query = st.text_input("Location in Singapore", "Orchard Road, Singapore")
with col2:
    radius = st.slider("Radius (meters)", min_value=300, max_value=2000, value=800, step=100)

network_type = st.sidebar.selectbox("Network Type", ["walk", "drive", "all"])

if st.button("Run Axial Analysis"):
    with st.spinner("Fetching street network and calculating centrality..."):
        try:
            # 2. Geocode & Fetch Street Graph from OpenStreetMap
            gdf_place = ox.geocode_to_gdf(search_query)
            center_lat = gdf_place.geometry.iloc[0].centroid.y
            center_lon = gdf_place.geometry.iloc[0].centroid.x
            
            G = ox.graph_from_point((center_lat, center_lon), dist=radius, network_type=network_type)
            
            # 3. Convert to Line Geometry & Calculate Integration/Betweenness Centrality
            # Line segment betweenness centrality closely mimics axial integration/choice metrics
            G_undirected = ox.convert.to_undirected(G)
            edge_centrality = nx.edge_betweenness_centrality(G_undirected)
            nx.set_edge_attributes(G_undirected, edge_centrality, "betweenness")
            
            # Convert graph edges to GeoDataFrame
            _, gdf_edges = ox.convert.graph_to_gdfs(G_undirected)
            
            # 4. Color Mapping for Space Syntax Visualization
            min_val = gdf_edges['betweenness'].min()
            max_val = gdf_edges['betweenness'].max()
            
            # Scale colors using 'turbo' or 'jet' (Space Syntax classic palette: red = high, blue = low)
            cmap = cm.get_cmap('turbo')
            
            def get_color(val):
                norm = (val - min_val) / (max_val - min_val + 1e-6)
                rgba = cmap(norm)
                return [int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255), 200]
            
            gdf_edges['color'] = gdf_edges['betweenness'].apply(get_color)
            
            # Prepare PyDeck Path Layer format
            paths = []
            for _, row in gdf_edges.iterrows():
                if row.geometry.geom_type == 'LineString':
                    coords = [[p[0], p[1]] for p in row.geometry.coords]
                    paths.append({'path': coords, 'color': row['color'], 'score': row['betweenness']})
            
            layer = pdk.Layer(
                "PathLayer",
                data=paths,
                get_path="path",
                get_color="color",
                width_min_pixels=3,
                pickable=True
            )
            
            view_state = pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=15,
                pitch=30
            )
            
            # Render interactive PyDeck map
            st.pydeck_chart(pdk.Deck(
                map_style="mapbox://styles/mapbox/dark-v10",
                initial_view_state=view_state,
                layers=[layer],
                tooltip={"text": "Centrality Score: {score}"}
            ))
            
        except Exception as e:
            st.error(f"Error generating analysis: {e}")