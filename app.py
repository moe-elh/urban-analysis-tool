import streamlit as st
import pandas as pd
import geopandas as gpd
import osmnx as ox
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point, Polygon, LineString
import networkx as nx
import io
import zipfile
import tempfile
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Urban Catchment Analysis | AECOM", 
    page_icon="🏙️", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS Styling ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem 1rem; border-radius: 12px; margin-bottom: 2rem; color: white; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    .main-header h1 { margin: 0; font-size: 2.5rem; font-weight: 700; text-shadow: 0 2px 4px rgba(0,0,0,0.2); }
    .main-header p { margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9; }
    .metric-card { background: white; border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; box-shadow: 0 2px 12px rgba(0,0,0,0.08); border-left: 4px solid #667eea; transition: transform 0.2s ease, box-shadow 0.2s ease; }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 4px 20px rgba(0,0,0,0.12); }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #2c3e50; margin: 0; }
    .metric-label { color: #7f8c8d; font-size: 0.9rem; margin: 0; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-icon { font-size: 2rem; margin-bottom: 0.5rem; }
    .section-header { color: #2c3e50; font-size: 1.2rem; font-weight: 600; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 2px solid #ecf0f1; }
    .stButton > button { width: 100%; border-radius: 8px; border: none; padding: 0.75rem 1rem; font-weight: 600; transition: all 0.2s ease; }
    .stButton > button[kind="primary"] { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3); }
    .stButton > button[kind="primary"]:hover { box-shadow: 0 4px 16px rgba(102, 126, 234, 0.4); transform: translateY(-1px); }
    .status-indicator { padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.85rem; font-weight: 600; text-align: center; margin-bottom: 1rem; }
    .status-ready { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .status-waiting { background-color: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }
</style>
""", unsafe_allow_html=True)

# --- Enhanced UI Components ---
def render_header(): st.markdown("""<div class="main-header"><h1>🏙️ Urban Catchment Analysis</h1><p>Advanced network-based accessibility analysis for urban planning</p></div>""", unsafe_allow_html=True)
def render_metric_card(title, value, icon, description=""): st.markdown(f"""<div class="metric-card"><div class="metric-icon">{icon}</div><p class="metric-label">{title}</p><h3 class="metric-value">{value}</h3>{f'<p style="color: #95a5a6; font-size: 0.8rem; margin-top: 0.5rem;">{description}</p>' if description else ''}</div>""", unsafe_allow_html=True)
def render_status_indicator(status, message): st.markdown(f'<div class="status-indicator status-{status}">{message}</div>', unsafe_allow_html=True)

# --- Core Analysis Functions ---
@st.cache_data(show_spinner=False)
def perform_analysis(lat, lon, dist):
    tags = {"amenity": True, "shop": True, "leisure": True, "public_transport": True}
    G = ox.graph_from_point((lat, lon), dist=dist + 500, dist_type='network', network_type='walk', simplify=True)
    center_node = ox.nearest_nodes(G, lon, lat)
    reachable_nodes_graph = nx.ego_graph(G, center_node, radius=dist, distance='length')
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(reachable_nodes_graph)
    if nodes_gdf.empty: raise ValueError("No reachable nodes found.")
    catchment_polygon = nodes_gdf.unary_union.convex_hull
    total_length_m = edges_gdf['length'].sum()
    catchment_gdf = gpd.GeoDataFrame([{'geometry': catchment_polygon}], crs="EPSG:4326")
    catchment_gdf_proj = ox.project_gdf(catchment_gdf)
    catchment_area_sqm = catchment_gdf_proj.geometry.area.iloc[0]
    pois_gdf = ox.features_from_polygon(catchment_polygon, tags)
    poi_counts = {}
    if not pois_gdf.empty:
        for _, row in pois_gdf.iterrows():
            for tag in ['amenity', 'shop', 'leisure']:
                if tag in row and pd.notna(row[tag]):
                    value = str(row[tag]).replace('_', ' ').title()
                    poi_counts[value] = poi_counts.get(value, 0) + 1
                    break 
    poi_summary_df = pd.DataFrame(list(poi_counts.items()), columns=['Amenity', 'Count']).sort_values(by='Count', ascending=False).reset_index(drop=True)
    network_export, pois_export = edges_gdf.copy(), pois_gdf.copy()
    for gdf in [network_export, pois_export]:
        if not gdf.empty and len(gdf) > 0:
            # Check if all values in the list are None or simple types before converting to string
            for col in gdf.columns:
                if isinstance(gdf[col].iloc[0], list):
                    gdf[col] = gdf[col].apply(lambda x: ', '.join(map(str, x)) if x is not None else '')
    return {
        "reachable_graph": reachable_nodes_graph, "network_gdf": network_export,
        "catchment_gdf": catchment_gdf, "pois_gdf": pois_export,
        "total_length_m": total_length_m, "catchment_area_sqm": catchment_area_sqm,
        "poi_summary_df": poi_summary_df, "center_point": (lat, lon)
    }

def create_gis_export(network_gdf, catchment_gdf, pois_gdf):
    """Creates a zip file of GIS data, handling mixed geometries for POIs."""
    buffer = io.BytesIO()
    with tempfile.TemporaryDirectory() as tempdir:
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            datasets = [("network_streets", network_gdf), ("catchment_area", catchment_gdf)]
            
            # Split POIs into points and polygons
            if not pois_gdf.empty:
                pois_points = pois_gdf[pois_gdf.geometry.geom_type == 'Point']
                pois_polygons = pois_gdf[pois_gdf.geometry.geom_type == 'Polygon']
                if not pois_points.empty: datasets.append(("pois_points", pois_points))
                if not pois_polygons.empty: datasets.append(("pois_polygons", pois_polygons))
                # GeoJSON can handle mixed types, so we write it once
                geojson_path = os.path.join(tempdir, "points_of_interest.geojson")
                pois_gdf.to_file(geojson_path, driver='GeoJSON')
                zipf.write(geojson_path, arcname="points_of_interest.geojson")

            for name, gdf in datasets:
                if name != "points_of_interest": # Avoid re-writing the mixed POI GeoJSON
                    geojson_path = os.path.join(tempdir, f"{name}.geojson")
                    gdf.to_file(geojson_path, driver='GeoJSON')
                    zipf.write(geojson_path, arcname=f"{name}.geojson")

                shp_path = os.path.join(tempdir, f"{name}.shp")
                try:
                    gdf.to_file(shp_path, driver='ESRI Shapefile', encoding='utf-8')
                    base_name = os.path.splitext(shp_path)[0]
                    for file_ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                        file_to_zip = base_name + file_ext
                        if os.path.exists(file_to_zip):
                            zipf.write(file_to_zip, arcname=os.path.basename(file_to_zip))
                except Exception as e:
                    print(f"Could not export {name} to shapefile: {e}")
    buffer.seek(0)
    return buffer.getvalue()

# --- App State Management ---
if 'map_center' not in st.session_state: st.session_state.map_center = [25.1972, 55.2744]
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None
if 'last_clicked' not in st.session_state: st.session_state.last_clicked = None

# --- Main Application Layout ---
render_header()
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    with st.container(border=True):
        st.markdown('<h3 class="section-header">📍 Location Settings</h3>', unsafe_allow_html=True)
        location_input = st.text_input("Search Location", value="Burj Khalifa, Dubai, UAE", help="Enter any address or landmark to center the map")
        if st.button("🎯 Center Map", use_container_width=True):
            try:
                coords = ox.geocode(location_input)
                st.session_state.map_center, st.session_state.analysis_results, st.session_state.last_clicked = [coords[0], coords[1]], None, None
                st.rerun()
            except Exception as e: st.error(f"Location not found: {e}")
    st.write("") 
    with st.container(border=True):
        st.markdown('<h3 class="section-header">⚙️ Analysis Parameters</h3>', unsafe_allow_html=True)
        distance_input = st.slider("Catchment Distance (meters)", 100, 2000, 400, 50, help="Walking distance radius for accessibility analysis")
        if st.session_state.last_clicked: render_status_indicator("ready", "📍 Point selected - Ready to analyze")
        else: render_status_indicator("waiting", "👆 Click a point on the map to begin")
        analyze_button = st.button("🔍 Run Analysis", type="primary", use_container_width=True, disabled=not st.session_state.last_clicked)

    if analyze_button and st.session_state.last_clicked:
        lat, lon = st.session_state.last_clicked['lat'], st.session_state.last_clicked['lng']
        with st.spinner("🔄 Running comprehensive analysis..."):
            try:
                st.session_state.analysis_results = perform_analysis(lat, lon, distance_input)
                st.session_state.map_center = [lat, lon]
                st.rerun()
            except Exception as e:
                st.session_state.analysis_results = None; st.error(f"❌ Analysis failed: {e}")
    
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        st.markdown('<h3 class="section-header">📊 Analysis Results</h3>', unsafe_allow_html=True)
        render_metric_card("Street Network Length", f"{results['total_length_m']:,.0f} m", "🛣️", "Total walkable street distance")
        render_metric_card("Catchment Area", f"{results['catchment_area_sqm'] / 1_000_000:.2f} km²", "📐", "Total accessible area coverage")
        render_metric_card("Points of Interest", f"{len(results['poi_summary_df'])} types", "📍", "Unique amenity categories found")
        if not results['poi_summary_df'].empty:
            with st.container(border=True):
                st.markdown('<h3 class="section-header">🏪 Amenities Breakdown</h3>', unsafe_allow_html=True)
                st.dataframe(results['poi_summary_df'], use_container_width=True, hide_index=True)
                col_a, col_b = st.columns(2)
                with col_a:
                    csv = results['poi_summary_df'].to_csv(index=False).encode('utf-8')
                    st.download_button("📄 Download CSV", csv, "poi_summary.csv", 'text/csv', use_container_width=True)
                with col_b:
                    gis_data = create_gis_export(results['network_gdf'], results['catchment_gdf'], results['pois_gdf'])
                    st.download_button("🗺️ Download GIS", gis_data, "catchment_analysis.zip", "application/zip", use_container_width=True)

with col2:
    with st.container(border=True):
        st.markdown('<h3 class="section-header">🗺️ Interactive Analysis Map</h3>', unsafe_allow_html=True)
        map_to_display = folium.Map(location=st.session_state.map_center, zoom_start=15, tiles="cartodbpositron")
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            ox.plot_graph_folium(results['reachable_graph'], graph_map=map_to_display, edge_color="#667eea", edge_width=2, edge_opacity=0.8)
            folium.GeoJson(results['catchment_gdf'], style_function=lambda x: {'fillColor': '#667eea', 'color': '#764ba2', 'weight': 3, 'fillOpacity': 0.2}).add_to(map_to_display)
            if not results['pois_gdf'].empty:
                for _, row in results['pois_gdf'].iterrows():
                    geom, tooltip_text = row.geometry, ""
                    if isinstance(geom, Point): loc = [geom.y, geom.x]
                    elif isinstance(geom, (Polygon, LineString)): loc = [geom.centroid.y, geom.centroid.x]
                    else: continue
                    if 'name' in row and pd.notna(row['name']): tooltip_text += f"<b>{row['name']}</b><br>"
                    for tag in ['amenity', 'shop', 'leisure']:
                        if tag in row and pd.notna(row[tag]): tooltip_text += f"Type: {str(row[tag]).replace('_', ' ').title()}"; break
                    folium.CircleMarker(location=loc, radius=5, color='#e74c3c', fill=True, fillColor='#e74c3c', fillOpacity=0.8, tooltip=tooltip_text).add_to(map_to_display)
            folium.Marker(list(results['center_point']), tooltip="Analysis Center", icon=folium.Icon(color="green", icon="star")).add_to(map_to_display)
        elif st.session_state.last_clicked:
            folium.Marker([st.session_state.last_clicked['lat'], st.session_state.last_clicked['lng']], tooltip="Selected Point", icon=folium.Icon(color='red', icon='map-pin')).add_to(map_to_display)
        
        map_data = st_folium(map_to_display, width='100%', height=600, returned_objects=['last_clicked'])
        if map_data and map_data.get('last_clicked'):
            if st.session_state.last_clicked != map_data.get('last_clicked'):
                st.session_state.last_clicked = map_data.get('last_clicked')
                st.session_state.analysis_results = None
                st.rerun()