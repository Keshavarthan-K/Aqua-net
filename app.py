import streamlit as st
import streamlit.components.v1 as components
import torch
import pandas as pd
import json
import shapely.geometry
from shapely.geometry import LineString
from train import AquaNet

st.set_page_config(layout="wide", page_title="AquaNet: AI Flood Engine")

# --- 1. SECURE API SETUP ---
try:
    maptiler_key = st.secrets["MAPTILER_KEY"]
except KeyError:
    st.error("⚠️ MapTiler API Key not found. Please add it to .streamlit/secrets.toml")
    st.stop()

# --- 2. CACHE THE BRAIN & GEOMETRY ---
@st.cache_resource
def load_ai_brain():
    try:
        model = AquaNet()
        model.load_state_dict(torch.load("aquanet_brain.pth", map_location=torch.device('cpu'), weights_only=True))
        model.eval()
        graph_data = torch.load("chennai_mega_data.pt", map_location=torch.device('cpu'), weights_only=False)
        return model, graph_data
    except Exception as e:
        st.error(f"Error loading model or data: {e}")
        st.stop()

@st.cache_data
def build_street_geometries():
    _, graph_data = load_ai_brain()
    edges = graph_data.edge_index.numpy()
    src_nodes, dst_nodes = edges[0], edges[1]
    
    try:
        df = pd.read_csv('chennai_dashboard_data.csv')
        lon_col = 'lon' if 'lon' in df.columns else 'longitude'
        lat_col = 'lat' if 'lat' in df.columns else 'latitude'
        lons, lats = df[lon_col].tolist(), df[lat_col].tolist()
    except Exception as e:
        st.error(f"Could not load GPS coordinates: {e}")
        st.stop()
    
    geometries = []
    buffer_degrees = 0.000045 
    
    for i in range(len(src_nodes)):
        src, dst = src_nodes[i], dst_nodes[i]
        if src == dst:
            geometries.append(None)
            continue
            
        p1, p2 = (lons[src], lats[src]), (lons[dst], lats[dst])
        if p1 == p2:
            geometries.append(None)
            continue
            
        line = LineString([p1, p2])
        poly = line.buffer(buffer_degrees, cap_style=2)
        geometries.append((src, dst, shapely.geometry.mapping(poly)))
        
    return geometries

# --- 3. DYNAMIC GNN INFERENCE ---
def get_dynamic_flood_geojson(rainfall_cm, duration_hr):
    model, graph_data = load_ai_brain()
    current_graph = graph_data.clone()
    
    # Safely clone the tensor to avoid memory warnings
    current_graph.x = graph_data.x.clone()
    current_graph.x[:, 2] = rainfall_cm  
    current_graph.x[:, 3] = duration_hr  
    
    with torch.no_grad():
        predictions = model(current_graph).squeeze().tolist()
        
    geometries = build_street_geometries()
    features = []
    max_depth = 0.0
    
    for geom_data in geometries:
        if geom_data is None: continue
        src, dst, poly_mapping = geom_data
        
        edge_depth_m = (predictions[src] + predictions[dst]) / 2.0
        
        if edge_depth_m > max_depth:
            max_depth = edge_depth_m
            
        if edge_depth_m > 0.05: # Only render water deeper than 5cm
            features.append({
                "type": "Feature",
                "geometry": poly_mapping,
                "properties": {"depth_m": float(edge_depth_m)}
            })
            
    return json.dumps({"type": "FeatureCollection", "features": features}), max_depth

# --- 4. MAPLIBRE RENDERING ---
def render_maplibre_3d(api_key, gnn_data):
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>MapLibre 3D Flood Map</title>
        <meta name="viewport" content="initial-scale=1,maximum-scale=1,user-scalable=no">
        <script src="https://unpkg.com/maplibre-gl@3.3.1/dist/maplibre-gl.js"></script>
        <link href="https://unpkg.com/maplibre-gl@3.3.1/dist/maplibre-gl.css" rel="stylesheet" />
        <style>
            body {{ margin: 0; padding: 0; font-family: 'Inter', sans-serif; overflow: hidden; }}
            #map {{ position: absolute; top: 0; bottom: 0; width: 100vw; height: 100vh; }}
            .maplibregl-popup-content {{
                background-color: rgba(15, 23, 42, 0.95) !important;
                backdrop-filter: blur(8px); border: 1px solid #0ea5e9 !important;
                border-radius: 8px !important; padding: 12px !important;
                box-shadow: 0 10px 30px rgba(0,0,0,0.8) !important; color: #ffffff !important;
            }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        <script>
            const map = new maplibregl.Map({{
                container: 'map',
                style: 'https://api.maptiler.com/maps/basic-v2-dark/style.json?key={api_key}',
                center: [80.2619, 13.0550],
                zoom: 13,
                pitch: 60, 
                bearing: -20,
                antialias: true
            }});

            map.on('load', () => {{
                map.addSource('terrain-dem', {{
                    'type': 'raster-dem',
                    'url': 'https://api.maptiler.com/tiles/terrain-rgb-v2/tiles.json?key={api_key}'
                }});
                map.setTerrain({{ 'source': 'terrain-dem', 'exaggeration': 1.5 }});

                map.addLayer({{
                    'id': '3d-buildings',
                    'source': 'maptiler_planet',
                    'source-layer': 'building',
                    'type': 'fill-extrusion',
                    'minzoom': 14,
                    'paint': {{
                        'fill-extrusion-color': '#334155',
                        'fill-extrusion-height': ['get', 'render_height'],
                        'fill-extrusion-opacity': 0.8
                    }}
                }});

                map.addSource('gnn-streets', {{
                    'type': 'geojson',
                    'data': {gnn_data}
                }});

                map.addLayer({{
                    'id': 'gnn-3d-streets-layer',
                    'type': 'fill-extrusion',
                    'source': 'gnn-streets',
                    'paint': {{
                        'fill-extrusion-color': [
                            'interpolate', ['linear'], ['get', 'depth_m'],
                            0.0, '#7dd3fc',   
                            0.5, '#0284c7',  
                            1.5, '#1e3a8a'   
                        ],
                        // INCREASED MULTIPLIER: Exaggerate height by 30x so it is visible against buildings
                        'fill-extrusion-height': ['*', ['get', 'depth_m'], 30], 
                        'fill-extrusion-opacity': 0.90
                    }}
                }});

                const popup = new maplibregl.Popup({{ closeButton: false, closeOnClick: false, offset: 15 }});

                map.on('mousemove', 'gnn-3d-streets-layer', (e) => {{
                    if (e.features.length > 0) {{
                        map.getCanvas().style.cursor = 'crosshair';
                        const depth = e.features[0].properties.depth_m;
                        
                        const htmlContent = 
                            '<div style="font-size: 11px; color: #94a3b8; margin-bottom: 4px; text-transform: uppercase;">AI Predicted Depth</div>' +
                            '<div style="font-size: 18px; font-weight: bold; color: #38bdf8;">' + depth.toFixed(2) + ' m</div>';

                        popup.setLngLat(e.lngLat).setHTML(htmlContent).addTo(map);
                    }}
                }});

                map.on('mouseleave', 'gnn-3d-streets-layer', () => {{
                    map.getCanvas().style.cursor = '';
                    popup.remove();
                }});
            }});
            map.addControl(new maplibregl.NavigationControl());
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=750)

# --- 5. STREAMLIT UI ---
st.title("🌊 AquaNet: AI Flood Prediction Engine")

col1, col2 = st.columns(2)
with col1:
    rain_input = st.slider("Forecasted Rainfall (cm)", min_value=0.0, max_value=40.0, value=20.0, step=1.0)
with col2:
    time_input = st.slider("Storm Duration (hours)", min_value=1.0, max_value=24.0, value=2.0, step=1.0)

# Generate GeoJSON and capture the highest water level predicted
gnn_geojson, max_depth_m = get_dynamic_flood_geojson(rain_input, time_input)

st.info(f"🧠 **GNN Physics Telemetry:** Processing {rain_input} cm of rain over {time_input} hours. The AI predicts a maximum street flood depth of **{max_depth_m:.2f} meters**.")

# Render the Map
render_maplibre_3d(maptiler_key, gnn_geojson)