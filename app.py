import streamlit as st
import streamlit.components.v1 as components

# 1. UI Setup
st.set_page_config(page_title="AquaNet Chennai", layout="wide")
st.title("🌊 AquaNet: 3D Urban Flood Twin")
st.markdown("Powered by MapLibre & MapTiler (No Credit Card Required)")

st.sidebar.markdown("### 🔑 MapTiler API Setup")
maptiler_key = st.sidebar.text_input("MapTiler API Key", type="password")

if not maptiler_key:
    st.warning("⚠️ Please paste your free MapTiler API Key in the sidebar to render the 3D City.")
    st.sidebar.markdown("""
    **Get your free key:**
    1. Go to [maptiler.com/cloud/](https://www.maptiler.com/cloud/)
    2. Sign up with email.
    3. Go to API Keys and copy the default key.
    """)
    st.stop()

def render_maplibre_3d(api_key):
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>MapLibre 3D Flood Map</title>
        <meta name="viewport" content="initial-scale=1,maximum-scale=1,user-scalable=no">
        
        <!-- Import Open-Source MapLibre GL JS -->
        <script src="https://unpkg.com/maplibre-gl@3.3.1/dist/maplibre-gl.js"></script>
        <link href="https://unpkg.com/maplibre-gl@3.3.1/dist/maplibre-gl.css" rel="stylesheet" />
        
        <style>
            body {{ margin: 0; padding: 0; font-family: 'Inter', sans-serif; overflow: hidden; }}
            #map {{ position: absolute; top: 0; bottom: 0; width: 100vw; height: 100vh; }}
            
            /* Custom Floating Dashboard */
            #floating-panel {{
                position: absolute; bottom: 40px; left: 50%; transform: translateX(-50%);
                background: rgba(15, 23, 42, 0.9); backdrop-filter: blur(12px);
                color: white; padding: 20px 30px; border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 0.1); box-shadow: 0 10px 40px rgba(0, 0, 0, 0.6);
                z-index: 10; width: 350px; text-align: center;
            }}
            #floating-panel h3 {{ margin: 0 0 15px 0; font-size: 16px; color: #38bdf8; }}
            input[type=range] {{ width: 100%; cursor: pointer; accent-color: #0ea5e9; }}
            
            /* Guaranteed visibility for the custom MapLibre popup */
            .maplibregl-popup-content {{
                background-color: rgba(15, 23, 42, 0.95) !important;
                backdrop-filter: blur(8px);
                border: 1px solid #0ea5e9 !important;
                border-radius: 8px !important;
                padding: 12px !important;
                box-shadow: 0 10px 30px rgba(0,0,0,0.8) !important;
                color: #ffffff !important;
                font-family: 'Inter', sans-serif !important;
                text-align: center !important;
                pointer-events: none; /* Let mouse pass through to map */
            }}
            .maplibregl-popup-tip {{ 
                border-top-color: #0ea5e9 !important; 
                border-bottom-color: #0ea5e9 !important; 
            }}
        </style>
    </head>
    <body>
    
        <div id="map"></div>
        
        <div id="floating-panel">
            <h3>Simulated Rainfall: <span id="rain-val">0.0</span> cm/hr</h3>
            <input type="range" id="rainfall-slider" min="0" max="25" step="0.5" value="0">
            <p style="margin: 10px 0 0 0; font-size: 12px; color: #94a3b8;">
                Estimated Flood Depth: <span id="depth-val">0.0</span> meters
            </p>
        </div>

        <script>
            // Initialize MapLibre focusing deep inside Chennai (Royapettah area)
            const map = new maplibregl.Map({{
                container: 'map',
                style: 'https://api.maptiler.com/maps/basic-v2-dark/style.json?key={api_key}',
                center: [80.2619, 13.0550], // Deep street level
                zoom: 15.5, // High zoom to guarantee 3D buildings render
                pitch: 65, 
                bearing: -20,
                antialias: true
            }});

            map.on('load', () => {{
                // 1. ADD TRUE 3D TERRAIN (Bumpy hills and valleys)
                map.addSource('terrain-dem', {{
                    'type': 'raster-dem',
                    'url': 'https://api.maptiler.com/tiles/terrain-rgb-v2/tiles.json?key={api_key}'
                }});
                map.setTerrain({{ 'source': 'terrain-dem', 'exaggeration': 1.5 }});

                // Find the first text label layer so we can insert under it
                const layers = map.getStyle().layers;
                let labelLayerId;
                for (let i = 0; i < layers.length; i++) {{
                    if (layers[i].type === 'symbol' && layers[i].layout['text-field']) {{
                        labelLayerId = layers[i].id; break;
                    }}
                }}

                // 2. ADD 3D BUILDINGS
                map.addLayer({{
                    'id': '3d-buildings',
                    'source': 'maptiler_planet',
                    'source-layer': 'building',
                    'type': 'fill-extrusion',
                    'minzoom': 14,
                    'paint': {{
                        'fill-extrusion-color': '#334155',
                        'fill-extrusion-height': ['get', 'render_height'],
                        'fill-extrusion-base': ['get', 'render_min_height'],
                        'fill-extrusion-opacity': 0.8
                    }}
                }}, labelLayerId);

                // 3. MASSIVE FLOOD POLYGON (Hidden by default)
                map.addSource('flood-zone', {{
                    'type': 'geojson',
                    'data': {{
                        'type': 'Feature',
                        'geometry': {{
                            'type': 'Polygon',
                            'coordinates': [[
                                [79.5, 12.5], [80.8, 12.5], [80.8, 13.5], [79.5, 13.5], [79.5, 12.5]
                            ]]
                        }}
                    }}
                }});

                map.addLayer({{
                    'id': 'flood-water',
                    'type': 'fill-extrusion',
                    'source': 'flood-zone',
                    'paint': {{
                        'fill-extrusion-color': '#0ea5e9',
                        'fill-extrusion-height': 0,
                        'fill-extrusion-base': 0,
                        'fill-extrusion-opacity': 0.0 // 100% invisible at 0 rainfall
                    }}
                }}, labelLayerId);

                // 4. REAL-TIME SLIDER LOGIC
                const slider = document.getElementById('rainfall-slider');
                const rainVal = document.getElementById('rain-val');
                const depthVal = document.getElementById('depth-val');

                slider.addEventListener('input', (e) => {{
                    const rainfall = parseFloat(e.target.value);
                    rainVal.innerText = rainfall.toFixed(1);
                    
                    if (rainfall === 0) {{
                        // Instantly hide water when slider is 0
                        map.setPaintProperty('flood-water', 'fill-extrusion-opacity', 0.0);
                        map.setPaintProperty('flood-water', 'fill-extrusion-height', 0);
                        depthVal.innerText = "0.0";
                    }} else {{
                        // Fade water in and raise height
                        const floodDepth = rainfall * 1.5; 
                        map.setPaintProperty('flood-water', 'fill-extrusion-opacity', 0.65);
                        map.setPaintProperty('flood-water', 'fill-extrusion-height', floodDepth);
                        depthVal.innerText = floodDepth.toFixed(1);
                    }}
                }});

                // 5. BULLETPROOF HOVER POPUP LOGIC
                const popup = new maplibregl.Popup({{
                    closeButton: false,
                    closeOnClick: false,
                    offset: 15
                }});

                // Listen to the entire map instead of the specific layer to prevent 3D clipping bugs
                map.on('mousemove', (e) => {{
                    const currentRainfall = parseFloat(document.getElementById('rainfall-slider').value);
                    
                    // Hide popup if slider is at 0
                    if (currentRainfall <= 0) {{
                        popup.remove();
                        map.getCanvas().style.cursor = '';
                        return;
                    }}
                    
                    map.getCanvas().style.cursor = 'crosshair';
                    
                    const lng = e.lngLat.lng;
                    const lat = e.lngLat.lat;
                    
                    // Procedural Math Simulation for Global Scale testing
                    // This generates dynamic hotspots across the map surface
                    const noise1 = Math.sin((lng - 80.0) * 80) * Math.cos((lat - 12.8) * 80);
                    const noise2 = Math.sin((lng - 80.0) * 200) * Math.cos((lat - 12.8) * 200);
                    const catchmentMultiplier = Math.abs(noise1 + noise2 * 0.5) * 20.0 + 5.0; 
                    
                    // Depth = Rainfall * Catchment (Calculated on the fly)
                    let predictedDepth = (currentRainfall / 100) * catchmentMultiplier; 
                    if (predictedDepth < 0.01) predictedDepth = 0.01;

                    // Standard HTML concatenation
                    const htmlContent = 
                        '<div style="font-size: 11px; color: #94a3b8; margin-bottom: 4px; text-transform: uppercase;">Predicted Flood Depth</div>' +
                        '<div style="font-size: 18px; font-weight: bold; color: #38bdf8;">' + predictedDepth.toFixed(2) + ' m</div>';

                    popup.setLngLat(e.lngLat)
                         .setHTML(htmlContent)
                         .addTo(map);
                }});

                map.on('mouseout', () => {{
                    map.getCanvas().style.cursor = '';
                    popup.remove();
                }});
            }});
            
            map.addControl(new maplibregl.NavigationControl());
        </script>
    </body>
    </html>
    """
    
    # Render component taking up full screen height
    components.html(html_code, height=750)

render_maplibre_3d(maptiler_key)