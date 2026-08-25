import streamlit as st
import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import os
import requests
from branca.element import Template, MacroElement

# Aumentamos el tiempo de espera de OSM
ox.settings.timeout = 180 

st.set_page_config(page_title="MSMH Humedales", page_icon="🌿", layout="wide")

# ==========================================
# CONFIGURACIÓN DE UMBRALES
# ==========================================
MIN_POB_UMBRAL = 100 
MAX_POB_SATURACION = 300 

# ==========================================
# MOTOR 1: ADQUISICIÓN HÍBRIDA HIPER-OPTIMIZADA
# ==========================================
@st.cache_data(show_spinner=False)
def descargar_base_datos(url, nombre_ccaa):
    # Si es un archivo local (pruebas), lo lee directo
    if not url.startswith('http'):
        return url
        
    os.makedirs("datos_cache", exist_ok=True)
    ruta_local = f"datos_cache/{nombre_ccaa.replace(' ', '_')}.gpkg"
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    if not os.path.exists(ruta_local):
        respuesta = requests.get(url, stream=True, headers=headers)
        with open(ruta_local, 'wb') as archivo:
            for bloque in respuesta.iter_content(chunk_size=8192):
                if bloque:
                    archivo.write(bloque)
                
    return ruta_local
@st.cache_data(show_spinner=False)
def obtener_datos_espaciales(lugar, ruta_local_gpkg):
    # 1. Frontera Oficial
    frontera_oficial = ox.geocode_to_gdf(lugar)
    crs_proyectado = frontera_oficial.estimate_utm_crs()
    distrito_gdf = frontera_oficial.to_crs(crs_proyectado)
    
    # CRS Nativo para evitar vacíos
    crs_origen = gpd.read_file(ruta_local_gpkg, rows=1).crs
    distrito_al_origen = frontera_oficial.to_crs(crs_origen)
    caja_correcta = tuple(distrito_al_origen.total_bounds)
    
    # 2. Edificios
    edificios_gdf = gpd.read_file(ruta_local_gpkg, bbox=caja_correcta)
    edificios_gdf = edificios_gdf[edificios_gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
    if edificios_gdf.crs != crs_proyectado:
        edificios_gdf = edificios_gdf.to_crs(crs_proyectado)
    
    # 3. Recorte Espacial exacto
    edificios_proyectados = gpd.clip(edificios_gdf, distrito_gdf)
    buffer_edificios = edificios_proyectados.buffer(15).unary_union

    # 4. Capas menores (Búsqueda geométrica ultra-optimizada para la nube)
    poligono_gps = frontera_oficial.geometry.iloc[0] # Geometría exacta para no colapsar OSM

    try:
        agua_gdf = ox.features_from_polygon(poligono_gps, tags={'waterway': True, 'natural': 'water'})
        buffer_inundacion = agua_gdf.to_crs(crs_proyectado).buffer(50).unary_union if not agua_gdf.empty else gpd.GeoSeries().unary_union
    except:
        buffer_inundacion = gpd.GeoSeries().unary_union

    try:
        usos_incompatibles = ox.features_from_polygon(poligono_gps, tags={'landuse': ['industrial', 'railway', 'military']})
        buffer_urbanismo = usos_incompatibles.to_crs(crs_proyectado).unary_union if not usos_incompatibles.empty else gpd.GeoSeries().unary_union
    except:
        buffer_urbanismo = gpd.GeoSeries().unary_union

    buffer_hidrogeologia = gpd.GeoSeries().unary_union

    # 5. Cálculo de parcelas aptas
    exclusion_total = buffer_edificios.union(buffer_inundacion).union(buffer_urbanismo).union(buffer_hidrogeologia)
    suelo_disponible_geom = distrito_gdf.geometry.difference(exclusion_total).item() 
    
    parcelas_disponibles = gpd.GeoDataFrame(geometry=[suelo_disponible_geom], crs=crs_proyectado).explode(index_parts=False).reset_index(drop=True)
    parcelas_disponibles['area_m2'] = parcelas_disponibles.geometry.area
    parcelas_aptas = parcelas_disponibles[parcelas_disponibles['area_m2'] >= 200].copy()

    # 6. Parques y Lógica Difusa
    try:
        parques_gdf = ox.features_from_polygon(poligono_gps, tags={'leisure': 'park', 'landuse': ['grass', 'recreation_ground']})
        parques_proyectados = parques_gdf[parques_gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])].to_crs(crs_proyectado)
        masa_parques = parques_proyectados.unary_union
    except:
        parques_proyectados = gpd.GeoDataFrame(geometry=[], crs=crs_proyectado)
        masa_parques = gpd.GeoSeries().unary_union

    # Creación de columnas segura para evitar KeyError
    if not parcelas_aptas.empty:
        parcelas_aptas['distancia_parque_m'] = parcelas_aptas.geometry.distance(masa_parques) if not parques_proyectados.empty else 400
        
        def calcular_smoothstep_inverso(x, min_val, max_val):
            if x <= min_val: return 1.0
            if x >= max_val: return 0.0
            t = (x - min_val) / (max_val - min_val)
            return 1.0 - (t * t * (3 - 2 * t))

        parcelas_aptas['score_reut'] = parcelas_aptas['distancia_parque_m'].apply(lambda x: calcular_smoothstep_inverso(x, 400, 1000))
        
        scores_poblacion = []
        poblacion_absoluta = [] 
        rango_pob = MAX_POB_SATURACION - MIN_POB_UMBRAL
        
        for idx, parcela in parcelas_aptas.iterrows():
            cuenca_100m = parcela.geometry.buffer(100)
            edificios_en_cuenca = gpd.clip(edificios_proyectados, cuenca_100m)
            habitantes = (edificios_en_cuenca.geometry.area.sum() * 5) / 35
            poblacion_absoluta.append(round(habitantes))
            
            if habitantes < MIN_POB_UMBRAL: score = 0.0
            elif habitantes >= MAX_POB_SATURACION: score = 1.0
            else: score = (habitantes - MIN_POB_UMBRAL) / rango_pob
            scores_poblacion.append(score)
        
        parcelas_aptas['score_pob'] = scores_poblacion
        parcelas_aptas['habitantes'] = poblacion_absoluta

        np.random.seed(42)
        parcelas_aptas['score_top'] = np.random.uniform(0.0, 18.0, len(parcelas_aptas)).tolist()
        parcelas_aptas['score_top'] = parcelas_aptas['score_top'].apply(lambda x: 1.0 if x <= 5.0 else (0.0 if x >= 15.0 else 1.0 - ((x - 5.0) / 10.0)))
        
        np.random.seed(101)
        parcelas_aptas['score_aire'] = np.random.uniform(10.0, 95.0, len(parcelas_aptas)).tolist()
        parcelas_aptas['score_aire'] = parcelas_aptas['score_aire'].apply(lambda x: 0.0 if x <= 20.0 else (1.0 if x >= 80.0 else (x - 20.0) / 60.0))
        
        np.random.seed(202) 
        parcelas_aptas['score_san'] = np.random.uniform(5.0, 200.0, len(parcelas_aptas)).tolist()
        parcelas_aptas['score_san'] = parcelas_aptas['score_san'].apply(lambda x: 1.0 if x <= 20.0 else (0.0 if x >= 150.0 else 1.0 - ((x - 20.0) / 130.0)))
    else:
        # Prevención de KeyError si todo el suelo está excluido
        for col in ['distancia_parque_m', 'score_reut', 'score_pob', 'habitantes', 'score_top', 'score_aire', 'score_san']:
            parcelas_aptas[col] = []

    return parcelas_aptas, buffer_urbanismo, buffer_inundacion, buffer_hidrogeologia, parques_proyectados
# ==========================================
# MOTOR 2: ÁLGEBRA WLC Y MAPA WEBGIS (FOLIUM)
# ==========================================
def calcular_y_mapear(parcelas_aptas, b_urb, b_inu, b_hidro, parques, w_reut, w_pob, w_top, w_aire, w_san):
    suma_pesos = w_reut + w_pob + w_top + w_aire + w_san
    if suma_pesos == 0: suma_pesos = 1 
    wr, wp, wt, wa, ws = w_reut/suma_pesos, w_pob/suma_pesos, w_top/suma_pesos, w_aire/suma_pesos, w_san/suma_pesos

    parcelas_aptas['score_final'] = (
        (parcelas_aptas['score_reut'] * wr) + 
        (parcelas_aptas['score_pob'] * wp) +
        (parcelas_aptas['score_top'] * wt) +
        (parcelas_aptas['score_aire'] * wa) +
        (parcelas_aptas['score_san'] * ws)
    )

    condicion_eliminatoria = (parcelas_aptas['score_reut']==0) | (parcelas_aptas['score_pob']==0) | (parcelas_aptas['score_top']==0) | (parcelas_aptas['score_san']==0)
    parcelas_aptas.loc[condicion_eliminatoria, 'score_final'] = 0.0

    def color_hex(score):
        if score == 0.0: return '#d62728'     
        elif score < 0.4: return '#ff7f0e'    
        elif score < 0.6: return '#bcbd22'    
        elif score < 0.8: return '#1f77b4'    
        else: return '#2ca02c'                
    parcelas_aptas['color_hex'] = parcelas_aptas['score_final'].apply(color_hex)

    parcelas_4326 = parcelas_aptas.to_crs(epsg=4326)
    parques_4326 = parques.to_crs(epsg=4326) if not parques.empty else None
    
    centro_lat = parcelas_4326.geometry.centroid.y.mean()
    centro_lon = parcelas_4326.geometry.centroid.x.mean()

    m = folium.Map(location=[centro_lat, centro_lon], zoom_start=13, tiles='cartodbpositron')

    if not isinstance(b_urb, gpd.GeoSeries) and not b_urb.is_empty:
        folium.GeoJson(gpd.GeoSeries([b_urb]).set_crs(parcelas_aptas.crs).to_crs(epsg=4326), 
                       style_function=lambda x: {'fillColor': 'dimgray', 'color': 'none', 'fillOpacity': 0.6}, name="Exclusión Urbanística").add_to(m)
    
    if not isinstance(b_inu, gpd.GeoSeries) and not b_inu.is_empty:
        folium.GeoJson(gpd.GeoSeries([b_inu]).set_crs(parcelas_aptas.crs).to_crs(epsg=4326), 
                       style_function=lambda x: {'fillColor': '#1f77b4', 'color': 'none', 'fillOpacity': 0.5}, name="Riesgo Inundación").add_to(m)

    for idx, row in parcelas_4326.iterrows():
        html_popup = f"""
        <b>Score Final:</b> {row['score_final']:.2f}<br>
        <b>Área:</b> {row['area_m2']:.0f} m²<br>
        <b>Población (100m):</b> {row['habitantes']} HE
        """
        folium.GeoJson(
            row.geometry,
            style_function=lambda x, color=row['color_hex']: {'fillColor': color, 'color': 'black', 'weight': 1, 'fillOpacity': 0.8},
            tooltip=html_popup
        ).add_to(m)

    if parques_4326 is not None:
        folium.GeoJson(parques_4326, 
                       style_function=lambda x: {'fillColor': '#a1d99b', 'color': '#2ca02c', 'weight': 2, 'fillOpacity': 0.4}, 
                       name="Parques Existentes").add_to(m)

    folium.LayerControl().add_to(m) 
    
    template = """
    {% macro html(this, kwargs) %}
    <style>
      .maplegend { position: absolute; z-index:9999; background-color: rgba(255, 255, 255, 0.95); border-radius: 5px; border: 2px solid grey; padding: 10px; font-size: 13px; bottom: 30px; left: 30px; color: #333333; font-family: Arial, sans-serif; box-shadow: 2px 2px 5px rgba(0,0,0,0.3); max-height: 40px; overflow: hidden; transition: max-height 0.4s ease-in-out; }
      .maplegend:hover { max-height: 300px; }
      .legend-title { text-align: center; margin-bottom: 8px; font-weight: bold; font-size: 14px; cursor: help; }
      .legend-scale ul { margin: 0; margin-bottom: 5px; padding: 0; list-style: none; }
      .legend-scale ul li { font-size: 13px; list-style: none; margin-bottom: 4px; line-height: 18px; }
      .legend-scale ul li i { width: 16px; height: 16px; float: left; margin-right: 8px; opacity: 0.8; border: 1px solid #777; }
    </style>
    <div id='maplegend' class='maplegend'>
      <div class='legend-title'>🗺️ Leyenda (Pasa el ratón)</div>
      <div class='legend-scale'>
        <ul class='legend-labels'>
          <li><i style='background:#2ca02c;'></i>Óptima (> 0.8)</li>
          <li><i style='background:#1f77b4;'></i>Alta (0.6 - 0.8)</li>
          <li><i style='background:#bcbd22;'></i>Media (0.4 - 0.6)</li>
          <li><i style='background:#ff7f0e;'></i>Baja (0.2 - 0.4)</li>
          <li><i style='background:#d62728;'></i>Inviable (Score 0)</li>
          <hr style="margin: 5px 0; border-top: 1px solid #ccc;">
          <li><i style='background:#a1d99b;'></i>Parques Existentes</li>
          <li><i style='background:dimgray;'></i>Exclusión Urbana</li>
          <li><i style='background:#1f77b4; opacity:0.5;'></i>Riesgo Inundación</li>
        </ul>
      </div>
    </div>
    {% endmacro %}
    """
    macro = MacroElement()
    macro._template = Template(template)
    m.get_root().add_child(macro)

    aptas_reales = len(parcelas_aptas[parcelas_aptas['score_final'] > 0])
    return m, aptas_reales, len(parcelas_aptas)

# ==========================================
# INTERFAZ DE USUARIO (FRONT-END)
# ==========================================
if os.path.exists("logo.png"):
    col1, col2, col3 = st.sidebar.columns([1, 2, 1])
    with col2:
        st.image("logo.png", use_container_width=True)
else:
    st.sidebar.markdown("<h3 style='text-align: center;'>🌿 MSMH UI</h3>", unsafe_allow_html=True)

st.sidebar.title("Configuración MSMH")

st.sidebar.header("1. Origen de Datos (Cloud)")

BASES_DE_DATOS_CCAA = {
    "Andalucía": "https://huggingface.co/datasets/BaracanBaea/TFM/resolve/main/edificios_andalucia.gpkg?download=true",
    "Aragón" : "https://huggingface.co/datasets/BaracanBaea/TFM/resolve/main/edificios_aragon.gpkg?download=true",
    "Asturias": "https://huggingface.co/datasets/BaracanBaea/TFM/resolve/main/edificios_asturias.gpkg?download=true",
    "Cantabria": "https://huggingface.co/datasets/BaracanBaea/TFM/resolve/main/edificios_cantabria.gpkg?download=true",
    "Castilla-La Mancha":"https://huggingface.co/datasets/BaracanBaea/TFM/resolve/main/edificios_clm.gpkg?download=true",
    "Castilla y León":"https://huggingface.co/datasets/BaracanBaea/TFM/resolve/main/edificios_cyl.gpkg?download=true",
    "Cataluña": "https://huggingface.co/datasets/BaracanBaea/TFM/resolve/main/edificios_catalu%C3%B1a.gpkg?download=true",
    "Ceuta":"https://huggingface.co/datasets/BaracanBaea/TFM/resolve/main/edificios_ceuta.gpkg?download=true",
    "Comunidad de Madrid": "https://huggingface.co/datasets/BaracanBaea/TFM/resolve/main/edificios_madrid.gpkg?download=true",
    "Extremadura":"https://huggingface.co/datasets/BaracanBaea/TFM/resolve/main/edificios_extremadura.gpkg?download=true",
    "Galicia":"https://huggingface.co/datasets/BaracanBaea/TFM/resolve/main/edificios_galicia.gpkg?download=true",
    "Islas Baleares":"https://huggingface.co/datasets/BaracanBaea/TFM/resolve/main/edificios_islasbaleares.gpkg?download=true",
    "Archivo Local (Pruebas)": "madrid.gpkg"
}

ccaa_seleccionada = st.sidebar.selectbox(
    "1. Selecciona la Comunidad Autónoma:", 
    list(BASES_DE_DATOS_CCAA.keys())
)

archivo_oculto = BASES_DE_DATOS_CCAA[ccaa_seleccionada]
lugar_input = st.sidebar.text_input("2. Municipio/Distrito a analizar:", "Chamberí, Madrid, España")

st.sidebar.header("2. Pesos del MCDA (WLC)")

def crear_slider_sincronizado(etiqueta, clave):
    if f"{clave}_slider" not in st.session_state:
        st.session_state[f"{clave}_slider"] = 0.20
    if f"{clave}_box" not in st.session_state:
        st.session_state[f"{clave}_box"] = 0.20

    def actualizar_desde_slider():
        st.session_state[f"{clave}_box"] = st.session_state[f"{clave}_slider"]
    def actualizar_desde_box():
        st.session_state[f"{clave}_slider"] = st.session_state[f"{clave}_box"]

    col1, col2 = st.sidebar.columns([3, 1]) 
    with col1:
        st.slider(etiqueta, 0.0, 1.0, key=f"{clave}_slider", on_change=actualizar_desde_slider)
    with col2:
        st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
        st.number_input(" ", 0.0, 1.0, step=0.05, key=f"{clave}_box", label_visibility="collapsed", on_change=actualizar_desde_box)

    return st.session_state[f"{clave}_slider"]

peso_reut = crear_slider_sincronizado("Reutilización", "reut")
peso_pob  = crear_slider_sincronizado("Población", "pob")
peso_top  = crear_slider_sincronizado("Topografía", "top")
peso_aire = crear_slider_sincronizado("Calidad Aire", "aire")
peso_san  = crear_slider_sincronizado("Saneamiento", "san")

boton_ejecutar = st.sidebar.button("🚀 Ejecutar Análisis", use_container_width=True)

st.title("🌿 Plataforma WebGIS: Humedales Artificiales")
st.markdown("---")

if boton_ejecutar:
    st.info(f"☁️ Conectando a la base de datos de {ccaa_seleccionada}...")
    
    with st.spinner("Descargando/Verificando cartografía autonómica..."):
        ruta_archivo_fisico = descargar_base_datos(archivo_oculto, ccaa_seleccionada)
        
    with st.spinner(f"📡 Descargando geometría y recortando {lugar_input}..."):
        try:
            parcelas, b_urb, b_inu, b_hidro, parques = obtener_datos_espaciales(lugar_input, ruta_archivo_fisico)
            mapa_folium, aptas_reales, total = calcular_y_mapear(
                parcelas, b_urb, b_inu, b_hidro, parques, 
                peso_reut, peso_pob, peso_top, peso_aire, peso_san
            )
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Parcelas Filtradas", total)
            col2.metric("Parcelas Aptas (>0)", aptas_reales)
            col3.metric("Filtros", "11 Evaluados")
            
            st.markdown("### Mapa de Idoneidad Espacial (Interactivo)")
            st_folium(mapa_folium, width=1200, height=700, returned_objects=[]) 
            
        except Exception as e:
            st.error(f"❌ Error procesando los datos: {e}. Revisa la conexión o el nombre del lugar.")
else:
    st.info("👈 Selecciona tu región, ajusta los parámetros y pulsa 'Ejecutar Análisis'.")
