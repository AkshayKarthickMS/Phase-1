import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import os
import data_utils
from scipy.spatial import distance
import numpy as np

# ---------- CONFIG ----------
# Real-world coordinates for network mapping (Lat, Long)
SETTLEMENT_COORDS = {
    # Gabasawa (partial)
    "malamawa": (12.26146, 8.25433),
    "santsi chikin gari": (11.99068, 8.54526),
    "wasarde": (12.03696, 8.86530),
    "hunbunare": (10.26729, 12.47183),
    "gagarawa": (12.40853, 9.52885),
    "mekiya gabas": (12.23550, 8.87346),
    "magama": (12.20037, 8.92292),
    "wadugur": (12.11098, 9.02575),
    "garin malam": (11.68375, 8.37204),
    "joda": (11.99896, 8.87447),
    "kurukuru": (13.55160, 6.01388),
    "kadage": (11.02315, 7.45228),
    "daurawa": (11.65416, 8.16492),
    "sabon gari": (12.01787, 8.53577),
    "gunduwa": (12.02285, 8.63746),
    "shargalle": (12.96121, 8.10400),
    "birgima": (11.95955, 7.37355),
    "yautar arewa": (12.26954, 8.75665),
    "cikin garin yauta": (10.72794, 7.93235),
    "badawa": (12.01379, 8.56845),
    "mazan gudu": (12.19529, 8.90722),
    "takalmawa": (12.18953, 8.85239),
    "doga": (12.21226, 8.79709),
    "badage": (12.27130, 9.14211),
    "tumbau": (11.99447, 8.81054),
    "unguwar gara": (11.94637, 8.55530),
    "daneji": (11.56536, 7.85565),
    "garin danga": (12.21088, 8.84309),
    "odoji": (7.11088, 5.07387),
    # Kiru
    "gidan danfadama": (13.05554, 5.17737),
    "unguwar liman": (10.01275, 9.78147),
    "tashar yanharawa": (11.96639, 7.74656),
    "makera": (10.46186, 7.39710),
    "tsohon gari": (11.25139, 8.39961),
    "maraku cikin gari": (10.64736, 8.68824),
    "unguwar maishuni": (11.55724, 8.99631),
    "gidan makama": (11.98870, 8.52104),
    "makera kofar gabas": (12.98455, 7.58892),
    "sarkakiya": (12.12741, 8.33788),
    "kwangwaro dutse": (11.49153, 9.58014),
    # Ungogo
    "jangaru": (11.49168, 8.58721),
    "rimi gata": (10.52769, 6.93541),
    "yar aduwa": (9.06752, 7.48674),
    "kududdufawa": (12.04614, 8.44683),
    "rijiyar zaki": (12.02836, 8.45240),
    "tsamiyar tazarce": (12.23450, 8.51193),
    "dausayi/rijiyar dinya": (12.05110, 8.47506),
    "rimin zakara": (12.00037, 8.44629),
    "muntsira": (11.97560, 8.43996),
    "kadawa?gabas": (11.64658, 8.44733),
    "gidan gona": (10.49585, 4.90003),
    "zangon marikita": (12.23481, 8.41515),
}

# ---------- CACHED DATA PREP ----------
@st.cache_resource
def get_additional_dashboard(selected_lgas=None, start_date=None, end_date=None, selected_genders=None, zerodose_path=None, visit_path=None):
    """
    Load data, filter, and generate figures for the Additional Analytics Dashboard.
    FOCUSED ON: Spatial Density Analysis (Restored to original visuals).
    """
    # Hardcoded absolute paths for specific environment
    ABS_FACILITY_PATH = r"C:\Users\AkshayKarthickMS\Desktop\Phase-1-main\data\facility_visit.csv"
    
    # Prioritize loading settlement.csv if it exists, as it contains the necessary Settlement column
    SETTLEMENT_PATH = data_utils.find_data_file("settlement.csv")
    ABS_ZERODOSE_PATH = r"C:\Users\AkshayKarthickMS\Desktop\Phase-1-main\data\zerodose.csv"
    
    # Default paths
    FACILITY_PATH = visit_path or ABS_FACILITY_PATH
    
    # Prioritize Settlement data if available, otherwise use default zerodose path
    if SETTLEMENT_PATH:
        ZERODOSE_PATH = SETTLEMENT_PATH
    else:
        ZERODOSE_PATH = zerodose_path or ABS_ZERODOSE_PATH
        
    # Robust Path Checking for the determined ZD path
    if not os.path.exists(FACILITY_PATH) and os.path.exists(ABS_FACILITY_PATH):
        FACILITY_PATH = ABS_FACILITY_PATH
    if not os.path.exists(ZERODOSE_PATH) and os.path.exists(ABS_ZERODOSE_PATH):
        ZERODOSE_PATH = ABS_ZERODOSE_PATH


    # Verify existence
    missing = [p for p in [FACILITY_PATH, ZERODOSE_PATH] if not os.path.exists(p)]
    if missing:
        st.error(f"Missing input files: {', '.join(missing)}")
        return [], {}

    # Load Data (FIXED: Robust Encoding)
    try:
        # Load Zero-Dose/Settlement Data
        if ZERODOSE_PATH.endswith('.xlsx'):
            zd_df = pd.read_excel(ZERODOSE_PATH, dtype=str).fillna('')
        else:
            try:
                zd_df = pd.read_csv(ZERODOSE_PATH, dtype=str).fillna('')
            except UnicodeDecodeError:
                # Fallback 1: Latin-1 encoding
                try:
                    zd_df = pd.read_csv(ZERODOSE_PATH, dtype=str, encoding='latin-1').fillna('')
                except:
                    # Fallback 2: Windows-1252 encoding
                    zd_df = pd.read_csv(ZERODOSE_PATH, dtype=str, encoding='cp1252').fillna('')


        # Load Facility
        if FACILITY_PATH.endswith('.xlsx'):
            fac_df = pd.read_excel(FACILITY_PATH, dtype=str).fillna('')
        else:
            try:
                fac_df = pd.read_csv(FACILITY_PATH, dtype=str).fillna('')
            except UnicodeDecodeError:
                # Fallback 1: Latin-1 encoding
                try:
                    fac_df = pd.read_csv(FACILITY_PATH, dtype=str, encoding='latin-1').fillna('')
                except:
                    # Fallback 2: Windows-1252 encoding
                    fac_df = pd.read_csv(FACILITY_PATH, dtype=str, encoding='cp1252').fillna('')
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return [], {}


    # --- PREPROCESSING & FILTERING ---
    
    # ZD Data Prep
    zd_df.columns = zd_df.columns.str.strip().str.lower()
    col_map = {
        'lga_name': 'LGA', 'lga': 'LGA',
        'resolution status': 'Status', 'status': 'Status',
        'distance to hf': 'Distance', 'distance_to_hf': 'Distance',
        'tracing_outcome': 'Outcome'
    }
    zd_df = zd_df.rename(columns={k: v for k, v in col_map.items() if k in zd_df.columns})
    
    # Standardize LGA & Gender
    lga_col_zd = next((c for c in ['LGA', 'lga_name'] if c in zd_df.columns), None)
    if lga_col_zd: zd_df['LGA'] = zd_df[lga_col_zd]
    else: zd_df['LGA'] = 'Unknown'

    g_col = 'Gender' if 'Gender' in zd_df.columns else 'gender'
    zd_df['Gender'] = zd_df.get(g_col, 'unknown').astype(str).str.lower().str.strip()
    
    # Filter ZD Data
    date_col_zd = next((c for c in ['enrollment date', 'visit_date'] if c in zd_df.columns), None)
    zd_df = data_utils.filter_data(
        zd_df, selected_lgas, start_date, end_date, date_col_zd, selected_genders
    )

    # Facility Data Prep
    fac_df.columns = fac_df.columns.str.strip().str.lower()
    lga_col_vis = next((c for c in ['lga_name', 'lga'] if c in fac_df.columns), None)
    if lga_col_vis: fac_df['LGA'] = fac_df[lga_col_vis]
    else: fac_df['LGA'] = 'Unknown'
    fac_df['Gender'] = fac_df.get('gender', 'unknown').astype(str).str.lower().str.strip()

    # Filter Facility Data
    date_col_vis = next((c for c in ['visit_date', 'date'] if c in fac_df.columns), None)
    fac_df = data_utils.filter_data(
        fac_df, selected_lgas, start_date, end_date, date_col_vis, selected_genders
    )

    if zd_df.empty:
        return [], {"total_cases": 0}

    # --- 4. AGGREGATION & FEATURE ENGINEERING ---
    
    # A. ZD Location Counts
    settlement_col = next((c for c in ['settlement', 'settlement_name'] if c in zd_df.columns), None)
    if settlement_col:
        zd_df['Settlement'] = zd_df[settlement_col].fillna("Unknown")
    else:
        zd_df['Settlement'] = "Unknown"
        
    id_col = 'id' if 'id' in zd_df.columns else next((c for c in zd_df.columns if 'id' in c), 'LGA')
    
    # Use id_col to count distinct cases
    ward_counts = zd_df.groupby(['LGA', 'Settlement'])[id_col].count().reset_index(name='ZeroDoseCount')

    # B. Service Volume (Facility) - Only needed for Service Gap (Removed)
    # C. Distance & Status - Only needed for Scatter (Removed)
    # D. Tracing Outcomes - Only needed for Tracing Breakdown (Removed)


    # --- 5. BUILD FIGURES ---
    px.defaults.template = "plotly_white"
    figs = []

    # FIG 1: Ward-wise Density Heatmap (Original Visual)
    if not ward_counts.empty:
        fig_heatmap = px.density_heatmap(
            ward_counts,
            x="Settlement",
            y="LGA",
            z="ZeroDoseCount",
            color_continuous_scale="Reds",
            title="Density Heatmap: Zero-Dose Burden by Location",
            text_auto=True
        )
        fig_heatmap.update_layout(coloraxis_colorbar_title="Count")
        figs.append(("Ward-wise Density Heatmap", fig_heatmap))

    # FIG 2: Geospatial Network Graph (Original Visual)
    
    # Group ZD data by settlement to get sizes
    settlement_sizes = zd_df['Settlement'].value_counts()
    
    # Identify nodes with coordinates
    nodes = []
    for settlement_name, size in settlement_sizes.items():
        clean_name = str(settlement_name).lower().strip()
        if clean_name in SETTLEMENT_COORDS:
            lat, lon = SETTLEMENT_COORDS[clean_name]
            nodes.append({
                "id": settlement_name,
                "lat": lat,
                "lon": lon,
                "size": size
            })
            
    if len(nodes) > 1:
        # Build DataFrame for easy distance calculation
        node_df = pd.DataFrame(nodes)
        
        # Create Geo Network
        edge_x, edge_y = [], []
        coords = node_df[['lat', 'lon']].values
        dist_matrix = distance.cdist(coords, coords, 'euclidean')
        
        for i in range(len(nodes)):
            nearest_indices = np.argsort(dist_matrix[i])[1:3]
            for neighbor_idx in nearest_indices:
                p1 = node_df.iloc[i]
                p2 = node_df.iloc[neighbor_idx]
                
                # Simple distance threshold to prevent cross-map lines
                if dist_matrix[i][neighbor_idx] < 0.5: 
                    edge_x.extend([p1['lon'], p2['lon'], None])
                    edge_y.extend([p1['lat'], p2['lat'], None])

        edge_trace = go.Scattermapbox(lat=edge_y, lon=edge_x, mode='lines', line=dict(width=1, color='#888'), hoverinfo='none')
        node_trace = go.Scattermapbox(
            lat=node_df['lat'], lon=node_df['lon'], mode='markers+text',
            text=[f"{row['id']}<br>Cases: {row['size']}" for _, row in node_df.iterrows()],
            marker=go.scattermapbox.Marker(size=[min(s * 3 + 5, 30) for s in node_df['size']], color='#EF553B', opacity=0.9),
            hoverinfo='text'
        )
        center_lat, center_lon = node_df['lat'].mean(), node_df['lon'].mean()

        fig_geo = go.Figure(data=[edge_trace, node_trace])
        fig_geo.update_layout(
            title='Geospatial Settlement Network (Proximity-Based)', mapbox_style="open-street-map",
            mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=9),
            margin={"r":0,"t":40,"l":0,"b":0}
        )
        figs.append(("Geospatial Network Graph", fig_geo))

    summary = {
        "total_cases": len(zd_df),
        "locations_mapped": len(ward_counts),
        "mapped_with_coords": len(nodes),
        "top_hotspot": ward_counts.sort_values('ZeroDoseCount', ascending=False).iloc[0]['Settlement'] if not ward_counts.empty else "N/A"
    }

    return figs, summary

# ---------- DASHBOARD RENDER ----------
def render_additional_dashboard(selected_lgas=None, start_date=None, end_date=None, selected_genders=None, 
                                zerodose_path=None, visit_path=None, precomputed_data=None, chart_callback=None):
    
    # 1. Get Data
    if precomputed_data:
        figs, summary = precomputed_data
    else:
        try:
            figs, summary = get_additional_dashboard(
                selected_lgas, start_date, end_date, selected_genders,
                zerodose_path, visit_path
            )
        except Exception as e:
            st.error(f"Error loading dashboard: {e}")
            return

    if not figs:
        st.warning("No data found matching filters.")
        return

    # 2. KPIs
    st.markdown("### 🧭 Spatial Density & Connectivity")
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Cases", f"{summary.get('total_cases',0):,}", help="Zero-dose cases in filtered selection")
    k2.metric("Settlements w/ Coords", f"{summary.get('mapped_with_coords',0)}/{summary.get('locations_mapped',0)}", help="Settlements successfully mapped to lat/long")
    k3.metric("Primary Hotspot", f"{summary.get('top_hotspot','N/A')}", help="Settlement with highest zero-dose count")

    st.divider()

    # 3. Helper for rendering
    def display(fig, title, idx):
        if chart_callback:
            chart_callback(fig, title, f"add_prof_{idx}")
        else:
            st.markdown(f"**{title}**")
            st.plotly_chart(fig, use_container_width=True)

    fig_dict = {title: fig for title, fig in figs}

    # 4. Grid Layout (Restored Original Layout)
    
    # ROW 1: Geospatial Map (Priority)
    if "Geospatial Network Graph" in fig_dict:
        st.markdown("##### Settlement Clusters (Map View)")
        display(fig_dict["Geospatial Network Graph"], "Geospatial Network Graph", 0)
    else:
        # Fallback message if map cannot be drawn
        if summary.get('mapped_with_coords', 0) == 0:
            st.info("ℹ️ Map available only for settlements with known coordinates (Gabasawa/Kiru/Ungogo).")

    # ROW 2: Heatmap
    st.markdown("---")
    st.markdown("##### Zero-Dose Burden Intensity")
    if "Ward-wise Density Heatmap" in fig_dict:
        display(fig_dict["Ward-wise Density Heatmap"], "Ward-wise Density Heatmap", 1)

# ---------- COMPATIBILITY ALIASES ----------
# These prevent ImportError in __init__.py
def show_additional_dashboard():
    """Alias for render_additional_dashboard to support legacy imports."""
    render_additional_dashboard()

def build_figures():
    """Alias for legacy build_figures, returns empty placeholders to satisfy imports."""
    return go.Figure(), go.Figure(), go.Figure()