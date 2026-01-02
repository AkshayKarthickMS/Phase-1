import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import os
import data_utils

# ---------- CONFIG ----------
LONG_ENROLL_DAYS = 180

# ---------- HELPER: FILE FINDER ----------
def find_file(provided_path, default_candidates):
    """
    Robustly finds a file, checking provided path, defaults, and extension swapping.
    """
    candidates = []
    if provided_path:
        candidates.append(provided_path)
        # Try swapping extension
        root, ext = os.path.splitext(provided_path)
        if ext == '.csv': candidates.append(root + '.xlsx')
        elif ext == '.xlsx': candidates.append(root + '.csv')
        
    candidates.extend(default_candidates)
    
    for path in candidates:
        if os.path.exists(path):
            return path
            
    return None

# ---------- CACHED DATA PREP ----------
@st.cache_resource
def load_and_process_data(selected_lgas=None, start_date=None, end_date=None, selected_genders=None, zerodose_path=None, visit_path=None):
    # 1. Resolve Paths
    fac_candidates = [
        r"C:\Users\AkshayKarthickMS\Desktop\Phase-1-main\data\facility_visits.csv",
        "data/facility_visits.csv", "facility_visits.csv",
        "data/facility_visit.csv", "facility_visit.csv"
    ]
    # Prioritize settlement.csv/zerodose.csv
    zd_candidates = [
        r"C:\Users\AkshayKarthickMS\Desktop\Phase-1-main\data\settlement.csv",
        "data/settlement.csv", "settlement.csv",
        r"C:\Users\AkshayKarthickMS\Desktop\Phase-1-main\data\zerodose.xlsx",
        "data/zerodose.xlsx", "zerodose.xlsx"
    ]

    FACILITY_PATH = find_file(visit_path, fac_candidates)
    ZERODOSE_PATH = find_file(zerodose_path, zd_candidates)

    if not ZERODOSE_PATH:
         st.error(f"Missing Zero-Dose/Settlement file. Searched: {zerodose_path} and defaults.")
         return pd.DataFrame()

    # 2. Load Data
    zd = pd.DataFrame() # Initialize to avoid UnboundLocalError
    try:
        # Load Zero-Dose / Settlement Data
        if ZERODOSE_PATH and ZERODOSE_PATH.endswith('.xlsx'):
            zd = pd.read_excel(ZERODOSE_PATH, dtype=str).fillna('')
        elif ZERODOSE_PATH:
            try:
                zd = pd.read_csv(ZERODOSE_PATH, dtype=str).fillna('')
            except UnicodeDecodeError:
                zd = pd.read_csv(ZERODOSE_PATH, dtype=str, encoding='latin1').fillna('')
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return {} # Return empty dict on critical failure for consistency

    # 3. Load Facility Data
    vis = pd.DataFrame() # Initialize
    try:
        if FACILITY_PATH:
            if FACILITY_PATH.endswith('.xlsx'):
                vis = pd.read_excel(FACILITY_PATH, dtype=str).fillna('')
            else:
                vis = pd.read_csv(FACILITY_PATH, dtype=str).fillna('')
    except Exception as e:
        # Non-critical, just continue with empty vis
        vis = pd.DataFrame()

    # --- PREPROCESSING & FILTERING ---
    date_col_zd = None # Initialize
    
    # Check if critical ZD data is loaded
    if zd.empty:
         # Check if we at least have facility data to make SOME chart
         if vis.empty:
             return {} 
    
    # 1. Zero-Dose / Settlement Data (zd)
    if not zd.empty:
        zd.columns = zd.columns.str.strip().str.lower()
        
        # Explicit Column Mapping for settlement.csv structure
        col_map = {
            'lga_name': 'LGA', 'lga': 'LGA',
            'resolution status': 'Status', 'status': 'Status',
            'enrollment date': 'Enrollment Date'
        }
        zd = zd.rename(columns={k: v for k, v in col_map.items() if k in zd.columns})
        
        # Standardize LGA
        lga_col_zd = next((c for c in ['LGA', 'lga_name'] if c in zd.columns), None)
        if lga_col_zd: zd['LGA'] = zd[lga_col_zd]
        else: zd['LGA'] = 'Unknown'

        # Standardize Gender
        g_col = next((c for c in ['gender', 'sex'] if c in zd.columns), None)
        if g_col: zd['Gender'] = zd[g_col].astype(str).str.lower().str.strip()
        else: zd['Gender'] = 'unknown'

        # --- STATUS STANDARDIZATION ---
        status_col = next((c for c in ['Status', 'resolution status'] if c in zd.columns), None)
        if status_col:
            # Convert to Title Case (Active, Resolved)
            zd['Status_std'] = zd[status_col].astype(str).str.strip().str.title()
            # Fix empty/nan values -> 'Active'
            zd['Status_std'] = zd['Status_std'].replace(['', 'Nan', 'None', 'Null'], 'Active')
            # Ensure only Active/Resolved exist
            zd['Status_std'] = zd['Status_std'].apply(lambda x: 'Resolved' if x == 'Resolved' else 'Active')
        else:
            zd['Status_std'] = 'Active'

        # Filter Data (Zero-Dose)
        date_col_zd = next((c for c in ['Enrollment Date', 'visit_date', 'date'] if c in zd.columns), None)
        zd = data_utils.filter_data(zd, selected_lgas, start_date, end_date, date_col_zd, selected_genders, include_na_dates=True)

    # Filter Data (Facility Visits)
    if not vis.empty:
        vis.columns = vis.columns.str.strip().str.lower()
        if 'lga' not in vis.columns and 'lga_name' in vis.columns: vis['lga'] = vis['lga_name']
        col_vis_date = 'visit_date' if 'visit_date' in vis.columns else None
        
        # Standardize Columns
        if 'vaccine_antigen' not in vis.columns: vis['vaccine_antigen'] = ''
        vis['vaccine_antigen'] = vis['vaccine_antigen'].astype(str).str.strip().str.title()
        
        if 'doses_given' not in vis.columns: vis['doses_given'] = 0
        vis['doses_given'] = pd.to_numeric(vis['doses_given'], errors='coerce').fillna(0)
        
        vis = data_utils.filter_data(vis, selected_lgas, start_date, end_date, col_vis_date, selected_genders)
        
        # Parse Dates
        if col_vis_date:
            vis['visit_dt'] = pd.to_datetime(vis[col_vis_date], errors='coerce')
            vis['visit_month'] = vis['visit_dt'].dt.to_period('M').astype(str)
            vis['visit_dow'] = vis['visit_dt'].dt.day_name()

    # --- AGGREGATION 1: ZD Enrollment Trends ---
    if not zd.empty and date_col_zd:
        # Fix UserWarning while preserving dayfirst=True for non-ISO dates (e.g. DD/MM/YYYY)
        # 1. Identify ISO dates (YYYY-MM-DD)
        iso_mask = zd[date_col_zd].astype(str).str.match(r'^\d{4}-\d{2}-\d{2}').fillna(False)
        
        # 2. Parse ISO (no dayfirst needed)
        zd.loc[iso_mask, 'Enrollment Date_parsed'] = pd.to_datetime(zd.loc[iso_mask, date_col_zd], errors='coerce')
        
        # 3. Parse non-ISO (likely DD/MM/YYYY) with dayfirst=True
        zd.loc[~iso_mask, 'Enrollment Date_parsed'] = pd.to_datetime(zd.loc[~iso_mask, date_col_zd], dayfirst=True, errors='coerce')

        zd = zd.dropna(subset=['Enrollment Date_parsed'])
        zd['enroll_month'] = zd['Enrollment Date_parsed'].dt.to_period('M').astype(str)
    elif not zd.empty:
        # Fallback if no date column but data exists
        zd['Enrollment Date_parsed'] = pd.to_datetime('today')
        zd['enroll_month'] = 'Unknown'
    else:
        zd['Enrollment Date_parsed'] = pd.to_datetime('today')
        zd['enroll_month'] = 'Unknown'

    if 'enroll_month' in zd.columns:
        enroll_timeline = zd.groupby(['enroll_month', 'Status_std']).size().reset_index(name='count')
        enroll_pivot = enroll_timeline.pivot(index='enroll_month', columns='Status_std', values='count').fillna(0).reset_index()
        for status in ['Active', 'Resolved']:
            if status not in enroll_pivot.columns: enroll_pivot[status] = 0
        
        enroll_pivot['enroll_month_dt'] = pd.to_datetime(enroll_pivot['enroll_month'], format='%Y-%m', errors='coerce')
        enroll_pivot = enroll_pivot.sort_values('enroll_month_dt')
    else:
        enroll_pivot = pd.DataFrame()

    # --- AGGREGATION 2: Antigen Trends (Monthly) ---
    antigen_trends = pd.DataFrame()
    if not vis.empty and 'visit_month' in vis.columns:
        # Focus on Key Antigens
        target_antigens = ['Bcg', 'Penta 1', 'Penta 3', 'Measles 1']
        vis_filt = vis[vis['vaccine_antigen'].isin(target_antigens)]
        
        antigen_trends = vis_filt.groupby(['visit_month', 'vaccine_antigen'])['doses_given'].sum().reset_index()
        antigen_trends['month_dt'] = pd.to_datetime(antigen_trends['visit_month'], format='%Y-%m', errors='coerce')
        antigen_trends = antigen_trends.sort_values('month_dt')

    # --- AGGREGATION 3: Dropout Trends (Penta1 - Penta3 Gap) ---
    dropout_trends = pd.DataFrame()
    if not antigen_trends.empty:
        piv = antigen_trends.pivot(index='visit_month', columns='vaccine_antigen', values='doses_given').fillna(0)
        if 'Penta 1' in piv.columns and 'Penta 3' in piv.columns:
            piv['Dropout Gap'] = piv['Penta 1'] - piv['Penta 3']
            piv['Dropout Rate %'] = (piv['Dropout Gap'] / piv['Penta 1'].replace(0, 1)) * 100
            dropout_trends = piv.reset_index()
            dropout_trends['month_dt'] = pd.to_datetime(dropout_trends['visit_month'], format='%Y-%m')
            dropout_trends = dropout_trends.sort_values('month_dt')

    # --- AGGREGATION 4: Seasonality (Day of Week) ---
    seasonality = pd.DataFrame()
    if not vis.empty and 'visit_dow' in vis.columns:
        # Sort order for DOW
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        vis['visit_dow'] = pd.Categorical(vis['visit_dow'], categories=dow_order, ordered=True)
        seasonality = vis.groupby('visit_dow', observed=False)['doses_given'].sum().reset_index()

    # --- AGGREGATION 5: Demand Forecasting (Next Appointment) ---
    forecast_df = pd.DataFrame()
    if not vis.empty and 'next_appointment_date' in vis.columns:
        vis['next_apt_dt'] = pd.to_datetime(vis['next_appointment_date'], errors='coerce')
        forecast_agg = vis.dropna(subset=['next_apt_dt']).groupby(vis['next_apt_dt'].dt.to_period('M').astype(str))['doses_given'].count().reset_index(name='Expected Demand')
        forecast_agg = forecast_agg.rename(columns={'next_apt_dt': 'month'})
        
        # Compare with Actuals (visit_date)
        actuals_agg = vis.groupby('visit_month')['doses_given'].count().reset_index(name='Actual Visits').rename(columns={'visit_month': 'month'})
        
        forecast_df = pd.merge(forecast_agg, actuals_agg, on='month', how='outer').fillna(0)
        forecast_df['month_dt'] = pd.to_datetime(forecast_df['month'], format='%Y-%m')
        forecast_df = forecast_df.sort_values('month_dt')

    return {
        "enrollment": enroll_pivot,
        "antigens": antigen_trends,
        "dropouts": dropout_trends,
        "seasonality": seasonality,
        "forecast": forecast_df
    }


# ---------- CACHED VISUALS ----------
@st.cache_resource(show_spinner=False)
def build_figures(data_dict):
    figs = []
    
    # 1. Enrollment Trends (Existing)
    enroll_pivot = data_dict.get('enrollment', pd.DataFrame())
    if not enroll_pivot.empty:
        status_cols = [c for c in ['Active', 'Resolved'] if c in enroll_pivot.columns]
        fig_enroll = px.area(
            enroll_pivot, x='enroll_month', y=status_cols,
            title='Zero-Dose Enrollment Trends (Active vs Resolved)',
            labels={'value': 'Number of Children', 'enroll_month': 'Month', 'variable': 'Status'},
            color_discrete_map={'Active': '#636EFA', 'Resolved': '#EF553B'}
        )
        figs.append(("Zero-dose Enrollment Trend", fig_enroll))

    # 2. Antigen Consumption Trends (New)
    ag_df = data_dict.get('antigens', pd.DataFrame())
    if not ag_df.empty:
        fig_ag = px.line(
            ag_df, x='visit_month', y='doses_given', color='vaccine_antigen',
            title='Monthly Antigen Consumption Trends',
            markers=True,
            labels={'doses_given': 'Doses Administered', 'visit_month': 'Month', 'vaccine_antigen': 'Antigen'}
        )
        figs.append(("Antigen Trends", fig_ag))

    # 3. Dropout Gap Analysis (New)
    drop_df = data_dict.get('dropouts', pd.DataFrame())
    if not drop_df.empty:
        # Dual Axis? Or just show Rate? Let's show Volume Gap and Rate as text
        fig_drop = px.bar(
            drop_df, x='visit_month', y='Dropout Gap',
            title='Penta 1 - Penta 3 Dropout Volume (Monthly Gap)',
            text='Dropout Rate %',
            labels={'Dropout Gap': 'Gap (Penta1 - Penta3)', 'visit_month': 'Month'}
        )
        fig_drop.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_drop.update_layout(yaxis_title="Dose Gap (Count)")
        figs.append(("Dropout Gap Analysis", fig_drop))

    # 4. Seasonality (Day of Week) (New)
    seas_df = data_dict.get('seasonality', pd.DataFrame())
    if not seas_df.empty:
        fig_seas = px.bar(
            seas_df, x='visit_dow', y='doses_given',
            title='Operational Volume by Day of Week',
            color='doses_given', color_continuous_scale='Viridis',
            labels={'visit_dow': 'Day of Week', 'doses_given': 'Total Doses'}
        )
        figs.append(("Weekly Seasonality", fig_seas))

    # 5. Forecasting (Demand vs Actuals) (New)
    fore_df = data_dict.get('forecast', pd.DataFrame())
    if not fore_df.empty:
        fig_fore = go.Figure()
        fig_fore.add_trace(go.Scatter(
            x=fore_df['month'], y=fore_df['Actual Visits'], name='Actual Visits',
            mode='lines+markers', line=dict(color='blue')
        ))
        fig_fore.add_trace(go.Bar(
            x=fore_df['month'], y=fore_df['Expected Demand'], name='Forecast Demand (Next Apt)',
            marker_color='rgba(255, 165, 0, 0.5)'
        ))
        fig_fore.update_layout(title="Demand Forecasting: Actual vs Scheduled Appointments", xaxis_title="Month", yaxis_title="Volume")
        figs.append(("Demand Forecast", fig_fore))

    return figs


# ---------- DASHBOARD RENDER ----------
def render_timeseries_dashboard(selected_lgas=None, start_date=None, end_date=None, selected_genders=None, 
                                zerodose_path=None, visit_path=None, precomputed_data=None, chart_callback=None):
    
    st.header("📈 Time-Series & Follow-up Dashboard")
    st.caption(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load Data with Filters
    # Load Data with Filters
    if precomputed_data:
        data_dict = precomputed_data
    else:
        data_dict = load_and_process_data(
            selected_lgas, start_date, end_date, selected_genders, zerodose_path, visit_path
        )

    # Check if empty (check enrollment or antigens as proxies)
    if data_dict.get('enrollment', pd.DataFrame()).empty and data_dict.get('antigens', pd.DataFrame()).empty:
        st.warning("No data found matching filters.")
        return

    figs = build_figures(data_dict)
    
    # Helper to display charts
    def display(fig, title, idx):
         if chart_callback: chart_callback(fig, title, f"ts_chart_{idx}")
         else: st.plotly_chart(fig, use_container_width=True)

    fig_dict = {title: fig for title, fig in figs}
    
    # Layout Strategy
    # Row 1: Enrollment & Antigens (Only Enrollment kept)
    # c1, c2 = st.columns(2)
    # with c1:
    if "Zero-dose Enrollment Trend" in fig_dict: display(fig_dict["Zero-dose Enrollment Trend"], "Enrollment Trends", 1)
    # with c2:
    #     if "Antigen Trends" in fig_dict: display(fig_dict["Antigen Trends"], "Antigen Consumption", 2)
        
    # Row 2: Dropouts & Seasonality (REMOVED)
    # st.divider()
    
    # Row 3: Forecasting (REMOVED)

# Helper alias for legacy calls
def get_timeseries_dashboard(selected_lgas=None, start_date=None, end_date=None, selected_genders=None, zerodose_path=None, visit_path=None):
    return load_and_process_data(selected_lgas, start_date, end_date, selected_genders, zerodose_path, visit_path)

if __name__ == "__main__":
    render_timeseries_dashboard()