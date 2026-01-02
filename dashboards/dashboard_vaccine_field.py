import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import os
import data_utils

# ---------------------------
# Vaccine Schedule Definition
# ---------------------------
VACCINE_SCHEDULE = {
    "At Birth": ["BCG", "OPV_0", "HepB_0"],
    "6 Weeks": ["Penta_1", "PCV_1", "OPV_1", "Rota_1", "IPV_1"],
    "10 Weeks": ["Penta_2", "PCV_2", "OPV_2", "Rota_2"],
    "14 Weeks": ["Penta_3", "PCV_3", "OPV_3", "Rota_3", "IPV_2"],
    "6 Months": ["VitaminA_1"],
    "9 Months": ["Measles_1", "YF", "meningitis"],
    "12 Months": ["VitaminA_2"],
    "15 Months": ["Measles_2"],
    "9-13 Years": ["HPV"]
}

# Flatten the schedule for easier column checking
ALL_VACCINES = [v for sublist in VACCINE_SCHEDULE.values() for v in sublist]

# ---------------------------
# Cached heavy computation
# ---------------------------
@st.cache_resource
def get_vaccine_dashboard(selected_lgas=None, start_date=None, end_date=None, selected_genders=None, zerodose_path=None, visit_path=None):
    """
    Load data, process metrics, and generate refined decision-support figures.
    Integrates Facility Visits, Zero-Dose List, and Settlement Data.
    """
    # Hardcoded absolute paths for specific environment
    ABS_FACILITY_PATH = r"C:\Users\AkshayKarthickMS\Desktop\Phase-1-main\data\facility_visits.csv"
    ABS_ZERODOSE_PATH = r"C:\Users\AkshayKarthickMS\Desktop\Phase-1-main\data\zerodose.xlsx"
    ABS_SETTLEMENT_PATH = r"C:\Users\AkshayKarthickMS\Desktop\Phase-1-main\data\settlement.csv"

    # Default paths if not provided
    FACILITY_PATH = data_utils.find_data_file(visit_path or "facility_visits.csv")
    ZERODOSE_PATH = data_utils.find_data_file(zerodose_path or "zerodose.xlsx")
    SETTLEMENT_PATH = data_utils.find_data_file("settlement.csv")

    # Verify existence
    if not FACILITY_PATH and os.path.exists(ABS_FACILITY_PATH): FACILITY_PATH = ABS_FACILITY_PATH
    if not ZERODOSE_PATH and os.path.exists(ABS_ZERODOSE_PATH): ZERODOSE_PATH = ABS_ZERODOSE_PATH
    
    # Load Data
    try:
        fac_df = pd.read_csv(FACILITY_PATH, dtype=str).fillna('') if FACILITY_PATH else pd.DataFrame()
    except: fac_df = pd.DataFrame()

    try:
        if ZERODOSE_PATH and ZERODOSE_PATH.endswith('.csv'):
            zd_df = pd.read_csv(ZERODOSE_PATH, dtype=str).fillna('')
        elif ZERODOSE_PATH:
            zd_df = pd.read_excel(ZERODOSE_PATH, dtype=str).fillna('')
        else:
            zd_df = pd.DataFrame()
    except: zd_df = pd.DataFrame()

    try:
        if SETTLEMENT_PATH and SETTLEMENT_PATH.endswith('.csv'):
            sett_df = pd.read_csv(SETTLEMENT_PATH, dtype=str).fillna('') 
        else: 
            sett_df = pd.DataFrame()
    except: sett_df = pd.DataFrame()

    # ==========================================
    # 1. PREPROCESS FACILITY DATA (Coverage)
    # ==========================================
    fac_df.columns = fac_df.columns.str.strip().str.lower()
    if 'lga_name' in fac_df.columns: fac_df['LGA'] = fac_df['lga_name']
    else: fac_df['LGA'] = 'Unknown'
    
    if 'health_center_name' not in fac_df.columns: fac_df['health_center_name'] = 'Unknown'

    fac_df = data_utils.filter_data(
        fac_df, selected_lgas, start_date, end_date, 'visit_date', selected_genders
    )
    fac_df = data_utils.apply_client_age_categories(fac_df, source_type='facility')
    fac_df['age_group'] = fac_df['Client_Age_Group']

    # Detect Vaccines
    vaccine_cols = [v for v in ALL_VACCINES if v in fac_df.columns]
    if len(vaccine_cols) == 0 and 'vaccines_administered' in fac_df.columns:
        parsed = fac_df['vaccines_administered'].astype(str).str.replace(r'[\{\}\"\'\[\]]','', regex=True).str.split(r'\s*,\s*')
        all_mentioned = set([item for sublist in parsed for item in sublist if item])
        vaccine_cols = sorted([v for v in all_mentioned if v in ALL_VACCINES])
        for v in vaccine_cols:
            fac_df[v] = parsed.apply(lambda L: 1 if isinstance(L, list) and v in L else 0)

    # ==========================================
    # 2. PREPROCESS ZERO-DOSE / SETTLEMENT (Burden)
    # ==========================================
    # Use settlement data as primary for "Hotspots" if available, else ZD
    if not sett_df.empty:
        burden_df = sett_df.copy()
    else:
        burden_df = zd_df.copy()

    burden_df.columns = burden_df.columns.str.strip().str.lower()
    col_map = {'lga_name': 'LGA', 'lga': 'LGA', 'resolution status': 'Status', 'status':'Status', 'reasons_for_zero_dose': 'Reason for ZD', 'reason for zd': 'Reason for ZD', 'distance to': 'Distance', 'distance to hf': 'Distance'}
    burden_df = burden_df.rename(columns={k:v for k,v in col_map.items() if k in burden_df.columns})
    if 'LGA' not in burden_df.columns: burden_df['LGA'] = 'Unknown'
    
    # Filter Burden
    date_col_b = next((c for c in ['enrollment date', 'visit_date'] if c in burden_df.columns), None)
    burden_df = data_utils.filter_data(
        burden_df, selected_lgas, start_date, end_date, date_col_b, selected_genders, include_na_dates=True
    )
    
    # Process Distance (Extract numeric)
    if 'Distance' in burden_df.columns:
        burden_df['dist_val'] = burden_df['Distance'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float).fillna(0)
    else:
        burden_df['dist_val'] = 0.0

    # Ensure Status is clean
    if 'Status' in burden_df.columns:
        burden_df['is_resolved'] = burden_df['Status'].astype(str).str.lower() == 'resolved'
    else:
        burden_df['is_resolved'] = False

    # ==========================================
    # 3. METRICS GENERATION
    # ==========================================
    
    # KPIs from FULL Zero-Dose List (for accuracy)
    # Ensure zd_df has status processed
    if not zd_df.empty:
        st_col = next((c for c in ['status', 'resolution status'] if c in zd_df.columns), None)
        if st_col:
            zd_df['is_resolved_calc'] = zd_df[st_col].astype(str).str.lower() == 'resolved'
        else:
            zd_df['is_resolved_calc'] = False
        
        # Filter zd_df consistent with dashboard filters (LGA/Gender) for KPI? 
        # Usually KPIs should reflect the filtered view.
        # Apply same filters as burden_df
        # Re-using data_utils.filter_data on zd_df
        d_col = next((c for c in ['enrollment date', 'visit_date'] if c in zd_df.columns), None)
        zd_filt = data_utils.filter_data(
            zd_df, selected_lgas, start_date, end_date, d_col, selected_genders, include_na_dates=True
        )
        
        total_zd_count = len(zd_filt)
        resolved_count = zd_filt['is_resolved_calc'].sum()
    else:
        total_zd_count = 0
        resolved_count = 0

    # A. Coverage
    if not fac_df.empty and vaccine_cols:
        coverage = pd.DataFrame({
            'vaccine': vaccine_cols,
            'doses_given': [int(fac_df[v].sum()) for v in vaccine_cols],
            'coverage_pct': [float(fac_df[v].mean()*100) for v in vaccine_cols]
        })
        # Sorting
        schedule_order = {v: i for i, v in enumerate(ALL_VACCINES)}
        coverage['order'] = coverage['vaccine'].map(schedule_order).fillna(999)
        coverage = coverage.sort_values('order')
        
        # Dropout Calculation (Penta1 vs Penta3)
        p1 = coverage[coverage['vaccine']=='Penta_1']['doses_given'].sum()
        p3 = coverage[coverage['vaccine']=='Penta_3']['doses_given'].sum()
        dropout_rate = ((p1 - p3) / p1 * 100) if p1 > 0 else 0
        
        # Target Age for Pie Chart
        def get_target_age(v_name):
            for age, v_list in VACCINE_SCHEDULE.items():
                if v_name in v_list: return age
            return "Unknown"
        coverage['Target Age'] = coverage['vaccine'].apply(get_target_age)
        
    else:
        coverage = pd.DataFrame()
        dropout_rate = 0

    # B. Settlement Hotspots (Active Burden)
    if not burden_df.empty:
        sett_col = next((c for c in ['settlement', 'settlement_name'] if c in burden_df.columns), None)
        if sett_col:
            active_burden = burden_df[~burden_df['is_resolved']]
            hotspots = active_burden.groupby([sett_col, 'LGA']).agg(
                Unresolved_Cases=('Status', 'count'),
                Avg_Distance=('dist_val', 'mean')
            ).reset_index().sort_values('Unresolved_Cases', ascending=False)
        else:
            hotspots = pd.DataFrame()
    else:
        hotspots = pd.DataFrame()

    # ==========================================
    # 4. BUILD FIGURES
    # ==========================================
    px.defaults.template = "plotly_white"
    figs = []

    # --- EXISTING VISUALIZATIONS (Preserved & Enhanced) ---

    # 1. Coverage by Antigen
    if not coverage.empty:
        fig_cov = px.bar(
            coverage, x='vaccine', y='coverage_pct',
            text=coverage['coverage_pct'].map(lambda x: f"{x:.1f}%"),
            title="Vaccine Coverage by Antigen",
            labels={'coverage_pct': 'Coverage (%)'},
            color_discrete_sequence=['#636EFA']
        )
        fig_cov.update_layout(yaxis=dict(range=[0, 110]))
        figs.append(("Coverage by Antigen", fig_cov))

    # 2. Gender (Preserved logic but ensuring display)
    if not fac_df.empty and vaccine_cols:
        cov_gender = fac_df.groupby('gender')[vaccine_cols].mean().T * 100
        cov_gender = cov_gender.reset_index().melt(id_vars='index', var_name='Gender', value_name='Coverage')
        fig_gen = px.bar(
            cov_gender, x='index', y='Coverage', color='Gender', barmode='group',
            title="Gender Equity in Coverage",
            color_discrete_map={'male': '#636EFA', 'female': '#EF553B'}
        )
        figs.append(("Coverage by Gender", fig_gen))

    # 3. Timeliness (Heatmap) - ENHANCED SIZE
    if not fac_df.empty and 'age_group' in fac_df.columns and vaccine_cols:
        cov_age = fac_df.groupby('age_group', observed=False)[vaccine_cols].mean() * 100
        fig_heat = px.imshow(
            cov_age.T, text_auto='.1f', 
            color_continuous_scale='YlGnBu', 
            title="Timeliness: Coverage by Age Group",
            aspect='auto', height=500
        )
        figs.append(("Coverage by Age Group", fig_heat))

    # 4. LGA Heatmap
    if not fac_df.empty and 'LGA' in fac_df.columns and vaccine_cols:
        cov_lga = fac_df.groupby('LGA')[vaccine_cols].mean() * 100
        fig_lga = px.imshow(cov_lga, text_auto='.0f', color_continuous_scale='OrRd', title="Coverage Intensity by LGA")
        figs.append(("Coverage by LGA", fig_lga))

    # 5. Coverage vs Burden (Action Matrix) - Kept in calculation, but removed from display as per previous request?
    # User said "REMOVE STRATEGIC OVERVIEW". This was in Strategic Overview.
    # We'll just skip adding it to figs if we don't plan to use it, or keep it in case.
    # skipping.
    
    # ... (Skipping removed visuals logic instructions) ...

    # 8. Dropout Summary
    if not coverage.empty:
        funnel_vax = ['BCG', 'Penta_1', 'Penta_3', 'Measles_1']
        funnel_df = coverage[coverage['vaccine'].isin(funnel_vax)].set_index('vaccine').reindex(funnel_vax).reset_index()
        fig_drop = px.funnel(
            funnel_df, x='doses_given', y='vaccine', 
            title=f"Dropout Funnel (Rate: {dropout_rate:.1f}%)"
        )
        figs.append(("Dropout Funnel", fig_drop))
        
    # 9. Drivers
    if not burden_df.empty and 'Reason for ZD' in burden_df.columns:
        reasons = burden_df['Reason for ZD'].value_counts().reset_index()
        reasons.columns = ['Reason', 'Count']
        fig_reas = px.pie(reasons, values='Count', names='Reason', title="Drivers of Zero-Dose", hole=0.4)
        figs.append(("Zero-Dose Drivers", fig_reas))

    # 10. Doses Administered (Volume)
    if not coverage.empty:
        fig_vol = px.bar(
            coverage, x='vaccine', y='doses_given',
            text='doses_given',
            title="Total Doses Administered (Volume)",
            labels={'doses_given': 'Number of Doses'},
            color='doses_given',
            color_continuous_scale='Viridis'
        )
        fig_vol.update_layout(yaxis=dict(showgrid=False))
        figs.append(("Total Doses Administered", fig_vol))

    # 11. Schedule Distribution
    if not coverage.empty:
        # Aggregate by Target Age
        sched_agg = coverage.groupby('Target Age')['doses_given'].sum().reset_index()
        fig_sched = px.pie(
            sched_agg, values='doses_given', names='Target Age',
            title="Doses by Schedule Stage",
            hole=0.3,
            color_discrete_sequence=px.colors.qualitative.Prism 
        )
        figs.append(("Schedule Distribution", fig_sched))

    # 13. Antigen Series Dropouts (Multiple Series)
    if not coverage.empty:
        series_pairs = [('Penta_1', 'Penta_3'), ('PCV_1', 'PCV_3'), ('Rota_1', 'Rota_3'), ('OPV_1', 'OPV_3')]
        dropout_data = []
        for start, end in series_pairs:
            s_val = coverage[coverage['vaccine']==start]['doses_given'].sum()
            e_val = coverage[coverage['vaccine']==end]['doses_given'].sum()
            rate = ((s_val - e_val) / s_val * 100) if s_val > 0 else 0
            dropout_data.append({'Series': f"{start}->{end}", 'Dropout Rate (%)': rate})
        
        if dropout_data:
            df_drop_series = pd.DataFrame(dropout_data)
            fig_drop_series = px.bar(
                df_drop_series, x='Series', y='Dropout Rate (%)',
                text=df_drop_series['Dropout Rate (%)'].map(lambda x: f"{x:.1f}%"),
                title="Dropout Rates by Antigua Series",
                color='Dropout Rate (%)', color_continuous_scale='Reds'
            )
            fig_drop_series.update_layout(yaxis=dict(range=[0, max(20, df_drop_series['Dropout Rate (%)'].max()+5)]))
            figs.append(("Series Dropouts", fig_drop_series))

    # 17. Zero-Dose Resolution Progress by LGA
    if not burden_df.empty:
        prog_df = burden_df.groupby(['LGA', 'is_resolved']).size().reset_index(name='Count')
        prog_df['Status'] = prog_df['is_resolved'].map({True: 'Resolved', False: 'Active'})
        fig_prog = px.bar(
            prog_df, x='LGA', y='Count', color='Status',
            title="Zero-Dose Resolution Progress by LGA",
            color_discrete_map={'Resolved': '#00CC96', 'Active': '#EF553B'},
            barmode='stack'
        )
        figs.append(("Resolution Progress", fig_prog))

    # Summary Dict
    summary = {
        "fac_visits": len(fac_df),
        "zd_burden": total_zd_count, # from ZD_DF full list
        "resolved_cases": resolved_count, # from ZD_DF full list
        "dropout_rate": dropout_rate,
        "hotspots": hotspots.head(10)
    }

    return figs, summary


# ---------------------------
# View Render
# ---------------------------
def render_vaccine_dashboard(selected_lgas=None, start_date=None, end_date=None, selected_genders=None, precomputed_data=None, chart_callback=None, zerodose_path=None, visit_path=None):
    
    if precomputed_data:
        figs, summary = precomputed_data
    else:
        figs, summary = get_vaccine_dashboard(selected_lgas, start_date, end_date, selected_genders, zerodose_path, visit_path)
    
    if not figs:
        st.warning("No data found.")
        return

    # --- KPI SCORECARD ---
    st.markdown("### 🏥 KPI Card")
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Facility Visits", f"{summary['fac_visits']:,}", delta_color="normal")
    
    active = summary['zd_burden'] - summary['resolved_cases']
    k2.metric("Active Zero-Dose Burden", f"{active:,}")
    
    dr = summary['dropout_rate']
    k3.metric("Penta1-Penta3 Dropout", f"{dr:.1f}%", help="Target < 10%", delta_color="inverse" if dr > 10 else "normal")
    
    # Priority Count
    # prio_count = len(summary['hotspots'])
    # k4.metric("Priority Settlements", f"{prio_count}", help="Top settlements in filtered view")
    
    st.divider()

    # --- HELPER ---
    def display(fig, title, idx):
        if chart_callback: chart_callback(fig, title, f"vac_v3_{idx}")
        else: st.plotly_chart(fig, use_container_width=True)
    
    fig_dict = {t: f for t, f in figs}

    # --- ROW 1: VACCINE UPTAKE & VOLUME (Compact) ---
    c1, c2 = st.columns(2)
    with c1:
        if "Total Doses Administered" in fig_dict: display(fig_dict["Total Doses Administered"], "Volume: Total Doses Administered", 2)
    with c2:
        if "Schedule Distribution" in fig_dict: display(fig_dict["Schedule Distribution"], "Distribution by Schedule Stage", 3)

    st.markdown("---")

    # --- ROW 2: COVERAGE & TIMELINESS ---
    c3, c4 = st.columns([1, 1.5]) # Uneven columns to give more space to Heatmap or stacked
    with c3:
         if "Coverage by Antigen" in fig_dict: display(fig_dict["Coverage by Antigen"], "Antigen Coverage (%)", 6)
    with c4:
         # Moving Timeliness Heatmap here, slightly wider
         if "Coverage by Age Group" in fig_dict: display(fig_dict["Coverage by Age Group"], "Timeliness: Coverage by Age", 7)

    # --- ROW 3: DROPOUTS & BARRIERS ---
    c5, c6 = st.columns(2)
    with c5:
        if "Series Dropouts" in fig_dict: display(fig_dict["Series Dropouts"], "Dropout Rates by Series", 10)
    with c6:
        if "Zero-Dose Drivers" in fig_dict: display(fig_dict["Zero-Dose Drivers"], "Barriers to Access", 8)

    # --- ROW 4: OPERATIONAL PROGRESS ---
    if "Resolution Progress" in fig_dict:
        display(fig_dict["Resolution Progress"], "Zero-Dose Resolution by LGA", 14)
