import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import os
import data_utils

# ---------------------------
# Vaccine Definitions for Dropout Analysis
# ---------------------------
START_VACCINE = "Penta_1"
END_VACCINE = "Penta_3"

# ---------------------------
# Cached heavy computation
# ---------------------------
@st.cache_resource
def get_gender_dashboard(selected_lgas=None, start_date=None, end_date=None, selected_genders=None, facility_path=None, zerodose_path=None):
    """
    Load CSVs, filter data, and generate figures focusing on Gender Equity.
    """
    # 1. Resolve Paths using data_utils
    FACILITY_PATH = data_utils.find_data_file(facility_path or "facility_visits.csv")
    ZERODOSE_PATH = data_utils.find_data_file(zerodose_path or "zerodose.csv")
    SETTLEMENT_PATH = data_utils.find_data_file("settlement.csv")

    if not ZERODOSE_PATH or not FACILITY_PATH:
         st.error(f"Missing input files. Checked defaults.")
         return [], {}, [], {}

    # 2. Load Data (Robust Load)
    try:
        # Load Zero-Dose
        if ZERODOSE_PATH.endswith('.xlsx'):
            zd_df = pd.read_excel(ZERODOSE_PATH, dtype=str).fillna('')
        else:
            try:
                zd_df = pd.read_csv(ZERODOSE_PATH, dtype=str).fillna('')
            except UnicodeDecodeError:
                zd_df = pd.read_csv(ZERODOSE_PATH, dtype=str, encoding='latin1').fillna('')
        
        # Load Facility
        if FACILITY_PATH.endswith('.xlsx'):
            fac_df = pd.read_excel(FACILITY_PATH, dtype=str).fillna('')
        else:
            try:
                fac_df = pd.read_csv(FACILITY_PATH, dtype=str).fillna('')
            except UnicodeDecodeError:
                fac_df = pd.read_csv(FACILITY_PATH, dtype=str, encoding='latin1').fillna('')

        # Load Settlement Data (New)
        sett_df = pd.DataFrame()
        if SETTLEMENT_PATH:
            if SETTLEMENT_PATH.endswith('.xlsx'):
                sett_df = pd.read_excel(SETTLEMENT_PATH, dtype=str).fillna('')
            else:
                sett_df = pd.read_csv(SETTLEMENT_PATH, dtype=str).fillna('')
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return [], {}, [], {}

    # Capture TOTAL counts before filtering
    total_fac_raw = len(fac_df)
    total_zd_raw = len(zd_df)

    # --- 3. PREPROCESS ZERO-DOSE DATA ---
    zd_df.columns = zd_df.columns.str.strip().str.lower()
    
    lga_col_zd = next((c for c in ['lga', 'lga_name', 'lga_id'] if c in zd_df.columns), None)
    zd_df['LGA'] = zd_df[lga_col_zd] if lga_col_zd else 'Unknown'
    
    zd_df['Gender'] = zd_df.get('gender', 'unknown').astype(str).str.lower().str.strip()

    date_col_zd = next((c for c in ['enrollment date', 'visit_date'] if c in zd_df.columns), None)
    zd_df = data_utils.filter_data(
        zd_df, selected_lgas, start_date, end_date, date_col_zd, selected_genders,
        include_na_dates=True
    )

    # --- 4. PREPROCESS FACILITY DATA ---
    fac_df.columns = fac_df.columns.str.strip().str.lower()
    
    lga_col_fac = next((c for c in ['lga', 'lga_name'] if c in fac_df.columns), None)
    fac_df['LGA'] = fac_df[lga_col_fac] if lga_col_fac else 'Unknown'
    
    fac_df['Gender'] = fac_df.get('gender', 'unknown').astype(str).str.lower().str.strip()

    date_col_fac = next((c for c in ['visit_date', 'date'] if c in fac_df.columns), None)
    fac_df = data_utils.filter_data(
        fac_df, selected_lgas, start_date, end_date, date_col_fac, selected_genders
    )

    # --- 5. PREPROCESS SETTLEMENT DATA ---
    if not sett_df.empty:
        sett_df.columns = sett_df.columns.str.strip().str.lower()
        
        lga_col_s = next((c for c in ['lga', 'lga_name'] if c in sett_df.columns), None)
        sett_df['LGA'] = sett_df[lga_col_s] if lga_col_s else 'Unknown'
        
        sett_df['Gender'] = sett_df.get('gender', 'unknown').astype(str).str.lower().str.strip()
        
        date_col_s = next((c for c in ['enrollment date', 'visit_date'] if c in sett_df.columns), None)
        sett_df = data_utils.filter_data(sett_df, selected_lgas, start_date, end_date, date_col_s, selected_genders)
        
        # Age processing for Settlement Data
        # Robust check for split numeric columns
        s_curr_y = next((c for c in sett_df.columns if c in ['current_age_years', 'current_age_year']), None)
        s_curr_m = next((c for c in sett_df.columns if c in ['current_age_months', 'current_age_month']), None)
        
        if s_curr_y and s_curr_m:
             # If columns exist, calculate directly
             sett_df['age_months_s'] = (
                pd.to_numeric(sett_df[s_curr_y], errors='coerce').fillna(0) * 12 +
                pd.to_numeric(sett_df[s_curr_m], errors='coerce').fillna(0)
             )
        else:
             # Fallback to string parsing if numeric splits not found
             age_col_s = next((c for c in ['estimated current age', 'current age', 'age'] if c in sett_df.columns), None)
             if age_col_s:
                sett_df['age_months_s'] = sett_df[age_col_s].apply(data_utils.parse_age_string)
             else:
                sett_df['age_months_s'] = 0.0
        
        sett_df['Age Group'] = sett_df['age_months_s'].apply(data_utils.classify_age_months)
        age_cats = data_utils.AGE_CATEGORIES
        sett_df['Age Group'] = pd.Categorical(sett_df['Age Group'], categories=age_cats, ordered=True)

        sett_name_col = next((c for c in ['settlement', 'settlement_name'] if c in sett_df.columns), None)
        sett_df['Settlement'] = sett_df[sett_name_col].str.title() if sett_name_col else "Unknown"

    # --- 6. METRICS CALCULATIONS ---
    # Facility
    fac_gender_stats = fac_df.groupby('Gender').agg(
        total_visits=('Gender', 'count')
    ).reset_index()
    
    # Zero-Dose
    if not zd_df.empty:
        # Robust Age String Construction
        if 'estimated current age' not in zd_df.columns:
            if 'current_age_years' in zd_df.columns:
                 years_part = zd_df['current_age_years'].astype(str).replace('nan', '0').replace('', '0')
                 
                 if 'current_age_month' in zd_df.columns:
                     months_part = zd_df['current_age_month'].astype(str).replace('nan', '0').replace('', '0')
                 else:
                     months_part = "0"
                     
                 zd_df['Estimated Current Age'] = years_part + " years " + months_part + " months"
        
        zd_df = data_utils.apply_client_age_categories(zd_df, source_type='zerodose')
    
    status_col = next((c for c in ['status', 'resolution status'] if c in zd_df.columns), None)
    if not zd_df.empty and status_col:
        zd_df['is_resolved'] = zd_df[status_col].astype(str).str.lower() == 'resolved'
    else:
        zd_df['is_resolved'] = False

    # Extract Distance if available
    dist_col = next((c for c in ['distance to hf', 'distance to'] if c in zd_df.columns), None)
    if not zd_df.empty and dist_col:
        zd_df['dist_val'] = zd_df[dist_col].astype(str).str.extract(r'(\d+\.?\d*)').astype(float).fillna(0)
    else:
        zd_df['dist_val'] = 0.0

    # Detect Vaccines in Facility Data
    VACCINE_LIST = [
        "BCG", "OPV_0", "HepB_0", "Penta_1", "PCV_1", "OPV_1", "Rota_1",
        "Penta_2", "PCV_2", "OPV_2", "Rota_2",
        "Penta_3", "PCV_3", "OPV_3", "Rota_3", "IPV_1", "IPV_2",
        "Measles_1", "YellowFever", "Meningitis", "Measles_2"
    ]
    
    found_vaccines = []
    if not fac_df.empty:
        found_vaccines = [v for v in VACCINE_LIST if v in fac_df.columns]
        # Fallback if explicit columns missing but 'vaccines_administered' exists? 
        # (Assuming robust columns for now based on other dashboards)

    # ==========================================
    # GENERATE FIGURES
    # ==========================================
    px.defaults.template = "plotly_white"
    figs = []
    
    # 1. Routine Access
    if not fac_gender_stats.empty:
        fig1 = px.pie(
            fac_gender_stats, values='total_visits', names='Gender', 
            title="Routine Immunization Access (Facility Visits)", hole=0.5,
            color='Gender', color_discrete_map={'male': '#636EFA', 'female': '#EF553B', 'unknown': 'gray'}
        )
        fig1.update_traces(textposition='inside', textinfo='percent+label')
        figs.append(("Routine Access by Gender", fig1))

    # REMOVED VIZ 2 (Dropout Rate)

    # 3. Resolution Success Rate
    if not zd_df.empty:
        res_stats = zd_df.groupby('Gender')['is_resolved'].agg(['count', 'sum']).reset_index()
        res_stats.columns = ['Gender', 'Total', 'Resolved']
        res_stats['Rate'] = (res_stats['Resolved'] / res_stats['Total']) * 100
        
        fig3 = px.bar(
            res_stats, x='Gender', y='Rate', color='Gender',
            title="Outreach Success: Resolution Rate",
            text=res_stats['Rate'].apply(lambda x: f"{x:.1f}%"),
            color_discrete_map={'male': '#636EFA', 'female': '#EF553B'}
        )
        fig3.update_layout(yaxis=dict(range=[0, 100]))
        figs.append(("Resolution Success Rate", fig3))

    # 4. Active Burden
    active_cases = zd_df[~zd_df['is_resolved']]
    if not active_cases.empty:
        active_counts = active_cases['Gender'].value_counts().reset_index()
        active_counts.columns = ['Gender', 'Count']
        fig4 = px.pie(
            active_counts, values='Count', names='Gender', 
            title="Current Active Zero-Dose Burden", hole=0.5,
            color='Gender', color_discrete_map={'male': '#636EFA', 'female': '#EF553B'}
        )
        fig4.update_traces(textposition='inside', textinfo='value+percent')
        figs.append(("Current Active Burden", fig4))

    # REMOVED VIZ 5 (Barriers by Gender)

    # 6. Age Profile (REMOVED - Redundant with Age Dashboard)
    # if not zd_df.empty and 'Client_Age_Group' in zd_df.columns:
    #     age_gender = zd_df.groupby(['Client_Age_Group', 'Gender'], observed=False).size().reset_index(name='count')
    #     fig6 = px.bar(
    #         age_gender, x='Client_Age_Group', y='count', color='Gender', 
    #         barmode='stack',
    #         title="Age Profile of Zero-Dose Children",
    #         labels={'Client_Age_Group': 'Age Category'},
    #         color_discrete_map={'male': '#636EFA', 'female': '#EF553B'}
    #     )
    #     figs.append(("Age Profile by Gender", fig6))

    # --- NEW VISUALS FROM SETTLEMENT DATA ---
    if not sett_df.empty:
        # 7. Reason for ZD by Age (Heatmap)
        reason_col_s = next((c for c in ['reason for zd', 'reasons', 'reason'] if c in sett_df.columns), None)
        if reason_col_s:
            # Show ALL reasons as requested (no nlargest filter)
            sett_r_df = sett_df.copy()
            reason_matrix_s = pd.crosstab(sett_r_df[reason_col_s], sett_r_df['Age Group'])
            
            fig7 = px.imshow(
                reason_matrix_s, text_auto=True, aspect="auto",
                title="Reasons for Zero-Dose by Age Group",
                color_continuous_scale='Reds'
            )
            figs.append(("Reasons by Age Group", fig7))
        
        # 8. Zero-Dose by Settlement & Age (Stacked)
        # Show ALL settlements as requested
        all_sett_s = sett_df['Settlement'].unique()
        sett_age_counts_s = sett_df.groupby(['Settlement', 'Age Group'], observed=False).size().reset_index(name='Count')
        
        # Dynamic height calculation
        n_settlements = len(all_sett_s)
        calc_height = max(600, n_settlements * 25)

        fig8 = px.bar(
            sett_age_counts_s, y='Settlement', x='Count', color='Age Group',
            title=f"Zero-Dose Burden by Settlement & Age",
            orientation='h',
            # category_orders={'Settlement': list(top_sett_s)}, # Let plotly sort or sort by total
            barmode='stack',
            height=calc_height
        )
        fig8.update_layout(yaxis=dict(autorange="reversed", tickmode='linear'))
        figs.append(("Zero-Dose by Settlement & Age", fig8))

    # --- NEW COMPREHENSIVE VISUALIZATIONS ---

    # A. Antigen Uptake by Gender (Volume)
    if not fac_df.empty and found_vaccines:
        # Sum doses for each gender
        vol_data = fac_df.groupby('Gender')[found_vaccines].sum().T.reset_index()
        vol_data = vol_data.melt(id_vars='index', var_name='Gender', value_name='Doses')
        
        fig_vol = px.bar(
            vol_data, x='index', y='Doses', color='Gender', barmode='group',
            title="Antigen Uptake Volume by Gender",
            labels={'index': 'Antigen'},
            color_discrete_map={'male': '#636EFA', 'female': '#EF553B'}
        )
        figs.append(("Antigen Uptake Volume", fig_vol))

    # B. Dropout Rates by Gender (Penta1 -> Penta3, Measles)
    if not fac_df.empty and 'Penta_1' in found_vaccines and 'Penta_3' in found_vaccines:
        drop_data = []
        for g in ['male', 'female']:
            g_df = fac_df[fac_df['Gender'] == g]
            p1 = g_df['Penta_1'].astype(float).sum()
            p3 = g_df['Penta_3'].astype(float).sum()
            rate = ((p1 - p3) / p1 * 100) if p1 > 0 else 0
            drop_data.append({'Gender': g, 'Series': 'Penta1-3', 'Rate': rate})
            
            # Measles check (Penta1 -> Measles1) - proxy for full infant
            if 'Measles_1' in found_vaccines:
                 m1 = g_df['Measles_1'].astype(float).sum()
                 m_rate = ((p1 - m1) / p1 * 100) if p1 > 0 else 0
                 drop_data.append({'Gender': g, 'Series': 'Penta1-Measles1', 'Rate': m_rate})

        if drop_data:
            df_drop = pd.DataFrame(drop_data)
            fig_drop = px.bar(
                df_drop, x='Series', y='Rate', color='Gender', barmode='group',
                title="Dropout Rates by Gender",
                text=df_drop['Rate'].map(lambda x: f"{x:.1f}%"),
                color_discrete_map={'male': '#636EFA', 'female': '#EF553B'}
            )
            figs.append(("Dropout Rates by Gender", fig_drop))

    # C. Timeliness: Measles Age Distribution (Violin)
    if not fac_df.empty and 'Measles_1' in found_vaccines and 'age_months' in fac_df.columns:
        m_df = fac_df[fac_df['Measles_1'].astype(str)=='1'].copy()
        if not m_df.empty:
            m_df['age_months'] = pd.to_numeric(m_df['age_months'], errors='coerce')
            m_df = m_df.dropna(subset=['age_months'])
            m_df = m_df[m_df['age_months'] < 36] # Filter outliers
            
            fig_viol = px.violin(
                m_df, y="age_months", x="Gender", color="Gender", box=True, points="all",
                title="Age at Measles Vaccination by Gender",
                color_discrete_map={'male': '#636EFA', 'female': '#EF553B'}
            )
            fig_viol.add_hline(y=9, line_dash="dash", line_color="green", annotation_text="Target 9m")
            figs.append(("Measles Age Distribution", fig_viol))

    # D. Barriers by Gender (Heatmap)
    reason_col = next((c for c in ['reason for zd', 'reasons', 'reason'] if c in zd_df.columns), None)
    if reason_col:
        # Top 10 reasons
        top_reasons = zd_df[reason_col].value_counts().nlargest(10).index
        zd_r = zd_df[zd_df[reason_col].isin(top_reasons)]
        
        ct = pd.crosstab(zd_r[reason_col], zd_r['Gender'])
        fig_barr = px.imshow(
            ct, text_auto=True, aspect="auto",
            title="Zero-Dose Barriers by Gender",
            color_continuous_scale='Purples'
        )
        figs.append(("Barriers by Gender", fig_barr))

    # E. LGA Gender Parity (Map/Bar)
    if not fac_df.empty:
        lga_gen = fac_df.groupby(['LGA', 'Gender']).size().unstack(fill_value=0)
        # Normalize by row sum to get mix
        # Or simplistic Ratio F/M
        if 'female' in lga_gen.columns and 'male' in lga_gen.columns:
            lga_gen['Ratio (F/M)'] = lga_gen['female'] / lga_gen['male'].replace(0, 1)
            lga_gen = lga_gen.sort_values('Ratio (F/M)')
            
            fig_par = px.bar(
                lga_gen.reset_index(), x='LGA', y='Ratio (F/M)',
                title="Gender Parity Ratio by LGA (Target = 1.0)",
                color='Ratio (F/M)', color_continuous_scale='RdBu'
            )
            fig_par.add_hline(y=1.0, line_dash="solid", line_color="black")
            figs.append(("LGA Gender Parity", fig_par))
            
    # F. Distance Analysis by Gender (Box Plot)
    if not zd_df.empty and 'dist_val' in zd_df.columns and zd_df['dist_val'].max() > 0:
        fig_dist = px.box(
            zd_df, x='Gender', y='dist_val', color='Gender',
            title="Distance to Health Facility by Gender (Active ZD)",
            labels={'dist_val': 'Distance (km)'},
            color_discrete_map={'male': '#636EFA', 'female': '#EF553B'}
        )
        figs.append(("Distance Distribution", fig_dist))
        
    # G. Missed Opportunities (Single Antigen Visits)
    if not fac_df.empty and found_vaccines:
         fac_df['vax_count'] = fac_df[found_vaccines].apply(pd.to_numeric, errors='coerce').sum(axis=1)
         avg_vax = fac_df.groupby('Gender')['vax_count'].mean().reset_index()
         
         fig_opp = px.bar(
             avg_vax, x='Gender', y='vax_count', color='Gender',
             title="Avg. Vaccines Received Per Visit",
             text=avg_vax['vax_count'].map(lambda x: f"{x:.2f}"),
             color_discrete_map={'male': '#636EFA', 'female': '#EF553B'}
         )
         figs.append(("Vaccines Per Visit", fig_opp))

    summary = {
        "fac_visits": len(fac_df),
        "fac_total_raw": total_fac_raw,
        "zd_count": len(zd_df),
        "zd_total_raw": total_zd_raw,
        "male_count": len(fac_df[fac_df['Gender'] == 'male']),
        "female_count": len(fac_df[fac_df['Gender'] == 'female'])
    }
    
    data_out = {"zerodose_df": zd_df, "facility_df": fac_df}
    insights = [] 

    return figs, summary, insights, data_out

# ---------------------------
# Render Function
# ---------------------------
def render_gender_dashboard(selected_lgas=None, start_date=None, end_date=None, selected_genders=None, 
                            facility_path=None, zerodose_path=None, precomputed_data=None, chart_callback=None):
    
    if precomputed_data:
        figs, summary, insights, data_out = precomputed_data
    else:
        figs, summary, insights, data_out = get_gender_dashboard(
            selected_lgas, start_date, end_date, selected_genders, facility_path, zerodose_path
        )

    if not figs:
        st.warning("No data found matching filters.")
        return

    st.markdown("### 🚻 Gender Equity: Access & Outcomes")
    k1, k2, k3, k4 = st.columns(4)
    # Display Filtered vs Total counts
    k1.metric("Facility Visits", f"{summary.get('fac_visits',0):,}", help=f"Filtered from {summary.get('fac_total_raw', 'N/A')} total records")
    k2.metric("Zero-Dose Cases", f"{summary.get('zd_count',0):,}", help=f"Filtered from {summary.get('zd_total_raw', 'N/A')} total records")
    k3.metric("Male Visits", f"{summary.get('male_count',0):,}")
    k4.metric("Female Visits", f"{summary.get('female_count',0):,}")
    st.divider()

    def display(fig, title, idx):
        if chart_callback: chart_callback(fig, title, f"gender_{idx}")
        else: st.plotly_chart(fig, use_container_width=True)
            
    fig_dict = {title: fig for title, fig in figs}

    c1, c2 = st.columns(2)
    with c1:
        if "Routine Access by Gender" in fig_dict: display(fig_dict["Routine Access by Gender"], "Routine Access", 0)
    with c2:
        if "Current Active Burden" in fig_dict: display(fig_dict["Current Active Burden"], "Active Burden", 1)

    st.markdown("---")
    
    # Settlement Insights (Expanded)
    st.markdown("### 🏘️ Settlement & Barriers Context")
    if "Reasons by Age Group" in fig_dict: 
        display(fig_dict["Reasons by Age Group"], "Barriers by Age Group", 4)
    
    if "Zero-Dose by Settlement & Age" in fig_dict: 
        # Full width for large settlement chart
        display(fig_dict["Zero-Dose by Settlement & Age"], "Age Profile by Settlement (All Settlements)", 5)

    # REMOVED SECTIONS (Kept hidden/omitted as requested previously)