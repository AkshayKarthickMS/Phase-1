import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import os
import data_utils

# ---------------------------
# Cached heavy computation
# ---------------------------
@st.cache_resource
def get_dashboard_age(selected_lgas=None, start_date=None, end_date=None, selected_genders=None, facility_path=None, zerodose_path=None):
    """
    Load CSVs, filter data, and generate figures for the Age Analytics Dashboard.
    Returns: figs, zerodose_df, facility_df
    """
    # Hardcoded absolute paths for specific environment
    ABS_FACILITY_PATH = r"C:\Users\AkshayKarthickMS\Desktop\Phase-1-main\data\facility_visits.csv"
    ABS_ZERODOSE_PATH_CSV = r"C:\Users\AkshayKarthickMS\Desktop\Phase-1-main\data\zerodose.csv"
    ABS_ZERODOSE_PATH_XLSX = r"C:\Users\AkshayKarthickMS\Desktop\Phase-1-main\data\zerodose.xlsx"

    # --- 1. ROBUST PATH RESOLUTION ---
    if zerodose_path and os.path.exists(zerodose_path): ZERODOSE_PATH = zerodose_path
    elif os.path.exists(ABS_ZERODOSE_PATH_XLSX): ZERODOSE_PATH = ABS_ZERODOSE_PATH_XLSX
    elif os.path.exists(ABS_ZERODOSE_PATH_CSV): ZERODOSE_PATH = ABS_ZERODOSE_PATH_CSV
    elif os.path.exists("zerodose.xlsx"): ZERODOSE_PATH = "zerodose.xlsx"
    elif os.path.exists("zerodose.csv"): ZERODOSE_PATH = "zerodose.csv"
    else: ZERODOSE_PATH = None

    if facility_path and os.path.exists(facility_path): FACILITY_PATH = facility_path
    elif os.path.exists(ABS_FACILITY_PATH): FACILITY_PATH = ABS_FACILITY_PATH
    elif os.path.exists("facility_visits.csv"): FACILITY_PATH = "facility_visits.csv"
    elif os.path.exists("facility_visit.csv"): FACILITY_PATH = "facility_visit.csv"
    else: FACILITY_PATH = None

    if not ZERODOSE_PATH or not FACILITY_PATH:
        st.error(f"Missing input files. Found: ZD={ZERODOSE_PATH}, Fac={FACILITY_PATH}")
        return [], pd.DataFrame(), pd.DataFrame()

    # --- 2. LOAD DATA ---
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
            fac_df = pd.read_csv(FACILITY_PATH, dtype=str).fillna('')
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return [], pd.DataFrame(), pd.DataFrame()

    # ==========================================
    # 3. PREPROCESS FACILITY DATA
    # ==========================================
    fac_df.columns = fac_df.columns.str.strip().str.lower()
    
    # Standardize LGA
    lga_col_fac = next((c for c in ['lga', 'lga_name'] if c in fac_df.columns), None)
    fac_df['LGA'] = fac_df[lga_col_fac] if lga_col_fac else 'Unknown'

    # Standardize Gender
    fac_df['gender'] = fac_df.get('gender', 'unknown').astype(str).str.lower().str.strip()

    # Filter
    date_col_fac = next((c for c in ['visit_date', 'date'] if c in fac_df.columns), None)
    fac_df = data_utils.filter_data(
        fac_df, selected_lgas, start_date, end_date, date_col_fac, selected_genders
    )

    # Calculate Age (Facility has split columns usually)
    # Using robust fallback logic
    if 'age_years' in fac_df.columns and 'age_months' in fac_df.columns:
        fac_df['age_total_months'] = (
            pd.to_numeric(fac_df['age_years'], errors='coerce').fillna(0) * 12 +
            pd.to_numeric(fac_df['age_months'], errors='coerce').fillna(0) +
            pd.to_numeric(fac_df.get('age_weeks', 0), errors='coerce').fillna(0) / 4.345
        )
    else:
        # Try finding any numeric age col or set to 0
        fac_df['age_total_months'] = 0.0

    # Categorize Age
    fac_df = data_utils.apply_client_age_categories(fac_df, source_type='facility')
    fac_df['age_group'] = fac_df['Client_Age_Group']

    # Extract Vaccines for Timeliness Analysis
    if 'vaccines_administered' in fac_df.columns:
        # Simple extraction of key vaccines to check age appropriateness
        target_vaccines = ['BCG', 'Penta_1', 'Penta_3', 'Measles_1']
        for v in target_vaccines:
            fac_df[v] = fac_df['vaccines_administered'].astype(str).str.contains(v, case=False, na=False)

    # ==========================================
    # 4. PREPROCESS ZERO-DOSE DATA
    # ==========================================
    zd_df.columns = zd_df.columns.str.strip().str.lower()
    
    # Standardize LGA
    lga_col_zd = next((c for c in ['lga', 'lga_name'] if c in zd_df.columns), None)
    zd_df['LGA'] = zd_df[lga_col_zd] if lga_col_zd else 'Unknown'

    # Standardize Gender
    zd_df['Gender'] = zd_df.get('gender', 'unknown').astype(str).str.lower().str.strip()

    # Filter
    date_col_zd = next((c for c in ['enrollment date', 'visit_date'] if c in zd_df.columns), None)
    zd_df = data_utils.filter_data(
        zd_df, selected_lgas, start_date, end_date, date_col_zd, selected_genders,
        include_na_dates=True
    )

    # --- ROBUST AGE CALCULATION (ZERO-DOSE) ---
    # 1. Current Age (From split columns)
    curr_y = next((c for c in zd_df.columns if c in ['current_age_years', 'current_age_year']), None)
    curr_m = next((c for c in zd_df.columns if c in ['current_age_months', 'current_age_month']), None)
    curr_w = next((c for c in zd_df.columns if c in ['current_age_weeks', 'current_age_week']), None)

    if curr_y and curr_m:
        zd_df['current_age_months_calc'] = (
            pd.to_numeric(zd_df[curr_y], errors='coerce').fillna(0) * 12 +
            pd.to_numeric(zd_df[curr_m], errors='coerce').fillna(0) +
            (pd.to_numeric(zd_df[curr_w], errors='coerce').fillna(0) / 4.345 if curr_w else 0)
        )
    else:
        # Fallback to util if columns missing (likely string parsing)
        zd_df['current_age_months_calc'] = zd_df.get('estimated current age', '').apply(data_utils.parse_age_string)

    # 2. Enrollment Age
    enroll_y = next((c for c in zd_df.columns if c in ['age_years', 'age_year']), None)
    enroll_m = next((c for c in zd_df.columns if c in ['age_months', 'age_month']), None)
    enroll_w = next((c for c in zd_df.columns if c in ['age_weeks', 'age_week']), None)

    if enroll_y and enroll_m:
        zd_df['enrollment_age_months_calc'] = (
            pd.to_numeric(zd_df[enroll_y], errors='coerce').fillna(0) * 12 +
            pd.to_numeric(zd_df[enroll_m], errors='coerce').fillna(0) +
            (pd.to_numeric(zd_df[enroll_w], errors='coerce').fillna(0) / 4.345 if enroll_w else 0)
        )
    else:
        zd_df['enrollment_age_months_calc'] = zd_df.get('age at enrollment', '').apply(data_utils.parse_age_string)

    # Apply Categorization using Calculated Current Age
    zd_df['Client_Age_Group'] = zd_df['current_age_months_calc'].apply(data_utils.classify_age_months)
    age_cats = data_utils.AGE_CATEGORIES
    zd_df['Client_Age_Group'] = pd.Categorical(zd_df['Client_Age_Group'], categories=age_cats, ordered=True)

    # Resolution Status
    status_col = next((c for c in ['status', 'resolution status'] if c in zd_df.columns), None)
    if status_col:
        zd_df['is_resolved'] = zd_df[status_col].astype(str).str.lower() == 'resolved'
    else:
        zd_df['is_resolved'] = False

    # ==========================================
    # 5. GENERATE FIGURES
    # ==========================================
    figs = []

    # --- FIG 1: Age Distribution (Population Pyramid style) ---
    # Comparing Facility (Access) vs Zero-Dose (Burden)
    # fig1 = go.Figure()
    # fig1.add_trace(go.Violin(
    #     y=fac_df['age_total_months'], name='Facility Visits',
    #     side='negative', line_color='#636EFA', meanline_visible=True
    # ))
    # fig1.add_trace(go.Violin(
    #     y=zd_df['current_age_months_calc'], name='Zero-Dose List',
    #     side='positive', line_color='#EF553B', meanline_visible=True
    # ))
    # fig1.update_layout(
    #     title="Age Profile Comparison: Access vs Burden",
    #     yaxis_title="Age (Months)",
    #     xaxis_showgrid=False,
    #     violinmode='overlay'
    # )
    # figs.append(("Age Distribution Comparison", fig1))

    # --- FIG 2: Vaccination Timeliness Heatmap (Facility) ---
    # When are vaccines actually given?
    if not fac_df.empty and 'vaccines_administered' in fac_df.columns:
        timeliness_data = []
        for v in ['BCG', 'Penta_1', 'Penta_3', 'Measles_1']:
            if v in fac_df.columns:
                # Get age distribution for children who received this vaccine
                ages = fac_df[fac_df[v] == True]['age_group'].value_counts(normalize=True).mul(100)
                for age_group, pct in ages.items():
                    timeliness_data.append({'Vaccine': v, 'Age Group': age_group, 'Pct': pct})
        
        if timeliness_data:
            df_time = pd.DataFrame(timeliness_data)
            fig2 = px.density_heatmap(
                df_time, x='Age Group', y='Vaccine', z='Pct',
                text_auto='.1f', color_continuous_scale='Mint',
                title="Vaccination Timeliness"
            )
            # Order axes
            fig2.update_xaxes(categoryorder='array', categoryarray=age_cats)
            figs.append(("Vaccination Timeliness", fig2))

    # --- FIG 3: Zero-Dose Stagnation (Enrollment Age vs Current Age) ---
    if not zd_df.empty:
        # Filter out resolved for burden analysis
        active_zd = zd_df[~zd_df['is_resolved']].copy()
        if not active_zd.empty and 'enrollment_age_months_calc' in zd_df.columns:
             # Ensure numeric
             active_zd['e_age'] = pd.to_numeric(active_zd['enrollment_age_months_calc'], errors='coerce')
             active_zd['c_age'] = pd.to_numeric(active_zd['current_age_months_calc'], errors='coerce')
             
             fig3 = px.scatter(
                 active_zd, x='e_age', y='c_age',
                 color='Client_Age_Group', size_max=10, opacity=0.6,
                 title="Zero-Dose Stagnation: Enrollment vs Current Age",
                 labels={'e_age': 'Age at Enrollment (Months)', 'c_age': 'Current Age (Months)'}
             )
             # Add reference line (y=x) - Children on line are new enrollments
             fig3.add_shape(type="line", x0=0, y0=0, x1=60, y1=60, line=dict(color="Gray", dash="dash"))
             figs.append(("Zero-Dose Stagnation", fig3))

    # --- FIG 4: Resolution by Age Group (Outcome) ---
    if not zd_df.empty:
        res_age = zd_df.groupby('Client_Age_Group', observed=False)['is_resolved'].mean().reset_index(name='resolution_rate')
        res_age['resolution_rate'] = res_age['resolution_rate'] * 100
        
        fig4 = px.bar(
            res_age, x='Client_Age_Group', y='resolution_rate',
            title="Resolution Outcome by Age Group",
            labels={'resolution_rate': 'Resolved (%)', 'Client_Age_Group': 'Age Group'},
            color='resolution_rate', color_continuous_scale='Blues'
        )
        fig4.update_layout(yaxis=dict(range=[0, 100]))
        figs.append(("Resolution by Age Group", fig4))

    # --- FIG 5: Missed Vaccines Estimates (Actionable) ---
    def estimate_missed(age_months):
        if age_months < 1.5: return "BCG/OPV0"
        if age_months < 4: return "Penta1/PCV1"
        if age_months < 9: return "Penta3/IPV"
        if age_months < 12: return "Measles1"
        return "Measles2/Catch-up"

    if not zd_df.empty:
        zd_df['est_missed'] = zd_df['current_age_months_calc'].apply(estimate_missed)
        missed_counts = zd_df.groupby(['Client_Age_Group', 'est_missed'], observed=False).size().reset_index(name='count')
        
        fig5 = px.bar(
            missed_counts, x='Client_Age_Group', y='count', color='est_missed',
            title="Estimated Missed Vaccines by Age Burden",
            barmode='stack'
        )
        figs.append(("Estimated Missed Vaccines", fig5))

    # --- NEW COMPREHENSIVE AGE VISUALIZATIONS ---

    # A. Birth Cohort Tracking (Based on Age)
    if not fac_df.empty and 'age_total_months' in fac_df.columns:
        # Approximate birth year relative to current date is complex without fixed reference, 
        # but we can histogram age in months directly to show cohort size
        fig_cohort = px.histogram(
            fac_df, x='age_total_months', nbins=60,
            title="Volume by Age (Month) - Birth Cohort Proxy",
            labels={'age_total_months': 'Age (Months)'},
            color_discrete_sequence=['#ff7f0e']
        )
        figs.append(("Birth Cohort Tracking", fig_cohort))

    # B. Catch-Up Eligibility (Zero Dose)
    if not zd_df.empty:
        def classify_catchup(m):
            if m < 12: return "Routine Schedule"
            if 12 <= m < 24: return "Intensive Catch-up"
            return "Supplementary/Camp"
        
        zd_df['CatchUp_Class'] = zd_df['current_age_months_calc'].apply(classify_catchup)
        catch_counts = zd_df['CatchUp_Class'].value_counts().reset_index()
        catch_counts.columns = ['Class', 'Count']
        
        fig_catch = px.pie(
            catch_counts, values='Count', names='Class',
            title="Catch-Up Strategy Segmentation",
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        figs.append(("Catch-Up Eligibility", fig_catch))

    # # C. Timeliness: Penta 3 Delay Analysis (Facility)
    # if not fac_df.empty and 'Penta_3' in fac_df.columns and 'age_total_months' in fac_df.columns:
    #     p3_df = fac_df[fac_df['Penta_3'].astype(str)=='True'].copy()
    #     if not p3_df.empty:
    #         p3_df['delay_weeks'] = (p3_df['age_total_months'] * 4.345) - 14 # Scheduled at 14 weeks
    #         p3_df = p3_df[p3_df['delay_weeks'] > -10] # Filter extreme errors
    #         p3_df = p3_df[p3_df['delay_weeks'] < 100] # Filter extreme outliers
            
    #         fig_delay = px.box(
    #             p3_df, y='delay_weeks', x='LGA', color='LGA',
    #             title="Penta 3 Timeliness Delay (Weeks > 14w)",
    #             labels={'delay_weeks': 'Delay (Weeks)'}
    #         )
    #         fig_delay.add_hline(y=0, line_dash="dash", line_color="green", annotation_text="On Time")
    #         figs.append(("Penta 3 Timeliness Gap", fig_delay))

    # # D. Age-Gender Pyramid (Zero-Dose)
    # if not zd_df.empty and 'Gender' in zd_df.columns:
    #     pyr_df = zd_df.copy()
    #     pyr_df['age_round'] = pyr_df['current_age_months_calc'].round().astype(int)
    #     pyr_df = pyr_df[pyr_df['age_round'] < 60] # Focus on under 5
        
    #     pyr_agg = pyr_df.groupby(['age_round', 'Gender']).size().reset_index(name='count')
    #     # Make male counts negative for pyramid effect
    #     pyr_agg.loc[pyr_agg['Gender']=='male', 'count'] *= -1
        
    #     fig_pyr = px.bar(
    #         pyr_agg, x='count', y='age_round', color='Gender', orientation='h',
    #         title="Zero-Dose Age-Gender Pyramid",
    #         labels={'age_round': 'Age (Months)', 'count': 'Count'},
    #         color_discrete_map={'male': '#636EFA', 'female': '#EF553B'},
    #         barmode='overlay' # or relative
    #     )
    #     figs.append(("Age-Gender Pyramid", fig_pyr))
        
    # E. Average Age by Settlement (Top 5 Active)
    if not zd_df.empty:
        sett_col = next((c for c in ['settlement', 'settlement_name'] if c in zd_df.columns), None)
        if sett_col:
            active_z = zd_df[~zd_df['is_resolved']]
            top_s = active_z[sett_col].value_counts().nlargest(10).index
            sett_age = active_z[active_z[sett_col].isin(top_s)].groupby(sett_col)['current_age_months_calc'].mean().reset_index()
            sett_age = sett_age.sort_values('current_age_months_calc')
            
            fig_sett_age = px.bar(
                sett_age, x='current_age_months_calc', y=sett_col,
                title="Avg Age of Active Zero-Dose Cases (Top Settlements)",
                labels={'current_age_months_calc': 'Avg Age (Months)'},
                color='current_age_months_calc', color_continuous_scale='Oranges'
            )
            figs.append(("Average Age by Settlement", fig_sett_age))

    return figs, zd_df, fac_df

# ---------------------------
# Render Function (Professional Layout)
# ---------------------------
def render_age_dashboard(selected_lgas=None, start_date=None, end_date=None, selected_genders=None, 
                         facility_path=None, zerodose_path=None, precomputed_data=None, chart_callback=None):
    
    if precomputed_data:
        figs, zd_df, fac_df = precomputed_data
    else:
        figs, zd_df, fac_df = get_dashboard_age(
            selected_lgas, start_date, end_date, selected_genders, facility_path, zerodose_path
        )

    if not figs:
        st.warning("No data found matching filters.")
        return

    # KPIs
    total_zd = len(zd_df)
    newborn_count = len(zd_df[zd_df['Client_Age_Group'] == "Newborn (<6w)"]) if not zd_df.empty else 0
    overaged_count = len(zd_df[zd_df['Client_Age_Group'] == "Overaged ZD"]) if not zd_df.empty else 0
    
    # KPI TOOLTIP UPDATE
    st.markdown("### 👶 Age-Based Vulnerability Indicators")
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Zero-Dose", f"{total_zd:,}")
    # k2.metric("Newborns (<6w)", f"{newborn_count:,}", help="Priority for immediate enrollment. (Verified: 0 records <1.5m in current dataset)")
    k3.metric("Overaged (>2y)", f"{overaged_count:,}", help="Hard-to-reach, requires catch-up campaigns")
    
    st.divider()

    def display(fig, title, idx):
        if chart_callback: chart_callback(fig, title, f"age_prof_{idx}")
        else: st.plotly_chart(fig, use_container_width=True)
            
    fig_dict = {title: fig for title, fig in figs}

    # GRID LAYOUT (Compacted)
    
    # Row 1: Timeliness & Missed Opportunities
    c1, c2 = st.columns(2)
    with c1:
        if "Vaccination Timeliness" in fig_dict: display(fig_dict["Vaccination Timeliness"], "Vaccination Timeliness", 1)
    with c2:
        if "Estimated Missed Vaccines" in fig_dict: display(fig_dict["Estimated Missed Vaccines"], "Estimated Missed Doses", 4)

    st.markdown("---")

    # Row 2: Resolution & Delay
    c3, c4 = st.columns(2)
    with c3:
        if "Resolution by Age Group" in fig_dict: display(fig_dict["Resolution by Age Group"], "Resolution Success", 2)
    # with c4:
    #     # From old Row 4
    #     if "Penta 3 Timeliness Gap" in fig_dict: display(fig_dict["Penta 3 Timeliness Gap"], "Penta 3 Delay Analysis", 7)

    # # Row 3: Demographics & Context
    # st.markdown("##### Demographics & Local Context")
    # c5, c6 = st.columns(2)
    # with c5:
    #      if "Age-Gender Pyramid" in fig_dict: display(fig_dict["Age-Gender Pyramid"], "Zero-Dose Age-Gender Pyramid", 8)
    # with c6:
    #      if "Average Age by Settlement" in fig_dict: display(fig_dict["Average Age by Settlement"], "Chronic vs New Issues (Avg Age)", 9)