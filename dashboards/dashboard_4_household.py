import os
import re
from datetime import datetime

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import data_utils

# === MAIN DASHBOARD LOGIC ===
@st.cache_resource
def get_household_dashboard(
    selected_lgas=None, 
    start_date=None, 
    end_date=None, 
    selected_genders=None,
    zerodose_path=None,
    visit_path=None,
    top_n=None # Changed default to None to show ALL settlements
):
    """
    Load data, filter, compute settlement-level summaries and priority scores, build figures.
    """
    # 1. Resolve Paths using data_utils
    FACILITY_PATH = data_utils.find_data_file(visit_path or "facility_visit.csv")
    
    # Prioritize settlement.csv ONLY if specifically requested or if it's a known better source (which it isn't here)
    # Reverting prioritization of settlement.csv as it lacks Resolution Status and has fewer rows.
    
    # Fallback to the provided zerodose path or default
    ZERODOSE_PATH = data_utils.find_data_file(zerodose_path or "zerodose.csv")

    # Verify existence
    missing = []
    if not ZERODOSE_PATH: missing.append("settlement.csv or zerodose.csv/.xlsx")
    if not FACILITY_PATH: missing.append("facility_visit.csv/.xlsx")
    
    if missing:
         st.error(f"Missing input files: {', '.join(missing)}")
         return [], pd.DataFrame(), {}

    # 2. Load Data (Dual Load Strategy)
    try:
        # Load FULL Zero-Dose List (for KPIs)
        zd_path_full = data_utils.find_data_file("zerodose.xlsx")
        if not zd_path_full: zd_path_full = data_utils.find_data_file("zerodose.csv")
        
        if zd_path_full and zd_path_full.endswith('.xlsx'):
            zd_full = pd.read_excel(zd_path_full, dtype=str).fillna('')
        elif zd_path_full:
             try: zd_full = pd.read_csv(zd_path_full, dtype=str).fillna('')
             except: zd_full = pd.read_csv(zd_path_full, dtype=str, encoding='latin1').fillna('')
        else:
            zd_full = pd.DataFrame()

        # Load Mapped Settlement Data (for Charts)
        sett_path = data_utils.find_data_file("settlement.csv")
        if sett_path:
             zd_mapped = pd.read_csv(sett_path, dtype=str).fillna('')
        else:
             zd_mapped = pd.DataFrame()

        # Load Facility Data
        if FACILITY_PATH.endswith('.xlsx'):
            vis = pd.read_excel(FACILITY_PATH, dtype=str).fillna('')
        else:
            vis = pd.read_csv(FACILITY_PATH, dtype=str).fillna('')
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return [], pd.DataFrame(), {}
    
    # Use zd_mapped for the main analysis logic, but keep zd_full for KPIs
    if zd_mapped.empty:
        st.warning("Settlement mapping data missing. Using full list without coordinates.")
        zd = zd_full
    else:
        zd = zd_mapped # Primary for charts


    # --- PREPROCESSING & FILTERING ---

    # 1. Facility Data Prep
    vis.columns = vis.columns.str.strip().str.lower()
    lga_col_vis = next((c for c in ['lga_name', 'lga'] if c in vis.columns), None)
    if lga_col_vis: vis['LGA'] = vis[lga_col_vis]
    else: vis['LGA'] = 'Unknown'
    
    vis['gender'] = vis.get('gender', 'unknown').astype(str).str.lower().str.strip()

    date_col_vis = 'visit_date' if 'visit_date' in vis.columns else None
    vis = data_utils.filter_data(
        vis, selected_lgas, start_date, end_date, date_col_vis, selected_genders
    )

    # 2. Zero-Dose Data Prep
    zd.columns = zd.columns.str.strip().str.lower()
    
    col_map = {
        'lga_name': 'LGA',
        'resolution status': 'Status',
        'reasons_for_zero_dose': 'Reason for ZD',
        'distance to': 'Distance'
    }
    zd = zd.rename(columns=col_map)
    
    if 'LGA' not in zd.columns:
        lga_col_zd = next((c for c in ['lga', 'lga_name'] if c in zd.columns), None)
        if lga_col_zd: zd['LGA'] = zd[lga_col_zd]
        else: zd['LGA'] = 'Unknown'

    g_col = 'Gender' if 'Gender' in zd.columns else 'gender'
    zd['Gender'] = zd.get(g_col, 'unknown').astype(str).str.lower().str.strip()

    # Filter Zero-Dose
    if 'enrollment date' in zd.columns: zd_date_col = 'enrollment date'
    elif 'visit_date' in zd.columns: zd_date_col = 'visit_date'
    else: zd_date_col = None

    zd = data_utils.filter_data(
        zd, selected_lgas, start_date, end_date, zd_date_col, selected_genders,
        include_na_dates=True
    )

    if zd.empty and vis.empty:
        return [], pd.DataFrame(), {"total_settlements": 0}

    # --- AGGREGATION LOGIC ---
    
    # Normalize Settlement
    settlement_candidates = ['settlement', 'settlement_name', 'village', 'community', 'ward', 'settlementid']
    settlement_col = next((c for c in settlement_candidates if c in zd.columns), None)
    
    if not settlement_col:
        settlement_col = next((c for c in zd.columns if 'settlement' in c and 'id' not in c), None)

    if settlement_col:
        zd['Settlement_std'] = zd[settlement_col].astype(str).str.strip().str.title()
    else:
        zd['Settlement_std'] = 'Unknown'

    # Normalize Status
    status_col = next((c for c in ['status', 'resolution status'] if c in zd.columns), None)
    zd['Status_std'] = zd[status_col].astype(str).str.strip().str.title() if status_col else ''
    
    zd['is_resolved'] = zd['Status_std'].str.lower() == 'resolved'

    woc_col = next((c for c in ['woman or child', 'category'] if c in zd.columns), None)
    if woc_col:
        zd['Woman_or_child_std'] = zd[woc_col].astype(str).str.strip().str.lower()
    else:
        zd['Woman_or_child_std'] = 'child'
    
    zd['LGA_std'] = zd['LGA'].astype(str).str.strip().str.title()
    vis['LGA_std'] = vis['LGA'].astype(str).str.strip().str.title()

    # Keep only child records
    zd_children = zd[zd['Woman_or_child_std'] == 'child'].copy()

    # --- KEY METRICS PER SETTLEMENT ---
    
    settlement_stats = zd_children.groupby(['Settlement_std', 'LGA_std']).agg(
        zero_dose_count=('LGA', 'count'),
        resolved_count=('is_resolved', 'sum'),
    ).reset_index()
    
    settlement_stats['active_burden'] = settlement_stats['zero_dose_count'] - settlement_stats['resolved_count']
    settlement_stats['dropout_count'] = settlement_stats['active_burden'] # Proxy

    # Distance
    dist_col = next((c for c in zd.columns if 'distance' in c), None)
    if dist_col:
        zd_children['dist_val'] = zd_children[dist_col].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
        dist_agg = zd_children.groupby('Settlement_std')['dist_val'].mean().reset_index(name='avg_distance_km')
        settlement_stats = pd.merge(settlement_stats, dist_agg, on='Settlement_std', how='left')
    else:
        settlement_stats['avg_distance_km'] = 0

    # Top Barrier
    reason_col = next((c for c in ['reason for zd', 'reasons_for_zero_dose', 'reasons'] if c in zd.columns), None)
    if reason_col:
        reason_mode = zd_children.groupby('Settlement_std')[reason_col].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"
        ).reset_index(name='primary_barrier')
        settlement_stats = pd.merge(settlement_stats, reason_mode, on='Settlement_std', how='left')
    else:
        settlement_stats['primary_barrier'] = "Unknown"

    # Priority Scoring
    zmax = settlement_stats['zero_dose_count'].max() or 1
    dmax = settlement_stats['dropout_count'].max() or 1
    
    settlement_stats['priority_score'] = (
        0.7 * (settlement_stats['zero_dose_count'] / zmax) + 
        0.3 * (settlement_stats['dropout_count'] / dmax)
    )
    
    # Sort
    settlement_stats = settlement_stats.sort_values('priority_score', ascending=False)

    # ==========================================
    # GENERATE FIGURES
    # ==========================================
    figs = []
    
    # Handle Displaying ALL settlements
    # If top_n is provided (e.g. 30), slice. If None, show all.
    if top_n:
        display_stats = settlement_stats.head(top_n)
        chart_title_suffix = f"Top"
    else:
        display_stats = settlement_stats
        chart_title_suffix = f"All"
        
    # Calculate dynamic height based on number of bars (25px per bar, min 600px)
    dynamic_height = max(600, len(display_stats) * 25)

    # 1. Settlements by Total Zero-Dose
    if not display_stats.empty:
        # Sort for chart display (highest on top)
        display_total = display_stats.sort_values('zero_dose_count', ascending=True)
        
        fig_zero = px.bar(
            display_total,
            x='zero_dose_count',
            y='Settlement_std',
            orientation='h',
            color='LGA_std',
            hover_name='Settlement_std', # Explicit hover name
            hover_data=['active_burden', 'priority_score', 'avg_distance_km'],
            title=f"{chart_title_suffix} Settlements — Total Zero-Dose Children",
            labels={'zero_dose_count': 'Total Identified Cases', 'Settlement_std': 'Settlement'},
            height=dynamic_height
        )
        # fig_zero.update_layout(yaxis={'categoryorder': 'total ascending'}) # sorting handled by dataframe sort
        figs.append(("Settlements by Total Zero-dose Children", fig_zero))

    # 2. Settlements by Active Burden
    if not display_stats.empty:
        display_active = display_stats.sort_values('active_burden', ascending=True)
        
        fig_active = px.bar(
            display_active,
            x='active_burden',
            y='Settlement_std',
            orientation='h',
            color='LGA_std',
            text='active_burden',
            hover_name='Settlement_std', # Explicit hover name
            hover_data=['zero_dose_count', 'priority_score'],
            title=f"{chart_title_suffix} Settlements — Active Unresolved Burden",
            labels={'active_burden': 'Active Unresolved Cases', 'Settlement_std': 'Settlement'},
            height=dynamic_height
        )
        figs.append(("Settlements by Unresolved Cases", fig_active))

    # 3. Settlement Prioritization (Scatter) - Always shows all
    if not settlement_stats.empty:
        fig_priority = px.scatter(
            settlement_stats,
            x='avg_distance_km',
            y='active_burden',
            size='priority_score',
            color='LGA_std',
            hover_name='Settlement_std', # Explicit hover name
            hover_data=['primary_barrier', 'zero_dose_count'],
            title="Settlement Prioritization: Distance vs Active Burden",
            labels={'avg_distance_km': 'Avg Distance to HF (km)', 'active_burden': 'Active Cases'},
            height=600
        )
        figs.append(("Settlement Prioritization (Scatter)", fig_priority))

    # --- NEW COMPREHENSIVE SETTLEMENT/HOUSEHOLD VISUALIZATIONS ---

    # A. Ward-Level Burden Heatmap (Treemap)
    # Check for Ward column
    ward_col = next((c for c in zd_children.columns if c in ['ward', 'ward_name']), None)
    if not ward_col:
        # Infer Ward from Settlement if possible or just use LGA
        zd_children['Ward_std'] = zd_children['LGA_std'] # Fallback
    else:
        zd_children['Ward_std'] = zd_children[ward_col].astype(str).str.title()

    ward_stats = zd_children.groupby(['LGA_std', 'Ward_std']).size().reset_index(name='count')
    fig_ward = px.treemap(
        ward_stats, path=['LGA_std', 'Ward_std'], values='count',
        title="Burden Distribution by LGA & Ward",
        color='count', color_continuous_scale='Reds'
    )
    figs.append(("Ward Burden Heatmap", fig_ward))

    # B. Distance Distribution Histogram
    if 'dist_val' in zd_children.columns:
        fig_dist_hist = px.histogram(
            zd_children, x='dist_val', nbins=50,
            title="Distance to Health Facility Distribution",
            labels={'dist_val': 'Distance (km)'},
            color_discrete_sequence=['#2CA02C']
        )
        figs.append(("Distance Distribution", fig_dist_hist))

    # C. Operational Efficiency (Settlement Case Finding)
    # Compare Settlment size (proxy by # cases) vs Distance
    # Actually, let's look at Resolution Rate by Distance Bin
    if 'dist_val' in zd_children.columns and 'is_resolved' in zd_children.columns:
        zd_children['dist_bin'] = pd.cut(zd_children['dist_val'], bins=[0, 2, 5, 10, 20, 100], labels=['<2km', '2-5km', '5-10km', '10-20km', '>20km'])
        dist_res = zd_children.groupby('dist_bin', observed=False)['is_resolved'].mean().reset_index(name='res_rate')
        dist_res['res_rate'] *= 100
        
        fig_dist_res = px.line(
            dist_res, x='dist_bin', y='res_rate', markers=True,
            title="Resolution Success vs Distance",
            labels={'dist_bin': 'Distance Bin', 'res_rate': 'Resolution Rate (%)'},
            range_y=[0, 100]
        )
        figs.append(("Resolution vs Distance", fig_dist_res))

    # D. Zero-Dose Clustering (Box Plot of Burden per Settlement by LGA)
    # Are cases spread out or clustered in super-spreader settlements?
    fig_cluster = px.box(
        settlement_stats, x='LGA_std', y='zero_dose_count',
        title="Settlement Burden Distribution by LGA (Clustering Check)",
        points='all',
        hover_data=['Settlement_std']
    )
    figs.append(("Zero-Dose Clustering", fig_cluster))

    # E. Dominant Barriers by Ward/LGA
    if reason_col:
        barrier_agg = zd_children.groupby(['LGA_std', reason_col]).size().reset_index(name='count')
        # Filter top barriers
        top_b = barrier_agg.groupby(reason_col)['count'].sum().nlargest(8).index
        barrier_agg = barrier_agg[barrier_agg[reason_col].isin(top_b)]
        
        fig_bar_ward = px.bar(
            barrier_agg, x='LGA_std', y='count', color=reason_col,
            title="Dominant Barriers by LGA",
            barmode='stack'
        )
        figs.append(("Barriers by LGA", fig_bar_ward))

    # F. Cumulative Settlement Burden (Pareto)
    # Sort settlements by burden
    pareto_data = settlement_stats.sort_values('zero_dose_count', ascending=False).reset_index(drop=True)
    pareto_data['cumulative_pct'] = pareto_data['zero_dose_count'].cumsum() / pareto_data['zero_dose_count'].sum() * 100
    pareto_data['settlement_rank'] = pareto_data.index + 1
    
    fig_pareto = px.line(
        pareto_data, x='settlement_rank', y='cumulative_pct',
        title="Settlement Burden Concentration (Pareto)",
        labels={'settlement_rank': 'Settlements (Ranked)', 'cumulative_pct': 'Cumulative Burden (%)'}
    )
    # Add reference lines like "Top 20% of settlements hold X% of burden"
    total_s = len(pareto_data)
    top_20_idx = int(total_s * 0.2)
    if top_20_idx < len(pareto_data):
        val_20 = pareto_data.iloc[top_20_idx]['cumulative_pct']
        fig_pareto.add_vline(x=top_20_idx, line_dash="dash", annotation_text=f"Top 20% = {val_20:.0f}% Burden")
        
    figs.append(("Burden Pareto Chart", fig_pareto))

    # Calculate Total Active Burden from FULL dataset (zd_full) if available
    # Must apply same filters to zd_full as we did to zd
    
    total_burden_val = 0
    if not zd_full.empty:
        # Standardize Columns for Filtering
        zd_full.columns = zd_full.columns.str.strip().str.lower()
        col_map_full = {'lga_name': 'LGA', 'resolution status': 'Status', 'reasons_for_zero_dose': 'Reason for ZD'}
        zd_full = zd_full.rename(columns=col_map_full)
        if 'LGA' not in zd_full.columns:
             lga_c = next((c for c in ['lga', 'lga_name'] if c in zd_full.columns), None)
             if lga_c: zd_full['LGA'] = zd_full[lga_c]
             
        # Filter zd_full
        date_c_full = 'enrollment date' if 'enrollment date' in zd_full.columns else ('visit_date' if 'visit_date' in zd_full.columns else None)
        zd_full_filt = data_utils.filter_data(zd_full, selected_lgas, start_date, end_date, date_c_full, selected_genders, include_na_dates=True)
        
        # Calculate Unresolved
        stat_c = next((c for c in ['status', 'resolution status'] if c in zd_full_filt.columns), None)
        if stat_c:
             # Count Unresolved (Active)
             # Resolution Status usually 'Resolved' or NaN/Active
             is_res = zd_full_filt[stat_c].astype(str).str.strip().str.lower() == 'resolved'
             total_burden_val = (~is_res).sum()
        else:
             total_burden_val = len(zd_full_filt)
    else:
        # Fallback to mapped data sum
        total_burden_val = int(settlement_stats['active_burden'].sum())

    summary = {
        "total_settlements": len(settlement_stats),
        "total_active_burden": total_burden_val,
        "avg_dist": settlement_stats['avg_distance_km'].mean()
    }

    # Format table for display
    table_df = settlement_stats[[
        'Settlement_std', 'LGA_std', 'active_burden', 'zero_dose_count', 
        'primary_barrier', 'avg_distance_km', 'priority_score'
    ]].copy()
    
    table_df.columns = ['Settlement', 'LGA', 'Active Burden', 'Total Cases', 'Top Barrier', 'Dist(km)', 'Priority Score']
    # Show more rows in table since user wants to see all
    table_df = table_df.head(500) 

    return figs, table_df, summary


# === STREAMLIT RENDERER ===
def render_household_dashboard(selected_lgas=None, start_date=None, end_date=None, selected_genders=None, zerodose_path=None, visit_path=None, top_n=None, precomputed_data=None, chart_callback=None):
    
    if precomputed_data:
        figs, table_df, summary = precomputed_data
    else:
        try:
            # Pass top_n=None to show ALL settlements
            figs, table_df, summary = get_household_dashboard(
                selected_lgas, start_date, end_date, selected_genders,
                zerodose_path, visit_path, top_n=None
            )
        except Exception as e:
            st.error(f"Error loading dashboard: {e}")
            return

    if not figs:
        st.warning("No data found matching filters.")
        return

    st.markdown("### 🏘️ Settlement & Household Analysis")
    k1, k2, k3 = st.columns(3)
    k1.metric("Settlements Mapped", f"{summary.get('total_settlements', 0):,}")
    k2.metric("Total Active Burden", f"{summary.get('total_active_burden', 0):,}")
    # k3.metric("Avg Distance to HF", f"{summary.get('avg_dist', 0):.1f} km")
    st.divider()

    def display(fig, title, idx):
        if chart_callback: chart_callback(fig, title, f"house_prof_{idx}")
        else: st.plotly_chart(fig, use_container_width=True)

    fig_dict = {title: fig for title, fig in figs}

    # GRID LAYOUT
    c1, c2 = st.columns(2)
    with c1:
        if "Settlements by Total Zero-dose Children" in fig_dict: 
            display(fig_dict["Settlements by Total Zero-dose Children"], "All Settlements", 0)
    with c2:
        if "Settlements by Unresolved Cases" in fig_dict: 
            display(fig_dict["Settlements by Unresolved Cases"], "All Settlements", 1)

    st.markdown("---")
    
    # --- NEW ADDITIONS: MICRO-PLANNING INSIGHTS ---
    # st.markdown("### 🗺️ Micro-Planning Insights")
    
    # Row 3: Ward & Pareto (REMOVED)
    # Ward Level Burden & Settlement Burden Concentration

    # Row 4: Distance & Resolution (REMOVED)
    # Distance Distribution & Resolution Success vs Distance

    # Row 5: Clustering & Barriers
    st.markdown("##### Clustering & Barriers")
    c7, c8 = st.columns(2)
    with c7:
        if "Settlement Prioritization (Scatter)" in fig_dict:
            display(fig_dict["Settlement Prioritization (Scatter)"], "Priority Matrix (Burden vs Distance)", 2)
    with c8:
        if "Barriers by LGA" in fig_dict: display(fig_dict["Barriers by LGA"], "Dominant ZD Barriers", 8)

    with st.expander("Detailed Priority List"):
        st.dataframe(table_df.style.background_gradient(subset=['Active Burden'], cmap='Reds'))

    st.markdown("**Methodology:** Priority score = `0.7 * normalized_zero_dose + 0.3 * normalized_dropout`.")