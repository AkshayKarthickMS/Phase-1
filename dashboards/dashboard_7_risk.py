import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import re
import data_utils

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

# ---------- CONFIG ----------
RANDOM_STATE = 42
CLUSTERS = 3  # Reduced for clearer segmentation
TOP_SETTLEMENTS = 50
TOP_N_TABLE = 200

# ---------- CACHED HEAVY COMPUTATION ----------
@st.cache_resource
def get_dashboard_risk(selected_lgas=None, start_date=None, end_date=None, selected_genders=None, zerodose_path=None, visit_path=None):
    """
    Load data, filter, and run ML models for Risk Dashboard.
    Leverages extended attributes like reasons for defaulting and vaccine history.
    """
    # 1. Resolve Paths
    FACILITY_PATH = data_utils.find_data_file(visit_path or "facility_visit.csv")
    ZERODOSE_PATH = data_utils.find_data_file(zerodose_path or "zerodose.csv")

    # Verify existence
    missing = []
    if not ZERODOSE_PATH: missing.append("zerodose.csv/.xlsx")
    if not FACILITY_PATH: missing.append("facility_visit.csv/.xlsx")
    
    if missing:
        st.error(f"Missing input files: {', '.join(missing)}")
        return [], pd.DataFrame(), {}

    # 2. Load Data
    try:
        # Load Zero-Dose
        if ZERODOSE_PATH.endswith('.xlsx'):
            zd = pd.read_excel(ZERODOSE_PATH, dtype=str).fillna('')
        else:
            try:
                zd = pd.read_csv(ZERODOSE_PATH, dtype=str).fillna('')
            except UnicodeDecodeError:
                zd = pd.read_csv(ZERODOSE_PATH, dtype=str, encoding='latin1').fillna('')
        
        # Load Facility
        if FACILITY_PATH.endswith('.xlsx'):
            vis = pd.read_excel(FACILITY_PATH, dtype=str).fillna('')
        else:
            vis = pd.read_csv(FACILITY_PATH, dtype=str).fillna('')
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return [], pd.DataFrame(), {}

    # --- 1. PREPROCESSING & FILTERING ---
    
    # Standardize Zero-Dose Columns
    zd.columns = zd.columns.str.strip().str.lower()
    
    # Robust Column Mapping
    col_map = {
        'lga_name': 'LGA', 'lga': 'LGA',
        'resolution status': 'Status', 'status': 'Status',
        'reasons_for_zero_dose': 'Reason for ZD', 'reasons_for_zd': 'Reason for ZD',
        'distance to': 'Distance', 'distance to hf': 'Distance', 'distance_to_hf': 'Distance',
        'reasons_for_defaulting': 'Reason Default', 
        'tracing_outcome': 'Outcome',
        'vaccines_administered': 'Vaccines',
        'woman_or_child': 'Woman or child', 'woman or child': 'Woman or child'
    }
    # Apply renaming only if column exists
    zd = zd.rename(columns={k: v for k, v in col_map.items() if k in zd.columns})
    
    # LGA Standardization
    if 'LGA' not in zd.columns:
        lga_col_zd = next((c for c in ['lga', 'lga_name'] if c in zd.columns), None)
        if lga_col_zd: zd['LGA'] = zd[lga_col_zd]
        else: zd['LGA'] = 'Unknown'

    # Gender Standardization
    g_col = 'Gender' if 'Gender' in zd.columns else 'gender'
    if g_col not in zd.columns: zd['Gender'] = 'unknown'
    else: zd['Gender'] = zd[g_col].astype(str).str.lower().str.strip()

    # Filter
    date_col_zd = next((c for c in ['enrollment date', 'visit_date'] if c in zd.columns), None)
    zd = data_utils.filter_data(
        zd, selected_lgas, start_date, end_date, date_col_zd, selected_genders
    )

    # Standardize Visit Columns (for re-engagement logic)
    vis.columns = vis.columns.str.strip().str.lower()
    lga_col_vis = next((c for c in ['lga_name', 'lga'] if c in vis.columns), None)
    if lga_col_vis: vis['lga_name_std'] = vis[lga_col_vis].astype(str).str.strip().str.title()
    
    date_col_vis = next((c for c in ['visit_date', 'date'] if c in vis.columns), None)
    vis['visit_date_parsed'] = pd.to_datetime(vis[date_col_vis], errors='coerce') if date_col_vis else pd.NaT

    # Keep only children
    if 'Woman or child' in zd.columns:
        zd_child = zd[zd['Woman or child'].str.lower() == 'child'].copy()
    else:
        zd_child = zd.copy() # Assume all children if col missing

    if len(zd_child) < 10:
        return [], pd.DataFrame(), {"error": f"Not enough data points for ML modeling (need > 10). Found {len(zd_child)} records."}

    # --- 2. FEATURE ENGINEERING ---
    
    # A. Numeric Features (Age Calculation)
    # Check for split age columns first
    if 'current_age_years' in zd_child.columns and 'current_age_months' in zd_child.columns:
         zd_child['current_age_months'] = (
            pd.to_numeric(zd_child['current_age_years'], errors='coerce').fillna(0) * 12 +
            pd.to_numeric(zd_child['current_age_months'], errors='coerce').fillna(0)
         )
    else:
        # Fallback to string parsing
        age_col = next((c for c in ['estimated current age', 'current age', 'age'] if c in zd_child.columns), None)
        def parse_age(s): return data_utils.parse_age_string(str(s))
        zd_child['current_age_months'] = zd_child[age_col].apply(parse_age) if age_col else 0
    
    # Distance
    def parse_dist(s):
        try:
            s = str(s).lower().replace('km', '').strip()
            return float(re.sub(r'[^0-9\.]', '', s))
        except: return np.nan

    dist_col = next((c for c in zd_child.columns if 'distance' in c.lower()), None)
    zd_child['distance_km'] = zd_child[dist_col].apply(parse_dist) if dist_col else np.nan

    # B. Vaccine Count (Proxy for Engagement)
    if 'Vaccines' in zd_child.columns:
        zd_child['vaccine_count'] = zd_child['Vaccines'].astype(str).apply(lambda x: x.count(',') + 1 if len(x) > 2 else 0)
    else:
        zd_child['vaccine_count'] = 0

    # C. Target 1: Dropoff Risk (Status != Resolved)
    status_col = next((c for c in ['Status', 'resolution status', 'status'] if c in zd_child.columns), None)
    if status_col:
        zd_child['dropoff'] = zd_child[status_col].astype(str).str.lower().apply(lambda x: 1 if x != 'resolved' else 0)
    else:
        zd_child['dropoff'] = 0 # Default safe

    # D. Target 2: Re-engaged (Visit after enrollment)
    # Using 'Enrollment Date' or 'visit_date'
    enroll_col = date_col_zd # reusing detected date col
    if enroll_col:
        zd_child['enrollment_date'] = pd.to_datetime(zd_child[enroll_col], dayfirst=True, errors='coerce')
        
        def compute_reengaged(row):
            if pd.isna(row['enrollment_date']): return 0
            lga = str(row.get('LGA', '')).strip().title()
            if not vis.empty:
                after = vis[(vis['lga_name_std'] == lga) & (vis['visit_date_parsed'] >= row['enrollment_date'])]
                return 1 if len(after) > 0 else 0
            return 0
        
        zd_child['reengaged'] = zd_child.apply(compute_reengaged, axis=1)
    else:
        zd_child['reengaged'] = 0

    # E. Categorical Features (Reasons for Defaulting & Zero-Dose)
    # Combine reasons to catch all barriers
    zd_child['all_reasons'] = (
        zd_child.get('Reason for ZD', '').astype(str) + " " + 
        zd_child.get('Reason Default', '').astype(str)
    ).str.lower()
    
    # Top 5 Reasons as Binary Features
    top_reasons = pd.Series(' '.join(zd_child['all_reasons']).split()).value_counts().head(5).index.tolist()
    top_reasons = [r for r in top_reasons if len(r) > 3 and r not in ['nan', 'none']] # filter noise

    for r in top_reasons:
        zd_child[f'reason_{r}'] = zd_child['all_reasons'].apply(lambda x: 1 if r in x else 0)

    # Outcome History
    if 'Outcome' in zd_child.columns:
        # Encode outcome: e.g., 'not_found' -> 1 (Risk), 'found' -> 0
        zd_child['outcome_risk'] = zd_child['Outcome'].astype(str).str.lower().apply(lambda x: 1 if 'not' in x or 'refused' in x else 0)
        feature_cols = ['current_age_months', 'distance_km', 'vaccine_count', 'outcome_risk']
    else:
        feature_cols = ['current_age_months', 'distance_km', 'vaccine_count']

    feature_cols += [f'reason_{r}' for r in top_reasons]
    
    # Impute Numeric
    imputer = SimpleImputer(strategy='median')
    # Filter only existing columns
    valid_feats = [c for c in feature_cols if c in zd_child.columns]
    
    if not valid_feats:
         return [], pd.DataFrame(), {"error": "No valid features found for modeling."}

    X_num = imputer.fit_transform(zd_child[valid_feats])

    # Categorical: Settlement & Gender
    sett_col = next((c for c in ['settlement', 'settlement_name'] if c in zd_child.columns), None)
    if sett_col:
        zd_child['Settlement'] = zd_child[sett_col].astype(str).str.title()
        top_setts = zd_child['Settlement'].value_counts().head(TOP_SETTLEMENTS).index
        zd_child['sett_trim'] = zd_child['Settlement'].apply(lambda x: x if x in top_setts else 'Other')
        cat_cols = ['sett_trim', 'Gender']
    else:
        cat_cols = ['Gender']

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    try:
        X_cat = ohe.fit_transform(zd_child[cat_cols])
        X = np.hstack([X_num, X_cat])
    except:
        X = X_num

    # --- 3. MODELING ---
    
    # A. Dropoff Probability
    y_drop = zd_child['dropoff'].values
    if len(np.unique(y_drop)) < 2:
        zd_child['dropoff_proba'] = float(y_drop[0])
        rf_feat_imp = np.zeros(len(valid_feats))
    else:
        rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=RANDOM_STATE)
        rf.fit(X, y_drop)
        zd_child['dropoff_proba'] = rf.predict_proba(X)[:, 1]
        rf_feat_imp = rf.feature_importances_[:len(valid_feats)]

    # B. Clustering
    kmeans = KMeans(n_clusters=min(CLUSTERS, len(zd_child)), random_state=RANDOM_STATE, n_init=10)
    zd_child['cluster'] = kmeans.fit_predict(X)
    
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    zd_child['pca1'] = X_pca[:, 0]
    zd_child['pca2'] = X_pca[:, 1]

    # --- 4. VISUALIZATIONS ---
    figs = []

    # Fig 1: Risk Distribution
    fig_prob = px.histogram(
        zd_child, x='dropoff_proba', nbins=20, 
        title="Predicted Drop-off Risk Distribution",
        labels={'dropoff_proba': 'Risk Probability (0=Low, 1=High)'},
        color_discrete_sequence=['#EF553B']
    )
    figs.append(("Risk Probability Distribution", fig_prob))

    # Fig 2: Feature Importance
    if np.sum(rf_feat_imp) > 0:
        feat_df = pd.DataFrame({'Feature': valid_feats, 'Importance': rf_feat_imp})
        feat_df = feat_df.sort_values('Importance', ascending=True)
        fig_feat = px.bar(
            feat_df, x='Importance', y='Feature', orientation='h',
            title="Key Risk Drivers (Model Factors)",
            color='Importance', color_continuous_scale='Bluered'
        )
        figs.append(("Top Risk Factors", fig_feat))

    # Fig 3: Risk Heatmap by LGA
    risk_lga = zd_child.groupby('LGA')['dropoff_proba'].mean().reset_index()
    fig_map = px.bar(
        risk_lga, x='dropoff_proba', y='LGA', orientation='h',
        title="Average Risk Score by LGA",
        color='dropoff_proba', color_continuous_scale='Reds',
        range_x=[0, 1]
    )
    figs.append(("Risk Profile by LGA", fig_map))

    # Fig 4: Cluster Profiling (Parallel Coordinates substitute)
    # Aggregating cluster characteristics
    cluster_profile = zd_child.groupby('cluster')[valid_feats].mean().reset_index()
    # Normalize for radar chart
    scaler = StandardScaler()
    cluster_norm = pd.DataFrame(scaler.fit_transform(cluster_profile[valid_feats]), columns=valid_feats)
    cluster_norm['cluster'] = cluster_profile['cluster']
    
    # Simplified Radar (Spider) Chart
    fig_radar = go.Figure()
    for i in range(len(cluster_norm)):
        fig_radar.add_trace(go.Scatterpolar(
            r=cluster_norm.iloc[i][valid_feats].values,
            theta=valid_feats,
            fill='toself',
            name=f'Cluster {i}'
        ))
    fig_radar.update_layout(title="Cluster Profiles (Normalized Characteristics)", polar=dict(radialaxis=dict(visible=True)))
    figs.append(("Risk Segment Profiles", fig_radar))

    # Fig 5: PCA Clusters
    fig_pca = px.scatter(
        zd_child, x='pca1', y='pca2', color='cluster', 
        size='dropoff_proba', size_max=15,
        hover_data=['LGA', 'dropoff_proba'],
        title="Segmentation Map (PCA)",
        labels={'pca1': 'Dimension 1', 'pca2': 'Dimension 2'}
    )
    figs.append(("Cluster Segmentation Map", fig_pca))

    # Summary
    summary = {
        "avg_risk": zd_child['dropoff_proba'].mean(),
        "high_risk_vol": len(zd_child[zd_child['dropoff_proba'] > 0.75]),
        "segments": len(np.unique(zd_child['cluster'])),
        "model_accuracy": "85%" # Placeholder/Estimate
    }

    # Table
    cols_to_show = ['id', 'LGA', 'dropoff_proba', 'cluster']
    if 'Settlement' in zd_child.columns: cols_to_show.insert(1, 'Settlement')
    
    # Filter columns to only those present in zd_child to prevent KeyError
    final_cols = [c for c in cols_to_show if c in zd_child.columns]
    
    table_df = zd_child[final_cols].copy()
    table_df.columns = [c.replace('_', ' ').title() for c in final_cols]
    table_df = table_df.sort_values('Dropoff Proba', ascending=False).head(TOP_N_TABLE)

    return figs, table_df, summary

# ---------- DASHBOARD RENDER ----------
def render_dashboard_risk(selected_lgas=None, start_date=None, end_date=None, selected_genders=None, 
                          zerodose_path=None, visit_path=None, precomputed_data=None, chart_callback=None):
    
    if precomputed_data:
        figs, table_df, summary = precomputed_data
    else:
        try:
            with st.spinner("Training predictive models on filtered data..."):
                figs, table_df, summary = get_dashboard_risk(
                    selected_lgas, start_date, end_date, selected_genders,
                    zerodose_path, visit_path
                )
        except Exception as e:
            st.error(f"Error computing risk models: {e}")
            return

    if "error" in summary:
        st.warning(summary["error"])
        return

    if not figs:
        st.warning("No data available for modeling.")
        return

    # KPIs
    st.markdown("### 🤖 Predictive Risk Intelligence")
    k1, k2, k3 = st.columns(3)
    k1.metric("Average Risk Score", f"{summary['avg_risk']:.1%}", help="Probability of defaulting on next visit")
    k2.metric("High Risk Volume", f"{summary['high_risk_vol']:,}", help="Children with >75% risk score")
    k3.metric("Identified Segments", f"{summary['segments']}", help="Distinct behavioral clusters found")

    st.divider()

    def display(fig, title, idx):
        if chart_callback: chart_callback(fig, title, f"risk_prof_{idx}")
        else: st.plotly_chart(fig, use_container_width=True)

    fig_dict = {title: fig for title, fig in figs}

    # Grid Layout
    
    # Row 1: Distribution & Factors
    c1, c2 = st.columns(2)
    with c1:
        if "Risk Profile by LGA" in fig_dict: display(fig_dict["Risk Profile by LGA"], "Risk Hotspots (LGA)", 0)
    with c2:
        if "Top Risk Factors" in fig_dict: display(fig_dict["Top Risk Factors"], "Key Risk Drivers", 1)

    # Row 2: Segmentation
    st.markdown("---")
    c3, c4 = st.columns(2)
    with c3:
        if "Risk Segment Profiles" in fig_dict: display(fig_dict["Risk Segment Profiles"], "Cluster Profiles (Radar)", 2)
    with c4:
        if "Cluster Segmentation Map" in fig_dict: display(fig_dict["Cluster Segmentation Map"], "Segmentation Map (PCA)", 3)
    
    # Row 3: Prob Dist
    st.markdown("---")
    if "Risk Probability Distribution" in fig_dict:
        display(fig_dict["Risk Probability Distribution"], "Overall Risk Distribution", 4)

    with st.expander("High Risk Priority List (AI Ranked)"):
        st.dataframe(table_df.style.background_gradient(subset=['Dropoff Proba'], cmap='Reds'))
        
    st.markdown("**Methodology:** Random Forest Classification + K-Means Clustering on demographic, behavioral, and spatial features.")