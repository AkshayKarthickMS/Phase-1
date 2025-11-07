# dashboards/dashboard_7_risk.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import re

@st.cache_resource
def get_dashboard_risk():
    # ---------- CONFIG ----------
    ZERODOSE_CSV = "data/zerodose.csv"
    VISIT_CSV = "data/facility_visit.csv"
    RANDOM_STATE = 42
    CLUSTERS = 4
    TOP_SETTLEMENTS = 100
    TOP_N_TABLE = 200

    # ---------- HELPERS ----------
    def parse_age_to_months(age_str):
        if pd.isna(age_str): return np.nan
        parts = {'year': 0, 'month': 0, 'week': 0}
        try:
            for part in str(age_str).split(','):
                p = part.strip().lower()
                if 'year' in p:
                    parts['year'] = int(re.sub(r'[^0-9-]', '', p.split('year')[0]) or 0)
                elif 'month' in p:
                    parts['month'] = int(re.sub(r'[^0-9-]', '', p.split('month')[0]) or 0)
                elif 'week' in p:
                    parts['week'] = int(re.sub(r'[^0-9-]', '', p.split('week')[0]) or 0)
            return parts['year'] * 12 + parts['month'] + parts['week'] / 4.345
        except Exception:
            return np.nan

    def parse_distance_km(dist_str):
        if pd.isna(dist_str): return np.nan
        try:
            s = str(dist_str).lower().replace('km', '').replace('kms','').strip()
            s = re.sub(r'[^0-9\.\-]', '', s)
            return float(s) if s != '' else np.nan
        except Exception:
            return np.nan

    def parse_date(col):
        return pd.to_datetime(col, dayfirst=True, errors='coerce')

    # ---------- LOAD DATA ----------
    zd = pd.read_csv(ZERODOSE_CSV)
    vis = pd.read_csv(VISIT_CSV)
    zd.columns = zd.columns.str.strip()
    vis.columns = vis.columns.str.strip()

    zd['enrollment_date'] = parse_date(zd.get('Enrollment Date'))
    zd['estimated_age_months'] = zd.get('Estimated Current Age').apply(parse_age_to_months)
    zd['age_at_enroll_months'] = zd.get('Age at Enrollment').apply(parse_age_to_months)
    zd['status_std'] = zd.get('Status').astype(str).str.strip().str.lower()
    zd['gender_std'] = zd.get('Gender').astype(str).str.strip().str.lower()
    zd['woman_or_child_std'] = zd.get('Woman or child').astype(str).str.strip().str.lower()
    zd['settlement_std'] = zd.get('Settlement').astype(str).str.strip().str.title()
    zd['lga_std'] = zd.get('LGA').astype(str).str.strip().str.title()
    zd['distance_km'] = zd.get('Distance to HF').apply(parse_distance_km)

    zd_child = zd[zd['woman_or_child_std'] == 'child'].copy()
    zd_child['dropoff'] = zd_child['status_std'].apply(lambda x: 1 if x != 'resolved' else 0)

    vis['visit_date_parsed'] = parse_date(vis.get('visit_date'))
    vis['lga_name_std'] = vis.get('lga_name').astype(str).str.strip().str.title()

    def compute_reengaged(row):
        if pd.isna(row['enrollment_date']): return 0
        lga = row['lga_std']
        after = vis[(vis['lga_name_std'] == lga) &
                    (vis['visit_date_parsed'] >= row['enrollment_date'])]
        return 1 if len(after) > 0 else 0

    zd_child['reengaged'] = zd_child.apply(compute_reengaged, axis=1)

    # ---------- FEATURE ENGINEERING ----------
    zd_child['reason_clean'] = zd_child.get('Reason for ZD', '').astype(str).str.replace(r'[\s]+', ' ', regex=True).str.strip()
    zd_child['reason_list'] = zd_child['reason_clean'].apply(lambda s: re.split(r'[,\s]+', s) if s and s!='nan' else [])
    all_reasons = pd.Series([r for sub in zd_child['reason_list'] for r in sub]).value_counts()
    top_reasons = all_reasons.head(10).index.tolist() if len(all_reasons) > 0 else []

    for reason in top_reasons:
        zd_child[f'reason_{reason}'] = zd_child['reason_list'].apply(lambda lst: 1 if reason in lst else 0)

    num_features = ['estimated_age_months','age_at_enroll_months','distance_km']
    for f in num_features:
        zd_child[f] = zd_child[f].fillna(zd_child[f].median())

    zd_child['gender_enc'] = zd_child['gender_std'].map({'male':0,'female':1}).fillna(0)
    feature_cols_num = ['estimated_age_months','age_at_enroll_months','distance_km','gender_enc'] + [f'reason_{r}' for r in top_reasons]

    zd_child['settlement_trim'] = zd_child['settlement_std'].apply(lambda s: s if s in zd_child['settlement_std'].value_counts().head(TOP_SETTLEMENTS).index.tolist() else 'OTHER')
    cat_cols = ['settlement_trim','lga_std']
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_mat = ohe.fit_transform(zd_child[cat_cols])
    cat_names = list(ohe.get_feature_names_out(cat_cols))

    X_base = zd_child[feature_cols_num].copy().values
    X = np.hstack([X_base, cat_mat])
    scaler = StandardScaler()
    X[:, :len(feature_cols_num)] = scaler.fit_transform(X[:, :len(feature_cols_num)])

    # ---------- MODELING ----------
    y_drop = zd_child['dropoff'].values
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    X_train, X_test, y_train, y_test = train_test_split(X, y_drop, test_size=0.2, random_state=RANDOM_STATE, stratify=y_drop)
    rf.fit(X_train, y_train)
    drop_proba_all = rf.predict_proba(X)[:,1]
    zd_child['dropoff_proba'] = drop_proba_all

    y_re = zd_child['reengaged'].values
    if len(np.unique(y_re)) > 1:
        log = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        log.fit(X_train, y_train)
        zd_child['reengage_proba'] = log.predict_proba(X)[:,1]
    else:
        zd_child['reengage_proba'] = np.nan

    # ---------- CLUSTERING ----------
    kmeans = KMeans(n_clusters=CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    zd_child['cluster'] = kmeans.fit_predict(X)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    zd_child['pca1'] = X_pca[:,0]
    zd_child['pca2'] = X_pca[:,1]

    # ---------- VISUALIZATIONS ----------
    fig_prob = px.histogram(zd_child, x='dropoff_proba', nbins=30, title='Predicted Dropoff Probability Distribution')
    fig_feat = px.bar(x=rf.feature_importances_[:12], y=feature_cols_num[:12], orientation='h', title='Top Feature Importances')
    fig_pca = px.scatter(zd_child, x='pca1', y='pca2', color='cluster', size='dropoff_proba',
                         title='Clusters by PCA (Dropoff Probability Size)',
                         hover_data=['Settlement','LGA','dropoff_proba','reengage_proba'])
    top_table = zd_child[['ID','Settlement','LGA','dropoff_proba','reengage_proba','cluster']].sort_values('dropoff_proba', ascending=False).head(TOP_N_TABLE)

    return fig_prob, fig_feat, fig_pca, top_table
