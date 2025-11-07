# dashboard_vaccine_field.py
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import os

# ---------------------------
# Cached heavy computation
# ---------------------------
@st.cache_resource
def get_vaccine_dashboard():
    """Preload, process, and return all figures for the Vaccine Field Dashboard."""
    INPUT_CSV = "data/cleaned_facility_visit.csv"   # change path if needed
    TOP_N_LGA = 8
    TOP_N_VACCINES = 12
    AGE_BIN_EDGES = [0, 1.5, 12, 24, np.inf]
    AGE_BIN_LABELS = [
        "≤6w-11m (Due Penta)",
        "12-23m (Active ZD)",
        "24-59m (Overaged ZD)", 
        "5y+ (Older children)"
    ]

    if not os.path.exists(INPUT_CSV):
        st.error(f"❌ Input CSV not found: {INPUT_CSV}")
        return []

    df = pd.read_csv(INPUT_CSV, dtype=str).fillna('')

    # compute numeric age
    if 'age_total_months' not in df.columns:
        def compute_months(row):
            try:
                yrs = float(row.get('age_years', 0)) if row.get('age_years','')!='' else 0.0
                mos = float(row.get('age_months', 0)) if row.get('age_months','')!='' else 0.0
                wks = float(row.get('age_weeks', 0)) if row.get('age_weeks','')!='' else 0.0
                return yrs*12 + mos + wks/4.345
            except Exception:
                return np.nan
        df['age_total_months'] = df.apply(compute_months, axis=1).astype(float)
    else:
        df['age_total_months'] = pd.to_numeric(df['age_total_months'], errors='coerce')

    # normalize gender
    df['gender'] = df.get('gender', 'unknown').astype(str).str.lower().str.strip()
    df.loc[df['gender'].isin(['', 'nan']), 'gender'] = 'unknown'

    # standardize LGA
    if 'lga_name' in df.columns:
        df['LGA'] = df['lga_name']
    elif 'LGA' not in df.columns:
        df['LGA'] = ''

    # detect vaccine columns
    known_vaccines = [
        "BCG","HPV","HepB_0","IPV_1","IPV_2","Measles_1","Measles_2",
        "OPV_0","OPV_1","OPV_2","OPV_3",
        "PCV_1","PCV_2","PCV_3",
        "Penta_1","Penta_2","Penta_3",
        "Rota_1","Rota_2","Rota_3",
        "VitaminA_1","VitaminA_2","YF","meningitis"
    ]
    vaccine_cols = [v for v in known_vaccines if v in df.columns]

    if len(vaccine_cols) == 0 and 'vaccines_administered' in df.columns:
        parsed = df['vaccines_administered'].astype(str).str.replace(r'[\{\}]','', regex=True).str.split(r'\s*,\s*')
        extra = set(v for sub in parsed.tolist() for v in sub if v and v.strip() != '')
        vaccine_cols = sorted(extra)
        for v in vaccine_cols:
            df[v] = parsed.apply(lambda L: 1 if isinstance(L, list) and v in L else 0)
    else:
        for v in vaccine_cols:
            df[v] = pd.to_numeric(df[v].replace('','0').replace('nan','0'), errors='coerce').fillna(0).astype(int)

    # derived columns
    df['age_group'] = pd.cut(df['age_total_months'], bins=AGE_BIN_EDGES, labels=AGE_BIN_LABELS, right=False)
    df['vacc_count'] = df[vaccine_cols].sum(axis=1)

    # summaries
    coverage = pd.DataFrame({
        'vaccine': vaccine_cols,
        'count': [int(df[v].sum()) for v in vaccine_cols],
        'coverage_pct': [float(df[v].mean()*100) for v in vaccine_cols]
    }).sort_values('coverage_pct', ascending=False)

    coverage_gender = df.groupby('gender')[vaccine_cols].mean().T * 100
    coverage_gender = coverage_gender.reset_index().rename(columns={'index':'vaccine'})

    coverage_age = df.groupby('age_group')[vaccine_cols].mean()*100
    coverage_age = coverage_age.reindex(index=AGE_BIN_LABELS).fillna(0)

    top_lgas = df['LGA'].value_counts().nlargest(TOP_N_LGA).index.tolist()
    coverage_lga = df[df['LGA'].isin(top_lgas)].groupby('LGA')[vaccine_cols].mean()*100
    coverage_lga = coverage_lga.loc[top_lgas]

    # Sankey data
    top_vaccs = coverage.head(TOP_N_VACCINES)['vaccine'].tolist()
    sankey_nodes = list(df['gender'].unique()) + top_vaccs
    node_map = {n:i for i,n in enumerate(sankey_nodes)}
    sankey_source, sankey_target, sankey_value = [], [], []
    for g in df['gender'].unique():
        sub = df[df['gender']==g]
        for v in top_vaccs:
            cnt = int(sub[v].sum())
            if cnt > 0:
                sankey_source.append(node_map[g])
                sankey_target.append(node_map[v])
                sankey_value.append(cnt)

    # ---------------------------
    # Build Figures
    # ---------------------------
    px.defaults.template = "plotly_white"
    figs = []

    fig_cov = px.bar(coverage, x='vaccine', y='coverage_pct',
                     text=coverage['coverage_pct'].map(lambda x: f"{x:.1f}%"),
                     title="Overall Vaccine Coverage (%)",
                     labels={'coverage_pct':'Coverage (%)','vaccine':'Vaccine'})
    fig_cov.update_traces(textposition='outside')
    fig_cov.update_layout(yaxis=dict(range=[0,100]))
    figs.append(("Overall Coverage", fig_cov))

    cov_gender_melt = coverage_gender.melt(id_vars='vaccine', var_name='gender', value_name='coverage_pct')
    fig_gender = px.bar(cov_gender_melt, x='vaccine', y='coverage_pct', color='gender', barmode='group',
                        title="Vaccine Coverage by Gender (%)")
    fig_gender.update_layout(xaxis_tickangle=-45)
    figs.append(("Coverage by Gender", fig_gender))

    fig_age_heat = go.Figure(data=go.Heatmap(
        z=coverage_age.T.values,
        x=coverage_age.index.tolist(),
        y=coverage_age.columns.tolist(),
        colorscale='YlGnBu',
        colorbar=dict(title='Coverage %'),
    ))
    fig_age_heat.update_layout(title="Coverage by Age Group (%)", xaxis_title="Age Group", yaxis_title="Vaccine")
    figs.append(("Coverage by Age Group", fig_age_heat))

    fig_lga_heat = go.Figure(data=go.Heatmap(
        z=coverage_lga.values,
        x=coverage_lga.columns.tolist(),
        y=coverage_lga.index.tolist(),
        colorscale='OrRd',
        colorbar=dict(title='Coverage %'),
    ))
    fig_lga_heat.update_layout(title="Coverage by LGA (%)", xaxis_title="Vaccine", yaxis_title="LGA")
    figs.append(("Coverage by LGA", fig_lga_heat))

    if sankey_value:
        sankey_fig = go.Figure(data=[go.Sankey(
            node=dict(label=sankey_nodes, pad=15, thickness=20),
            link=dict(source=sankey_source, target=sankey_target, value=sankey_value)
        )])
        sankey_fig.update_layout(title=f"Sankey: Gender → Top {len(top_vaccs)} Vaccines (counts)", font_size=10)
        figs.append(("Sankey Gender→Vaccine", sankey_fig))

    top_v_for_facet = coverage.head(6)['vaccine'].tolist()
    melt_age = df.groupby('age_group')[top_v_for_facet].mean().reset_index().melt(
        id_vars='age_group', var_name='vaccine', value_name='coverage')
    fig_facet = px.bar(melt_age, x='age_group', y='coverage', color='vaccine', barmode='group',
                       title="Top vaccines across age groups (%)",
                       labels={'coverage':'Coverage %','age_group':'Age group'})
    figs.append(("Top Vaccines vs Age Group", fig_facet))

    summary = {
        "total_records": len(df),
        "unique_lgas": df['LGA'].nunique(),
        "vaccine_count": len(vaccine_cols)
    }

    return figs, summary


# ---------------------------
# Render for Streamlit Tab
# ---------------------------
def render_vaccine_dashboard():
    figs, summary = get_vaccine_dashboard()
    st.subheader("Vaccine Field Dashboard — Decision Support")
    st.info(f"**Records:** {summary['total_records']:,} | **Unique LGAs:** {summary['unique_lgas']:,} | **Vaccines tracked:** {summary['vaccine_count']}")
    for title, fig in figs:
        st.markdown(f"### {title}")
        st.plotly_chart(fig, use_container_width=True)
