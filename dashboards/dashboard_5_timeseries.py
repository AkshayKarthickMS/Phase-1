# dashboard_5_timeseries.py
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import streamlit as st

# ---------- CONFIG ----------
ZERODOSE_CSV = "data/zerodose.csv"
VISIT_CSV = "data/facility_visit.csv"
LONG_ENROLL_DAYS = 180
PENTA3_DUE_MONTHS = 4
MEASLES_DUE_MONTHS = 9


# ---------- HELPERS ----------
def parse_age_to_months(age_str):
    if pd.isna(age_str):
        return np.nan
    parts = {'year': 0, 'month': 0, 'week': 0}
    try:
        for part in str(age_str).split(','):
            part = part.strip().lower()
            if 'year' in part:
                parts['year'] = int(part.split('year')[0].strip())
            elif 'month' in part:
                parts['month'] = int(part.split('month')[0].strip())
            elif 'week' in part:
                parts['week'] = int(part.split('week')[0].strip())
        return parts['year'] * 12 + parts['month'] + parts['week'] / 4.345
    except Exception:
        return np.nan


def parse_date(col):
    return pd.to_datetime(col, dayfirst=True, errors='coerce')


# ---------- CACHED DATA PREP ----------
@st.cache_data(show_spinner=False)
def load_and_process_data():
    zd = pd.read_csv(ZERODOSE_CSV)
    vis = pd.read_csv(VISIT_CSV)

    zd.columns = zd.columns.str.strip()
    vis.columns = vis.columns.str.strip()

    zd['Enrollment Date_parsed'] = parse_date(zd['Enrollment Date'])
    zd['Estimated_current_age_months'] = zd['Estimated Current Age'].apply(parse_age_to_months)
    zd['Age_at_enrollment_months'] = zd['Age at Enrollment'].apply(parse_age_to_months)
    zd['Status_std'] = zd['Status'].astype(str).str.strip().str.lower()
    zd['Settlement_std'] = zd['Settlement'].astype(str).str.strip().str.title()
    zd['LGA_std'] = zd['LGA'].astype(str).str.strip().str.title()

    today = pd.to_datetime(datetime.now().date())
    zd['days_since_enrollment'] = (today - zd['Enrollment Date_parsed']).dt.days

    zd['long_enrolled_unresolved'] = (
        (zd['days_since_enrollment'] >= LONG_ENROLL_DAYS)
        & (zd['Status_std'] != 'resolved')
    )

    settlement_summary = zd.groupby(['Settlement_std', 'LGA_std']).agg(
        total_enrolled=('ID', 'nunique'),
        unresolved_count=('ID', lambda s: zd.loc[s.index, 'Status_std'].ne('resolved').sum()),
        long_enrolled_unresolved_count=('ID', lambda s: zd.loc[s.index, 'long_enrolled_unresolved'].sum()),
        avg_days_since_enrollment=('days_since_enrollment', 'mean'),
        avg_estimated_age_months=('Estimated_current_age_months', 'mean')
    ).reset_index()

    zd['enroll_month'] = zd['Enrollment Date_parsed'].dt.to_period('M').astype(str)
    enroll_timeline = zd.groupby(['enroll_month', 'Status_std']).agg(
        count=('ID', 'nunique')
    ).reset_index()
    enroll_pivot = enroll_timeline.pivot(index='enroll_month', columns='Status_std', values='count').fillna(0).reset_index()
    enroll_pivot['enroll_month_dt'] = pd.to_datetime(enroll_pivot['enroll_month'], format='%Y-%m', errors='coerce')
    enroll_pivot = enroll_pivot.sort_values('enroll_month_dt')

    zd['overdue_penta3'] = zd['Estimated_current_age_months'] >= PENTA3_DUE_MONTHS
    zd['overdue_measles'] = zd['Estimated_current_age_months'] >= MEASLES_DUE_MONTHS

    overdue_summary = zd.groupby(['Settlement_std', 'LGA_std']).agg(
        num_children=('ID', 'nunique'),
        overdue_penta3_count=('overdue_penta3', 'sum'),
        overdue_measles_count=('overdue_measles', 'sum'),
        unresolved_count=('Status_std', lambda s: s.ne('resolved').sum())
    ).reset_index()

    sett_summary = settlement_summary.merge(
        overdue_summary[['Settlement_std', 'overdue_penta3_count', 'overdue_measles_count']],
        on='Settlement_std', how='left'
    ).fillna(0)

    sett_summary['penta3_overdue_rate'] = sett_summary['overdue_penta3_count'] / sett_summary['total_enrolled']
    sett_summary['measles_overdue_rate'] = sett_summary['overdue_measles_count'] / sett_summary['total_enrolled']
    sett_summary['long_unresolved_rate'] = sett_summary['long_enrolled_unresolved_count'] / sett_summary['total_enrolled']

    return zd, sett_summary, enroll_pivot


# ---------- CACHED VISUALS ----------
@st.cache_resource(show_spinner=False)
def build_figures():
    zd, sett_summary, enroll_pivot = load_and_process_data()

    fig_enroll = px.area(
        enroll_pivot,
        x='enroll_month',
        y=[c for c in enroll_pivot.columns if c not in ('enroll_month', 'enroll_month_dt')],
        title='Zero-dose Enrollment Over Time (by status)',
        labels={'value': 'Number of Enrollments', 'enroll_month': 'Month'}
    )
    fig_enroll.update_layout(legend_title_text='Status')

    top_long_unresolved = sett_summary.sort_values('long_enrolled_unresolved_count', ascending=False).head(30)
    fig_long = px.bar(
        top_long_unresolved,
        x='long_enrolled_unresolved_count',
        y='Settlement_std',
        orientation='h',
        color='LGA_std',
        hover_data=['total_enrolled', 'avg_days_since_enrollment', 'long_unresolved_rate'],
        title=f"Top settlements with children enrolled >{LONG_ENROLL_DAYS} days & unresolved"
    )
    fig_long.update_layout(yaxis={'categoryorder': 'total ascending'})

    table_df = sett_summary.sort_values('long_enrolled_unresolved_count', ascending=False).head(200)
    return fig_enroll, fig_long, table_df


# ---------- DASHBOARD RENDER ----------
def show_timeseries_dashboard():
    st.header("📈 Time-Series & Follow-up Dashboard")
    st.caption(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    fig_enroll, fig_long, table_df = build_figures()

    st.plotly_chart(fig_enroll, use_container_width=True)
    st.plotly_chart(fig_long, use_container_width=True)

    st.subheader("Settlement-level Summary (Top Rows)")
    st.dataframe(table_df, use_container_width=True, height=500)

    st.markdown(f"""
    **Notes:**
    - Overdue flags are computed heuristically:
      - Penta3 due if child ≥ {PENTA3_DUE_MONTHS} months
      - Measles due if ≥ {MEASLES_DUE_MONTHS} months  
    - Use child-level vaccination records to refine overdue counts.
    """)


# ---------- FOR TESTING LOCALLY ----------
if __name__ == "__main__":
    import streamlit.web.cli as stcli
    stcli._main_run_clExplicit("dashboard_5_timeseries.py", "streamlit run")
