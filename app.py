# app.py
"""
Unified Vaccine Analytics Dashboard (Streamlit)
- Preloads 7 dashboard modules (cached) so tab-switching is instant.
- Expects a `dashboards/` package with modules:
    dashboard_vaccine_field.py
    dashboard_gender.py
    dashboard_3_age.py
    dashboard_4_household.py
    dashboard_5_timeseries.py
    dashboard_6_additional.py
    dashboard_7_risk.py
- Run: `streamlit run app.py`
"""

import streamlit as st
from textwrap import dedent

st.set_page_config(page_title="Unified Vaccine Analytics Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- Imports of your dashboard modules (these are the files we converted) ---
# Each module exposes at least one cached "get_*" or "build_*" function to precompute heavy work,
# and many include a light "render_*" function to display content in a tab.
from dashboards.dashboard_vaccine_field import get_vaccine_dashboard, render_vaccine_dashboard
from dashboards.dashboard_gender import get_gender_dashboard, render_gender_dashboard
from dashboards.dashboard_3_age import get_dashboard_age
from dashboards.dashboard_4_household import get_household_dashboard, render_household_dashboard
from dashboards.dashboard_5_timeseries import build_figures as build_timeseries_figures, show_timeseries_dashboard
from dashboards.dashboard_6_additional import build_figures as build_additional_figures, show_additional_dashboard
from dashboards.dashboard_7_risk import get_dashboard_risk

# --- Sidebar (logo, dataset paths, quick controls) ---
import os

if os.path.exists("client_logo.png"):
    st.sidebar.image("client_logo.png", width=160)
st.sidebar.title("Unified Dashboard")
st.sidebar.markdown("Preloaded dashboards — switch tabs for instant views.")

# Optional: let user change csv filenames if needed
st.sidebar.header("Data files (optional override)")
zd_path = st.sidebar.text_input("Zero-dose CSV", value="zerodose.csv")
vis_path = st.sidebar.text_input("Facility visit CSV", value="facility_visit.csv")
facility_csv_override = vis_path  # keep names consistent where called

# Preload toggle (but we will preload by default)
preload_on_start = st.sidebar.checkbox("Preload all dashboards on start", value=True)

# Small instruction
st.sidebar.markdown(dedent("""
**Notes**
- Large CSVs and models run once (cached) and are reused.
- To clear cached computations: _Streamlit menu → Clear cache and rerun_.
"""))


# --- Main header ---
st.title("Unified Vaccine Analytics Dashboard")
st.write("All dashboards are precomputed and cached for instant switching. Use the sidebar to change source file names if needed.")

# --- Preload all dashboards (cached functions) ---
preload_errors = {}
preloaded = {}

if preload_on_start:
    with st.spinner("Preloading all dashboards — this runs heavy computations once... ⏳"):
        # Vaccine field (figs, summary)
        try:
            figs_vaccine, summary_vaccine = get_vaccine_dashboard()
            preloaded['vaccine'] = (figs_vaccine, summary_vaccine)
        except Exception as e:
            preload_errors['vaccine'] = str(e)
            preloaded['vaccine'] = None

        # Gender
        try:
            # get_gender_dashboard accepts optional file paths in the module; pass overrides if needed
            figs_gender, summary_gender, insights_gender, data_gender = get_gender_dashboard(facility_path=facility_csv_override, zerodose_path=zd_path)
            preloaded['gender'] = (figs_gender, summary_gender, insights_gender, data_gender)
        except Exception as e:
            preload_errors['gender'] = str(e)
            preloaded['gender'] = None

        # Age
        try:
            figs_age, zd_age_df, facility_age_df = get_dashboard_age()
            preloaded['age'] = (figs_age, zd_age_df, facility_age_df)
        except Exception as e:
            preload_errors['age'] = str(e)
            preloaded['age'] = None

        # Household / settlements
        try:
            figs_house, table_house, summary_house = get_household_dashboard(zerodose_path=zd_path, visit_path=facility_csv_override)
            preloaded['household'] = (figs_house, table_house, summary_house)
        except Exception as e:
            preload_errors['household'] = str(e)
            preloaded['household'] = None

        # Time-series / follow-up
        try:
            fig_enroll_ts, fig_long_ts, table_ts = build_timeseries_figures()
            preloaded['timeseries'] = (fig_enroll_ts, fig_long_ts, table_ts)
        except Exception as e:
            preload_errors['timeseries'] = str(e)
            preloaded['timeseries'] = None

        # Additional (heatmap, network, trend)
        try:
            fig_heat, fig_net, fig_trend = build_additional_figures()
            preloaded['additional'] = (fig_heat, fig_net, fig_trend)
        except Exception as e:
            preload_errors['additional'] = str(e)
            preloaded['additional'] = None

        # Risk & ML
        try:
            fig_prob, fig_feat, fig_pca, table_risk = get_dashboard_risk()
            preloaded['risk'] = (fig_prob, fig_feat, fig_pca, table_risk)
        except Exception as e:
            preload_errors['risk'] = str(e)
            preloaded['risk'] = None

    if preload_errors:
        st.warning("Some dashboards failed to preload. They may still run when selected. See sidebar for details.")
        with st.expander("Preload errors (click to expand)"):
            for k, v in preload_errors.items():
                st.write(f"**{k}**: {v}")

# --- Tabs layout ---
tab_labels = [
    "1 • Vaccine Field",
    "2 • Gender Analysis",
    "3 • Age Analytics",
    "4 • Household / Settlements",
    "5 • Time-series & Follow-up",
    "6 • Additional Analytics",
    "7 • Risk & Segmentation"
]
tabs = st.tabs(tab_labels)

# Tab 1: Vaccine Field
with tabs[0]:
    st.header("Vaccine Field Dashboard")
    if preloaded.get('vaccine'):
        figs_vaccine, summary_vaccine = preloaded['vaccine']
        st.info(f"Records: {summary_vaccine.get('total_records', 'N/A'):,} | Unique LGAs: {summary_vaccine.get('unique_lgas','N/A'):,} | Vaccines tracked: {summary_vaccine.get('vaccine_count','N/A')}")
        for title, fig in figs_vaccine:
            st.markdown(f"### {title}")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Vaccine dashboard not preloaded or failed. Attempting to render using render_vaccine_dashboard()...")
        try:
            render_vaccine_dashboard()
        except Exception as e:
            st.error(f"Failed to render vaccine dashboard: {e}")

# Tab 2: Gender Analysis
with tabs[1]:
    st.header("Zero-Dose Gender Analytics")
    if preloaded.get('gender'):
        figs_gender, summary_gender, insights_gender, data_gender = preloaded['gender']
        if insights_gender:
            with st.expander("Key gender insights"):
                for line in insights_gender:
                    st.write("•", line)
        for title, fig in figs_gender:
            st.markdown(f"### {title}")
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("### Sample zero-dose data")
        st.dataframe(data_gender.get('zerodose_df').head(200) if data_gender is not None else "No data")
    else:
        st.warning("Gender dashboard not preloaded. Attempting to render using render_gender_dashboard()...")
        try:
            render_gender_dashboard(facility_file=facility_csv_override, zero_dose_file=zd_path)
        except Exception as e:
            st.error(f"Failed to render gender dashboard: {e}")

# Tab 3: Age Analytics
with tabs[2]:
    st.header("Age-based Zero-Dose Analytics")
    if preloaded.get('age'):
        figs_age, zd_age_df, facility_age_df = preloaded['age']
        for fig in figs_age:
            st.plotly_chart(fig, use_container_width=True)
        with st.expander("Preview zero-dose data"):
            st.dataframe(zd_age_df.head(200))
    else:
        st.warning("Age dashboard was not preloaded. Attempting to call get_dashboard_age() directly...")
        try:
            figs_age, zd_age_df, facility_age_df = get_dashboard_age()
            for fig in figs_age:
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to load Age dashboard: {e}")

# Tab 4: Household / Settlements
with tabs[3]:
    st.header("Household / Settlement Level Analysis")
    if preloaded.get('household'):
        figs_house, table_house, summary_house = preloaded['household']
        st.info(f"Total settlements: {summary_house.get('total_settlements', 'N/A')}")
        for title, fig in figs_house:
            st.markdown(f"### {title}")
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("### Settlement details (top rows)")
        st.dataframe(table_house)
    else:
        st.warning("Household dashboard not preloaded. Attempting render...")
        try:
            render_household_dashboard(zerodose_path=zd_path, visit_path=facility_csv_override)
        except Exception as e:
            st.error(f"Failed to render household dashboard: {e}")

# Tab 5: Time-series & Follow-up
with tabs[4]:
    st.header("Time-Series & Follow-up")
    if preloaded.get('timeseries'):
        fig_enroll_ts, fig_long_ts, table_ts = preloaded['timeseries']
        st.plotly_chart(fig_enroll_ts, use_container_width=True)
        st.plotly_chart(fig_long_ts, use_container_width=True)
        st.subheader("Settlement summary (top rows)")
        st.dataframe(table_ts)
    else:
        st.warning("Timeseries dashboard not preloaded. Attempting to run show_timeseries_dashboard()...")
        try:
            show_timeseries_dashboard()
        except Exception as e:
            st.error(f"Failed to render timeseries dashboard: {e}")

# Tab 6: Additional Analytics
with tabs[5]:
    st.header("Additional Analytics")
    if preloaded.get('additional'):
        fig_heat, fig_net, fig_trend = preloaded['additional']
        st.subheader("Ward-wise Zero-Dose Density Heatmap")
        st.plotly_chart(fig_heat, use_container_width=True)
        st.subheader("Household Network Graph (simulated)")
        st.plotly_chart(fig_net, use_container_width=True)
        st.subheader("Drop-off Trend by Enrollment Month")
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.warning("Additional analytics not preloaded. Attempting to render show_additional_dashboard()...")
        try:
            show_additional_dashboard()
        except Exception as e:
            st.error(f"Failed to render additional analytics: {e}")

# Tab 7: Risk & Segmentation (ML)
with tabs[6]:
    st.header("Predictive Risk & Segmentation")
    if preloaded.get('risk'):
        fig_prob, fig_feat, fig_pca, table_risk = preloaded['risk']
        st.subheader("Predicted Dropoff Probability Distribution")
        st.plotly_chart(fig_prob, use_container_width=True)
        st.subheader("Top Feature Importances")
        st.plotly_chart(fig_feat, use_container_width=True)
        st.subheader("Clusters (PCA)")
        st.plotly_chart(fig_pca, use_container_width=True)
        st.subheader("Top children by dropoff probability")
        st.dataframe(table_risk)
    else:
        st.warning("Risk dashboard not preloaded. Attempting to compute on demand (heavy) — this may take time.")
        try:
            fig_prob, fig_feat, fig_pca, table_risk = get_dashboard_risk()
            st.plotly_chart(fig_prob, use_container_width=True)
            st.plotly_chart(fig_feat, use_container_width=True)
            st.plotly_chart(fig_pca, use_container_width=True)
            st.dataframe(table_risk)
        except Exception as e:
            st.error(f"Failed to compute risk dashboard: {e}")

# --- Footer / quick help ---
st.markdown("---")
st.caption("Tip: to force recompute (clear all caches), use Streamlit's 'Clear cache and rerun' from the app menu.")
