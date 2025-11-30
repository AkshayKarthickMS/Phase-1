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
import ai_helper


st.set_page_config(page_title="AI-Powered Insights from MCHTrack Zero-dose Immunization Data",
                   layout="wide",
                   initial_sidebar_state="expanded")

# --- Hide Streamlit UI elements completely ---
hide_streamlit_style = """
    <style>
        /* Hide hamburger menu */
        #MainMenu {visibility: hidden;}

        /* Hide header (includes profile icon in new Streamlit versions) */
        header {visibility: hidden;}

        /* Hide footer */
        footer {visibility: hidden;}

        /* Hide "Hosted with Streamlit" badge and profile icon (Cloud-specific classes) */
        .viewerBadge_container__1QSob {display: none !important;}
        .viewerBadge_link__1S137 {display: none !important;}
        .st-emotion-cache-13ln4jf {display: none !important;} /* Streamlit Cloud profile icon */
        .st-emotion-cache-17ziqus {display: none !important;} /* Extra profile container */
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


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
st.sidebar.title("Datharm")
st.sidebar.markdown("Switch tabs for instant views.")

# --- Gemini API Key Input ---
# Add a text input in the sidebar for the user to enter their Gemini API key.
st.sidebar.header("AI Analysis Settings")
gemini_api_key = st.sidebar.text_input("Enter Gemini API Key", type="password", help="Required for the 'Analyse' button on charts.")
st.sidebar.markdown("---")

# Optional: let user change csv filenames if needed
# st.sidebar.header("Data files")
# zd_path = st.sidebar.text_input("Zero-dose CSV", value="zerodose.csv")
# vis_path = st.sidebar.text_input("Facility visit CSV", value="facility_visit.csv")
# facility_csv_override = vis_path  # keep names consistent where called
# # Preload toggle (but we will preload by default)
# preload_on_start = st.sidebar.checkbox("Preload all dashboards on start", value=True)

zd_path = "zerodose.csv"
vis_path = "facility_visit.csv"
facility_csv_override = vis_path
preload_on_start = True

# Small instruction
# st.sidebar.markdown(dedent("""
# **Notes**
# - Large CSVs and models run once (cached) and are reused.
# - To clear cached computations: _Streamlit menu → Clear cache and rerun_.
# """))

# --- Main header ---
st.title("AI-Powered Insights from MCHTrack Zero-dose Immunization Data")
st.write("Switch tabs for various categories of visualizations")

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

# --- Helper function to render chart with AI analysis button ---
def render_with_ai_analysis(fig, title, key_suffix):
    """
    Displays a Plotly chart and an 'Analyse' button.
    When the button is clicked, it calls the AI helper to generate and display an analysis.

    Args:
        fig: The Plotly figure to display.
        title: The title of the chart.
        key_suffix: A unique suffix for the button's key to prevent conflicts.
    """
    # Display the chart's title
    st.markdown(f"### {title}")
    # Display the Plotly chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Create a unique key for the button based on the suffix
    button_key = f"analyse_btn_{key_suffix}"
    
    # Create the "Analyse" button
    if st.button(f"✨ Analyse with AI", key=button_key):
        # Call the helper function to get the analysis
        analysis = ai_helper.get_gemini_analysis(fig, title, gemini_api_key)
        # If an analysis is returned, display it in an info box
        if analysis:
            st.info(analysis)

# --- Tabs layout ---
tab_labels = [
    "1 • Vaccine Field",
    "2 • Gender Analysis",
    "3 • Age Analytics",
    "4 • Household / Settlements",
    "5 • Time-series & Follow-up",
    "6 • Risk & Segmentation",
    "7 • Additional Analytics"
]
tabs = st.tabs(tab_labels)

# Tab 1: Vaccine Field
with tabs[0]:
    st.header("Vaccine Field Dashboard")
    if preloaded.get('vaccine'):
        figs_vaccine, summary_vaccine = preloaded['vaccine']
        st.info(f"Records: {summary_vaccine.get('total_records', 'N/A'):,} | Unique LGAs: {summary_vaccine.get('unique_lgas','N/A'):,} | Vaccines tracked: {summary_vaccine.get('vaccine_count','N/A')}")
        # Iterate through the figures and render them with the AI analysis button
        for i, (title, fig) in enumerate(figs_vaccine):
            render_with_ai_analysis(fig, title, f"vaccine_{i}")
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
        # Iterate through the figures and render them with the AI analysis button
        for i, (title, fig) in enumerate(figs_gender):
            render_with_ai_analysis(fig, title, f"gender_{i}")
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
        for i, fig in enumerate(figs_age):
            # Extract title from layout if available, otherwise generate a generic one
            title = fig.layout.title.text if fig.layout.title.text else f"Age Analysis Chart {i+1}"
            # Render the figure with the AI analysis button
            render_with_ai_analysis(fig, title, f"age_{i}")
        with st.expander("Preview zero-dose data"):
            st.dataframe(zd_age_df.head(200))
    else:
        st.warning("Age dashboard was not preloaded. Attempting to call get_dashboard_age() directly...")
        try:
            figs_age, zd_age_df, facility_age_df = get_dashboard_age()
            for i, fig in enumerate(figs_age):
                title = fig.layout.title.text if fig.layout.title.text else f"Age Analysis Chart {i+1}"
                render_with_ai_analysis(fig, title, f"age_{i}_direct")
        except Exception as e:
            st.error(f"Failed to load Age dashboard: {e}")

# Tab 4: Household / Settlements
with tabs[3]:
    st.header("Household / Settlement Level Analysis")
    if preloaded.get('household'):
        figs_house, table_house, summary_house = preloaded['household']
        st.info(f"Total settlements: {summary_house.get('total_settlements', 'N/A')}")
        # Iterate through the figures and render them with the AI analysis button
        for i, (title, fig) in enumerate(figs_house):
            render_with_ai_analysis(fig, title, f"household_{i}")
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
        # Render each figure with the AI analysis button
        render_with_ai_analysis(fig_enroll_ts, "Enrollment Time-Series", "timeseries_enroll")
        render_with_ai_analysis(fig_long_ts, "Longitudinal Follow-up", "timeseries_long")
        st.subheader("Settlement summary (top rows)")
        st.dataframe(table_ts)
    else:
        st.warning("Timeseries dashboard not preloaded. Attempting to run show_timeseries_dashboard()...")
        try:
            show_timeseries_dashboard()
        except Exception as e:
            st.error(f"Failed to render timeseries dashboard: {e}")

# Tab 6: Additional Analytics
with tabs[6]:
    st.header("Additional Analytics")
    if preloaded.get('additional'):
        fig_heat, fig_net, fig_trend = preloaded['additional']
        # Render each figure with the AI analysis button
        render_with_ai_analysis(fig_heat, "Ward-wise Zero-Dose Density Heatmap", "additional_heat")
        render_with_ai_analysis(fig_net, "Household Network Graph (simulated)", "additional_net")
        render_with_ai_analysis(fig_trend, "Drop-off Trend by Enrollment Month", "additional_trend")
    else:
        st.warning("Additional analytics not preloaded. Attempting to render show_additional_dashboard()...")
        try:
            show_additional_dashboard()
        except Exception as e:
            st.error(f"Failed to render additional analytics: {e}")

# Tab 7: Risk & Segmentation (ML)
with tabs[5]:
    st.header("Predictive Risk & Segmentation")
    if preloaded.get('risk'):
        fig_prob, fig_feat, fig_pca, table_risk = preloaded['risk']
        # Render each figure with the AI analysis button
        render_with_ai_analysis(fig_prob, "Predicted Dropoff Probability Distribution", "risk_prob")
        render_with_ai_analysis(fig_feat, "Top Feature Importances", "risk_feat")
        render_with_ai_analysis(fig_pca, "Clusters (PCA)", "risk_pca")
        st.subheader("Top children by dropoff probability")
        st.dataframe(table_risk)
    else:
        st.warning("Risk dashboard not preloaded. Attempting to compute on demand (heavy) — this may take time.")
        try:
            fig_prob, fig_feat, fig_pca, table_risk = get_dashboard_risk()
            render_with_ai_analysis(fig_prob, "Predicted Dropoff Probability Distribution", "risk_prob_direct")
            render_with_ai_analysis(fig_feat, "Top Feature Importances", "risk_feat_direct")
            render_with_ai_analysis(fig_pca, "Clusters (PCA)", "risk_pca_direct")
            st.dataframe(table_risk)
        except Exception as e:
            st.error(f"Failed to compute risk dashboard: {e}")
            
# --- Footer / quick help ---
st.markdown("---")















