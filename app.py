"""
Unified Vaccine Analytics Dashboard (Streamlit)
- Preloads 7 dashboard modules (cached) so tab-switching is instant.
- Run: `streamlit run app.py`
"""

import streamlit as st
from textwrap import dedent
import ai_helper
import data_utils
import pandas as pd
import os
import datetime

st.set_page_config(page_title="AI-Powered Insights from MCHTrack Zero-dose Immunization Data",
                   layout="wide",
                   initial_sidebar_state="expanded")

# --- Hide Streamlit header/menu/footer ---
hide_streamlit_style = """
    <style>
      #MainMenu {visibility: hidden;}
      header {visibility: hidden;}
      footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# --- Imports of your dashboard modules ---
from dashboards.dashboard_vaccine_field import get_vaccine_dashboard, render_vaccine_dashboard
from dashboards.dashboard_gender import get_gender_dashboard, render_gender_dashboard
from dashboards.dashboard_3_age import get_dashboard_age, render_age_dashboard
from dashboards.dashboard_4_household import get_household_dashboard, render_household_dashboard
from dashboards.dashboard_5_timeseries import get_timeseries_dashboard, render_timeseries_dashboard
# from dashboards.dashboard_6_additional import get_additional_dashboard, render_additional_dashboard
# from dashboards.dashboard_7_risk import get_dashboard_risk, render_dashboard_risk

# --- Sidebar (logo, dataset paths, quick controls) ---
st.sidebar.title("Datharm")
if os.path.exists("client_logo.png"):
    st.sidebar.image("client_logo.png", width=160)

# ==========================================
#         GLOBAL FILTERS
# ==========================================
st.sidebar.header("Global Filters")

# Corrected paths based on your csv info (singular 'visit')
zd_path = r"C:\Users\AkshayKarthickMS\Desktop\Phase-1-main\data\zerodose.xlsx"
vis_path = r"C:\Users\AkshayKarthickMS\Desktop\Phase-1-main\data\facility_visits.csv"
facility_csv_override = vis_path
preload_on_start = True

# 1. LGA Filter - Using Robust Data Utils
@st.cache_data
def load_lgas_cached(path):
    return data_utils.get_unique_lgas(path)

all_lgas = load_lgas_cached(zd_path) 

# --- USE A FORM TO PREVENT CONSTANT RELOADING ---
with st.sidebar.form("global_filters_form"):
    st.write("Refine Analysis Scope")
    
    selected_lgas = st.multiselect("Select LGA(s)", options=all_lgas, default=all_lgas if all_lgas else None)

    # 2. Date Range Filter
    today = datetime.date.today()
    default_start = today - datetime.timedelta(days=365)
    start_date = st.date_input("Start Date", value=default_start)
    end_date = st.date_input("End Date", value=today)

    # 3. Gender Filter
    gender_options = ["Male", "Female"]
    selected_genders = st.multiselect("Select Gender", options=gender_options, default=gender_options)
    
    # Submit Button
    submitted = st.form_submit_button("Apply Filters")

if not all_lgas:
    st.sidebar.warning("No LGAs loaded or file not found.")

st.sidebar.markdown("---")

# --- Gemini API Key Input ---
st.sidebar.header("AI Analysis Settings")
gemini_api_key = st.sidebar.text_input("Enter Gemini API Key", type="password", help="Required for the 'Analyse' button on charts.")
st.sidebar.markdown("---")


# --- Main header ---
st.title("AI-Powered Insights from MCHTrack Zero-dose Immunization Data")
st.write("Switch tabs for various categories of visualizations")

# --- Preload all dashboards (cached functions) ---
preload_errors = {}
preloaded = {}

# Pass all filters to the dashboards that support them
filter_kwargs = {
    "selected_lgas": selected_lgas,
    "start_date": start_date,
    "end_date": end_date,
    "selected_genders": selected_genders
}

if preload_on_start:
    with st.spinner("Processing dashboards with applied filters... ⏳"):
        
        # 1. Vaccine field
        try:
            figs_vaccine, summary_vaccine = get_vaccine_dashboard(
                zerodose_path=zd_path, visit_path=vis_path, **filter_kwargs
            )
            preloaded['vaccine'] = (figs_vaccine, summary_vaccine)
        except Exception as e:
            preload_errors['vaccine'] = str(e)
            preloaded['vaccine'] = None

        # 2. Gender
        try:
            figs_gender, summary_gender, insights_gender, data_gender = get_gender_dashboard(
                facility_path=vis_path, zerodose_path=zd_path, **filter_kwargs
            )
            preloaded['gender'] = (figs_gender, summary_gender, insights_gender, data_gender)
        except Exception as e:
            preload_errors['gender'] = str(e)
            preloaded['gender'] = None

        # 3. Age
        try:
            figs_age, zd_age_df, facility_age_df = get_dashboard_age(
                facility_path=vis_path, zerodose_path=zd_path, **filter_kwargs
            )
            preloaded['age'] = (figs_age, zd_age_df, facility_age_df)
        except Exception as e:
            preload_errors['age'] = str(e)
            preloaded['age'] = None

        # 4. Household
        try:
            figs_house, table_house, summary_house = get_household_dashboard(
                zerodose_path=zd_path, visit_path=vis_path, **filter_kwargs
            )
            preloaded['household'] = (figs_house, table_house, summary_house)
        except Exception as e:
            preload_errors['household'] = str(e)
            preloaded['household'] = None

        # 5. Time-series
        try:
            figs_ts, table_ts, summary_ts = get_timeseries_dashboard(
                zerodose_path=zd_path, visit_path=vis_path, **filter_kwargs
            )
            preloaded['timeseries'] = (figs_ts, table_ts, summary_ts)
        except Exception as e:
            preload_errors['timeseries'] = str(e)
            preloaded['timeseries'] = None

        # # 6. Additional
        # try:
        #     figs_add, summary_add = get_additional_dashboard(
        #         zerodose_path=zd_path, visit_path=vis_path, **filter_kwargs
        #     )
        #     preloaded['additional'] = (figs_add, summary_add)
        # except Exception as e:
        #     preload_errors['additional'] = str(e)
        #     preloaded['additional'] = None

        # 7. Risk
        # try:
        #     risk_data = get_dashboard_risk(
        #         zerodose_path=zd_path, visit_path=vis_path, **filter_kwargs
        #     )
        #     preloaded['risk'] = risk_data
        # except Exception as e:
        #     preload_errors['risk'] = str(e)
        #     preloaded['risk'] = None

# --- Callback for Professional Layouts ---
def ai_chart_renderer(fig, title, key_suffix):
    st.markdown(f"**{title}**")
    st.plotly_chart(fig, use_container_width=True)
    
    button_key = f"analyse_btn_{key_suffix}"
    if st.button(f"✨ Analyse with AI", key=button_key):
        analysis = ai_helper.get_gemini_analysis(fig, title, gemini_api_key)
        if analysis:
            st.info(analysis)

# --- Standard Helper for Legacy Layouts ---
def render_with_ai_analysis(fig, title, key_suffix):
    st.markdown(f"### {title}")
    st.plotly_chart(fig, use_container_width=True)
    
    button_key = f"analyse_btn_{key_suffix}"
    if st.button(f"✨ Analyse with AI", key=button_key):
        analysis = ai_helper.get_gemini_analysis(fig, title, gemini_api_key)
        if analysis:
            st.info(analysis)

# --- Tabs layout ---
tab_labels = [
    "1 • Vaccine Field",
    "2 • Gender Analysis",
    "3 • Age Analytics",
    "4 • Household / Settlements",
    "5 • Time-series & Follow-up",
    # "6 • Risk & Segmentation",
    # "7 • Additional Analytics"
]
tabs = st.tabs(tab_labels)

# Tab 1: Vaccine Field
with tabs[0]:
    st.header("Vaccine Field Dashboard")
    if preloaded.get('vaccine'):
        render_vaccine_dashboard(
            precomputed_data=preloaded['vaccine'], chart_callback=ai_chart_renderer
        )
    else:
        st.warning("Filters updated. Reloading Vaccine dashboard...")
        render_vaccine_dashboard(
            zerodose_path=zd_path, visit_path=vis_path, **filter_kwargs, chart_callback=ai_chart_renderer
        )

# Tab 2: Gender Analysis
with tabs[1]:
    st.header("Zero-Dose Gender Analytics")
    if preloaded.get('gender'):
        render_gender_dashboard(
            precomputed_data=preloaded['gender'], chart_callback=ai_chart_renderer
        )
    else:
        render_gender_dashboard(
            facility_path=vis_path, zerodose_path=zd_path, **filter_kwargs, chart_callback=ai_chart_renderer
        )

# Tab 3: Age Analytics
with tabs[2]:
    st.header("Age-based Zero-Dose Analytics")
    if preloaded.get('age'):
        render_age_dashboard(
            precomputed_data=preloaded['age'], chart_callback=ai_chart_renderer
        )
    else:
        render_age_dashboard(
            facility_path=vis_path, zerodose_path=zd_path, **filter_kwargs, chart_callback=ai_chart_renderer
        )

# Tab 4: Household
with tabs[3]:
    st.header("Household / Settlement Level Analysis")
    if preloaded.get('household'):
        render_household_dashboard(
            precomputed_data=preloaded['household'], chart_callback=ai_chart_renderer
        )
    else:
        render_household_dashboard(
            zerodose_path=zd_path, visit_path=vis_path, **filter_kwargs, chart_callback=ai_chart_renderer
        )

# Tab 5: Time-series
with tabs[4]:
    st.header("Time-Series & Follow-up")
    if preloaded.get('timeseries'):
        render_timeseries_dashboard(
            precomputed_data=preloaded['timeseries'], chart_callback=ai_chart_renderer
        )
    else:
        render_timeseries_dashboard(
            zerodose_path=zd_path, visit_path=vis_path, **filter_kwargs, chart_callback=ai_chart_renderer
        )

# # Tab 6: Additional
# with tabs[6]:
#     st.header("Additional Analytics")
#     if preloaded.get('additional'):
#         render_additional_dashboard(
#             precomputed_data=preloaded['additional'], chart_callback=ai_chart_renderer
#         )
#     else:
#         render_additional_dashboard(
#             zerodose_path=zd_path, visit_path=vis_path, **filter_kwargs, chart_callback=ai_chart_renderer
#         )

# Tab 7: Risk
# with tabs[5]:
#     st.header("Predictive Risk & Segmentation")
#     if preloaded.get('risk'):
#         render_dashboard_risk(
#             precomputed_data=preloaded['risk'], chart_callback=ai_chart_renderer
#         )
#     else:
#         render_dashboard_risk(
#             zerodose_path=zd_path, visit_path=vis_path, **filter_kwargs, chart_callback=ai_chart_renderer
#         )
        
st.markdown("---")