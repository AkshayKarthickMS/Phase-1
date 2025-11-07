import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

@st.cache_resource
def get_dashboard_age():
    # -------------------------------
    # 1. Load and preprocess datasets
    # -------------------------------
    facility_df = pd.read_csv("data/facility_visit.csv")
    zerodose_df = pd.read_csv("data/zerodose.csv")

    # Parse vaccines_administered safely
    def parse_vaccines(vaccine_str):
        try:
            if pd.isna(vaccine_str):
                return []
            vaccine_str = str(vaccine_str).strip('{}').replace('"', '').replace("'", "")
            vaccines = [v.strip() for v in vaccine_str.split(',') if v.strip()]
            return vaccines
        except:
            return []

    facility_df['vaccines_list'] = facility_df['vaccines_administered'].apply(parse_vaccines)

    # Convert to months
    facility_df['age_in_months'] = (
        facility_df['age_years'].fillna(0) * 12 +
        facility_df['age_months'].fillna(0) +
        facility_df['age_weeks'].fillna(0) / 4.345
    )

    # Parse age strings
    def parse_age_to_months(age_str):
        if pd.isna(age_str):
            return None
        age_str = str(age_str).lower().strip()
        patterns = [
            r"(\d+)\s*year[s]?[,\s]*(\d+)?\s*month[s]?[,\s]*(\d+)?\s*week[s]?",
            r"(\d+)y\s*(\d+)?m\s*(\d+)?w",
            r"(\d+)\s*yr[s]?[,\s]*(\d+)?\s*mo[s]?[,\s]*(\d+)?\s*wk[s]?"
        ]
        for pattern in patterns:
            match = re.findall(pattern, age_str, re.IGNORECASE)
            if match:
                years, months, weeks = match[0]
                years = int(years) if years else 0
                months = int(months) if months else 0
                weeks = int(weeks) if weeks else 0
                return years*12 + months + weeks/4.345
        return None

    zerodose_df['enrollment_age_months'] = zerodose_df['Age at Enrollment'].apply(parse_age_to_months)
    zerodose_df['current_age_months'] = zerodose_df['Estimated Current Age'].apply(parse_age_to_months)

    zerodose_df['months_since_enrollment'] = (
        zerodose_df['current_age_months'] - zerodose_df['enrollment_age_months']
    )

    # -------------------------------
    # 2. Age bins & risk categorization
    # -------------------------------
    age_bins = [0, 1.5, 12, 24, 60, np.inf]
    age_labels = ["Too Young", "Due for Penta", "Active ZD", "Overaged ZD", "Out of scope"]

    facility_df['immunization_age_group'] = pd.cut(
        facility_df['age_in_months'], bins=age_bins, labels=age_labels, right=False, include_lowest=True
    )

    zerodose_df['immunization_age_group'] = pd.cut(
        zerodose_df['current_age_months'], bins=age_bins, labels=age_labels, right=False, include_lowest=True
    )

    def categorize_risk(age_months):
        if pd.isna(age_months):
            return "Unknown"
        if 1.5 <= age_months < 12:
            return "Due for Penta"
        elif 12 <= age_months < 24:
            return "Active ZD"
        elif 24 <= age_months < 60:
            return "Overaged ZD"
        elif age_months < 1.5:
            return "Too Young"
        else:
            return "Out of scope"

    zerodose_df['risk_category'] = zerodose_df['current_age_months'].apply(categorize_risk)

    # -------------------------------
    # 3. Vaccine schedule & coverage
    # -------------------------------
    VACCINE_SCHEDULE = {
        "BCG": {"age": 0, "dose": "At birth"},
        "HepB_0": {"age": 0, "dose": "At birth"},
        "OPV_0": {"age": 0, "dose": "At birth"},
        "Penta_1": {"age": 2, "dose": "1st dose"},
        "PCV_1": {"age": 2, "dose": "1st dose"},
        "OPV_1": {"age": 2, "dose": "1st dose"},
        "Rota_1": {"age": 2, "dose": "1st dose"},
        "IPV_1": {"age": 2, "dose": "1st dose"},
        "Penta_2": {"age": 4, "dose": "2nd dose"},
        "PCV_2": {"age": 4, "dose": "2nd dose"},
        "OPV_2": {"age": 4, "dose": "2nd dose"},
        "Rota_2": {"age": 4, "dose": "2nd dose"},
        "Penta_3": {"age": 6, "dose": "3rd dose"},
        "PCV_3": {"age": 6, "dose": "3rd dose"},
        "OPV_3": {"age": 6, "dose": "3rd dose"},
        "Rota_3": {"age": 6, "dose": "3rd dose"},
        "IPV_2": {"age": 6, "dose": "2nd dose"},
        "VitaminA_1": {"age": 6, "dose": "1st dose"},
        "Measles_1": {"age": 9, "dose": "1st dose"},
        "YF": {"age": 9, "dose": "Single dose"},
        "meningitis": {"age": 9, "dose": "Single dose"},
        "VitaminA_2": {"age": 12, "dose": "2nd dose"},
        "Measles_2": {"age": 15, "dose": "2nd dose"},
        "HPV": {"age": 60, "dose": "For girls 9-14 years"}
    }

    def identify_missed_vaccines(age_months):
        if pd.isna(age_months):
            return "Unknown"
        if age_months < 2:
            return "BCG, OPV0, HepB0 (At birth)"
        elif age_months < 4:
            return "Penta1, OPV1, PCV1, Rota1, IPV1"
        elif age_months < 6:
            return "Penta2, OPV2, PCV2, Rota2"
        elif age_months < 9:
            return "Penta3, OPV3, PCV3, Rota3, IPV2, VitaminA1"
        elif age_months < 12:
            return "Measles1, YF, meningitis"
        elif age_months < 15:
            return "VitaminA2"
        elif age_months < 60:
            return "Measles2 + Catch-up needed"
        else:
            return "Multiple vaccines - Full catch-up + HPV"

    zerodose_df['missed_vaccines'] = zerodose_df['current_age_months'].apply(identify_missed_vaccines)

    def analyze_vaccine_coverage_by_age(facility_df):
        age_labels = ["Too Young", "Due for Penta", "Active ZD", "Overaged ZD", "Out of scope"]
        coverage_data = []
        for age_group in age_labels:
            age_data = facility_df[facility_df['immunization_age_group'] == age_group]
            total_children = len(age_data)
            if total_children == 0:
                continue
            vaccine_coverage = {}
            for vaccine in VACCINE_SCHEDULE.keys():
                children_vaccinated = len(age_data[age_data['vaccines_list'].apply(lambda x: vaccine in x)])
                coverage_pct = (children_vaccinated / total_children) * 100 if total_children > 0 else 0
                vaccine_coverage[vaccine] = coverage_pct
            coverage_data.append({
                'age_group': age_group,
                'total_children': total_children,
                **vaccine_coverage
            })
        return pd.DataFrame(coverage_data)

    vaccine_coverage_df = analyze_vaccine_coverage_by_age(facility_df)
    zerodose_df['Distance_km'] = (
        zerodose_df['Distance to HF'].astype(str).str.replace("Km", "", case=False).astype(float)
    )

    # -------------------------------
    # 4. Visualizations
    # -------------------------------
    figures = []

    fig1 = px.scatter(
        zerodose_df, x="current_age_months", y="Distance_km",
        color="risk_category", hover_data=["Settlement", "LGA", "Reason for ZD", "missed_vaccines"],
        title="<b>Risk Priority Matrix: Age vs Distance to Health Facility</b>",
        labels={"current_age_months": "Age (Months)", "Distance_km": "Distance (Km)"}
    )
    figures.append(fig1)

    vaccine_gap_analysis = zerodose_df.groupby(['immunization_age_group', 'missed_vaccines']).size().reset_index(name='count')
    fig2 = px.bar(
        vaccine_gap_analysis, x='immunization_age_group', y='count', color='missed_vaccines',
        title="<b>Vaccination Gap Analysis by Age Group</b>"
    )
    figures.append(fig2)

    if not vaccine_coverage_df.empty:
        melted_coverage = vaccine_coverage_df.melt(
            id_vars=['age_group', 'total_children'],
            var_name='vaccine', value_name='coverage_pct'
        )
        fig3 = px.imshow(
            melted_coverage.pivot(index='vaccine', columns='age_group', values='coverage_pct'),
            zmin=0, zmax=100, color_continuous_scale='RdYlGn',
            title="<b>Vaccine Coverage by Age Group (%)</b>"
        )
        figures.append(fig3)

    fig4 = px.histogram(
        zerodose_df, x='months_since_enrollment', color='Status',
        nbins=20, barmode='stack',
        title="<b>Time Sensitivity: Months Since Enrollment</b>"
    )
    figures.append(fig4)

    fig5 = make_subplots(rows=1, cols=2, subplot_titles=("Facility Visits", "Zero-Dose Children"))
    fig5.add_trace(go.Histogram(x=facility_df['age_in_months'], nbinsx=30, name='Facility'), row=1, col=1)
    fig5.add_trace(go.Histogram(x=zerodose_df['current_age_months'], nbinsx=30, name='Zero-Dose'), row=1, col=2)
    fig5.update_layout(title_text="<b>Age Distribution Comparison</b>")
    figures.append(fig5)

    priority_df = zerodose_df.groupby(['LGA', 'risk_category']).size().reset_index(name='count')
    fig6 = px.sunburst(priority_df, path=['LGA', 'risk_category'], values='count',
                       title="<b>Intervention Priority Dashboard</b>")
    figures.append(fig6)

    age_risk_profile = zerodose_df.groupby('immunization_age_group').agg({
        'current_age_months': 'count','Distance_km': 'mean'}).reset_index()
    age_risk_profile['risk_score'] = (
        age_risk_profile['current_age_months'] / age_risk_profile['current_age_months'].max() * 0.7 +
        age_risk_profile['Distance_km'] / age_risk_profile['Distance_km'].max() * 0.3
    )
    fig7 = px.bar(age_risk_profile, x='immunization_age_group', y='risk_score',
                  title="<b>Predictive Risk Score by Age Group</b>")
    figures.append(fig7)

    reason_age = pd.crosstab(zerodose_df['Reason for ZD'], zerodose_df['immunization_age_group'])
    fig8 = px.imshow(reason_age, text_auto=True, aspect="auto",
                     title="<b>Reason for Zero-Dose by Age Group</b>")
    figures.append(fig8)

    return figures, zerodose_df, facility_df
