# dashboard_gender.py
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import re
import os

# ---------------------------
# Safe path resolution setup
# ---------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

def resolve_path(filename: str) -> str:
    """Return absolute path for data files, checking cwd and /data folder."""
    if not filename:
        return None
    # Absolute path already
    if os.path.isabs(filename):
        return filename
    # Try current working directory (when running from root)
    if os.path.exists(filename):
        return filename
    # Fallback to /data/ under project root
    data_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(data_path):
        return data_path
    # Nothing found
    return filename  # return as-is; caller will raise FileNotFoundError

# ---------------------------
# Helper: parse age strings -> months
# ---------------------------

def parse_age_to_months(age_str):
    if pd.isna(age_str):
        return np.nan
    s = str(age_str)
    patterns = [
        r"(\d+)\s*year[s]?[,\s]*(\d+)?\s*month[s]?[,\s]*(\d+)?\s*week[s]?",
        r"(\d+)y\s*(\d+)?m\s*(\d+)?w",
        r"(\d+)\s*yr[s]?[,\s]*(\d+)?\s*mo[s]?[,\s]*(\d+)?\s*wk[s]?"
    ]
    for pattern in patterns:
        m = re.findall(pattern, s, re.IGNORECASE)
        if m:
            years, months, weeks = m[0]
            years = int(years) if years and years.isdigit() else 0
            months = int(months) if months and months.isdigit() else 0
            weeks = int(weeks) if weeks and weeks.isdigit() else 0
            return years * 12 + months + weeks / 4.345
    # fallback: extract any number followed by "month" or "m"
    m = re.search(r"(\d+)\s*month", s, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return np.nan

# ---------------------------
# Heavy precomputation (cached)
# ---------------------------

@st.cache_resource
def get_gender_dashboard(facility_path=None, zerodose_path=None):
    """
    Load CSVs, run preprocessing and analytics. Returns:
      - figs: list of (title, plotly_figure)
      - summary: dict of basic metrics
      - insights: list of text insights
      - data: dict of important DataFrames
    """
    facility_path = resolve_path(facility_path or "facility_visit.csv")
    zerodose_path = resolve_path(zerodose_path or "zerodose.csv")

    missing = [p for p in [facility_path, zerodose_path] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing input CSV(s): {', '.join(missing)}")

    # Load CSVs
    facility_df = pd.read_csv(facility_path)
    zerodose_df = pd.read_csv(zerodose_path)

    # Standardize gender values
    facility_df['gender'] = facility_df.get('gender', '').astype(str).str.lower().str.strip()
    facility_df['gender'] = facility_df['gender'].replace({'': 'unknown', 'nan': 'unknown'}).fillna('unknown')

    # Normalize gender column in zero-dose data
    if 'Gender' in zerodose_df.columns:
        zerodose_df['Gender'] = zerodose_df['Gender'].astype(str).str.lower().str.strip()
    elif 'gender' in zerodose_df.columns:
        zerodose_df['Gender'] = zerodose_df['gender'].astype(str).str.lower().str.strip()
    else:
        zerodose_df['Gender'] = 'unknown'
    zerodose_df['Gender'] = zerodose_df['Gender'].replace({'': 'unknown', 'nan': 'unknown'}).fillna('unknown')

    # Facility: compute age in months
    if {'age_years', 'age_months', 'age_weeks'}.issubset(facility_df.columns):
        facility_df['age_in_months'] = (
            pd.to_numeric(facility_df['age_years'], errors='coerce').fillna(0) * 12
            + pd.to_numeric(facility_df['age_months'], errors='coerce').fillna(0)
            + pd.to_numeric(facility_df['age_weeks'], errors='coerce').fillna(0) / 4.345
        )
    else:
        facility_df['age_in_months'] = np.nan

    # Zero-dose: parse estimated current age -> months
    if 'Estimated Current Age' in zerodose_df.columns:
        zerodose_df['current_age_months'] = zerodose_df['Estimated Current Age'].apply(parse_age_to_months)
    else:
        zerodose_df['current_age_months'] = np.nan

    # Age groups
    age_bins = [0, 6, 12, 24, 36, 60]
    age_labels = ["0-6m", "6-12m", "1-2y", "2-3y", "3-5y"]
    zerodose_df['age_group'] = pd.cut(zerodose_df['current_age_months'], bins=age_bins, labels=age_labels, right=False)

    # Distance normalization
    if 'Distance to HF' in zerodose_df.columns:
        zerodose_df['Distance_km'] = zerodose_df['Distance to HF'].astype(str).str.replace("Km", "", case=False).str.strip()
        zerodose_df['Distance_km'] = pd.to_numeric(zerodose_df['Distance_km'], errors='coerce')
    else:
        zerodose_df['Distance_km'] = np.nan

    # Safe defaults for missing columns
    for col in ['Settlement', 'LGA', 'Reason for ZD', 'Status']:
        if col not in zerodose_df.columns:
            zerodose_df[col] = np.nan

    # ---------------------------
    # Create plots
    # ---------------------------
    figs = []

    # 1️⃣ Facility vs Zero-dose gender distribution
    try:
        facility_gender_counts = facility_df['gender'].value_counts().reset_index()
        facility_gender_counts.columns = ['gender', 'count']
        facility_gender_counts['source'] = 'Facility'

        zerodose_gender_counts = zerodose_df['Gender'].value_counts().reset_index()
        zerodose_gender_counts.columns = ['gender', 'count']
        zerodose_gender_counts['source'] = 'Zero-Dose'

        combined_gender = pd.concat([facility_gender_counts, zerodose_gender_counts], ignore_index=True)
        fig1 = px.bar(
            combined_gender, x='gender', y='count', color='source', barmode='group',
            title="Gender Distribution: Facility Visits vs Zero-Dose Children"
        )
    except Exception:
        fig1 = go.Figure(); fig1.update_layout(title="Gender Distribution (error rendering)")
    figs.append(("Gender: Facility vs Zero-Dose", fig1))

    # 2️⃣ Age-group vs Gender
    try:
        gender_age = pd.crosstab(zerodose_df['age_group'], zerodose_df['Gender'])
        fig2 = px.bar(gender_age, barmode='group', title="Zero-Dose Children by Age Group and Gender")
    except Exception:
        fig2 = go.Figure(); fig2.update_layout(title="Zero-Dose by Age Group (error)")
    figs.append(("Zero-Dose by Age Group & Gender", fig2))

    # 3️⃣ Reason × Gender heatmap
    try:
        gender_reason = pd.crosstab(zerodose_df['Reason for ZD'], zerodose_df['Gender'])
        fig3 = px.imshow(
            gender_reason.fillna(0).astype(int),
            text_auto=True, aspect="auto",
            title="Reason for Zero-Dose by Gender"
        )
    except Exception:
        fig3 = go.Figure(); fig3.update_layout(title="Reason × Gender (error)")
    figs.append(("Reason for Zero-Dose by Gender", fig3))

    # 4️⃣ Settlement distribution by gender
    try:
        if 'Settlement' in zerodose_df.columns:
            gender_settlement = pd.crosstab(zerodose_df['Settlement'], zerodose_df['Gender'])
            fig4 = px.bar(
                gender_settlement, barmode='group',
                title="Zero-Dose by Settlement and Gender"
            )
            figs.append(("Zero-Dose by Settlement & Gender", fig4))
    except Exception:
        fig4 = go.Figure(); fig4.update_layout(title="Settlement × Gender (error)")
        figs.append(("Zero-Dose by Settlement & Gender", fig4))

    # 5️⃣ Average distance to HF by gender
    try:
        dist_gender = zerodose_df.groupby('Gender')['Distance_km'].mean().reset_index()
        fig5 = px.bar(
            dist_gender, x='Gender', y='Distance_km', color='Gender',
            title="Average Distance to Health Facility by Gender"
        )
        figs.append(("Avg Distance to HF by Gender", fig5))
    except Exception:
        fig5 = go.Figure(); fig5.update_layout(title="Avg Distance × Gender (error)")
        figs.append(("Avg Distance to HF by Gender", fig5))

    # 6️⃣ Status by gender
    try:
        if 'Status' in zerodose_df.columns:
            gender_status = pd.crosstab(zerodose_df['Status'], zerodose_df['Gender'])
            fig6 = px.bar(
                gender_status, barmode='group',
                title="Zero-Dose Status by Gender"
            )
            figs.append(("Zero-Dose Status by Gender", fig6))
    except Exception:
        fig6 = go.Figure(); fig6.update_layout(title="Status × Gender (error)")
        figs.append(("Zero-Dose Status by Gender", fig6))

    # 7️⃣ LGA-wise gender counts
    try:
        if 'LGA' in zerodose_df.columns:
            gender_lga = pd.crosstab(zerodose_df['LGA'], zerodose_df['Gender'])
            fig7 = px.bar(
                gender_lga, barmode='group',
                title="Zero-Dose Distribution by LGA and Gender"
            )
            figs.append(("Zero-Dose by LGA & Gender", fig7))
    except Exception:
        fig7 = go.Figure(); fig7.update_layout(title="LGA × Gender (error)")
        figs.append(("Zero-Dose by LGA & Gender", fig7))

    # 8️⃣ Pie chart: overall gender composition
    try:
        pie_counts = zerodose_df['Gender'].value_counts().reset_index()
        pie_counts.columns = ['Gender', 'Count']
        fig8 = px.pie(
            pie_counts, values='Count', names='Gender',
            title="Zero-Dose Gender Composition (%)", hole=0.3
        )
        figs.append(("Zero-Dose Gender Composition", fig8))
    except Exception:
        fig8 = go.Figure(); fig8.update_layout(title="Gender Composition (error)")
        figs.append(("Zero-Dose Gender Composition", fig8))


    # Return summary + data
    summary = {
        "facility_total": len(facility_df),
        "zero_total": len(zerodose_df)
    }
    data = {"zerodose_df": zerodose_df, "facility_df": facility_df}
    insights = []

    return figs, summary, insights, data


# ---------------------------
# Streamlit render function
# ---------------------------

def render_gender_dashboard(facility_path=None, zerodose_path=None):
    facility_path = resolve_path(facility_path or "facility_visit.csv")
    zerodose_path = resolve_path(zerodose_path or "zerodose.csv")

    try:
        with st.spinner("Preloading gender analytics..."):
            figs, summary, insights, data_out = get_gender_dashboard(facility_path, zerodose_path)
    except FileNotFoundError as e:
        st.error(str(e))
        return
    except Exception as e:
        st.exception(e)
        return

    st.subheader("Zero-Dose Gender Analytics")
    st.info(f"Zero-dose records: {summary['zero_total']:,} | Facility records: {summary['facility_total']:,}")

    if insights:
        with st.expander("Key gender insights (summary)"):
            for line in insights:
                st.write("•", line)

    for title, fig in figs:
        st.markdown(f"### {title}")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Preview data (zero-dose sample)"):
        st.dataframe(data_out["zerodose_df"].head(200))
