"""
Dashboards Package
------------------
This package contains all individual dashboard modules for the Unified Vaccine Analytics App.

Each module corresponds to a specific domain of analysis:
1. Vaccine Field Overview
2. Gender Distribution
3. Age Analysis
4. Household-Level Insights
5. Time Series / Drop-off Trends
6. Additional Visualizations (Facility, Network, Heatmaps)
7. Predictive Risk & Segmentation (ML Models)
"""

# Import all dashboards (Renderers and Data Loaders)
from .dashboard_vaccine_field import render_vaccine_dashboard, get_vaccine_dashboard
from .dashboard_gender import render_gender_dashboard, get_gender_dashboard
from .dashboard_3_age import render_age_dashboard, get_dashboard_age
from .dashboard_4_household import render_household_dashboard, get_household_dashboard
from .dashboard_5_timeseries import render_timeseries_dashboard, get_timeseries_dashboard
from .dashboard_6_additional import render_additional_dashboard, get_additional_dashboard
from .dashboard_7_risk import render_dashboard_risk, get_dashboard_risk

# List of available dashboards for easy iteration in app.py
DASHBOARDS = {
    "Vaccine Field Overview": render_vaccine_dashboard,
    "Gender Distribution": render_gender_dashboard,
    "Age Distribution": render_age_dashboard,
    "Household Analysis": render_household_dashboard,
    "Time Series & Drop-off Trends": render_timeseries_dashboard,
    "Additional Insights": render_additional_dashboard,
    "Predictive Risk & Segmentation": render_dashboard_risk,
}

__all__ = [
    # Renderers (UI)
    "render_vaccine_dashboard",
    "render_gender_dashboard",
    "render_age_dashboard",
    "render_household_dashboard",
    "render_timeseries_dashboard",
    "render_additional_dashboard",
    "render_dashboard_risk",
    
    # Data Loaders (Logic)
    "get_vaccine_dashboard",
    "get_gender_dashboard",
    "get_dashboard_age",
    "get_household_dashboard",
    "get_timeseries_dashboard",
    "get_additional_dashboard",
    "get_dashboard_risk",
    
    # Registry
    "DASHBOARDS",
]