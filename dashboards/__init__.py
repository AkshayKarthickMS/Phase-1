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

# Import all dashboards
from .dashboard_vaccine_field import render_vaccine_dashboard
from .dashboard_gender import get_gender_dashboard
from .dashboard_3_age import get_dashboard_age
from .dashboard_4_household import render_household_dashboard
from .dashboard_5_timeseries import show_timeseries_dashboard
from .dashboard_6_additional import show_additional_dashboard
from .dashboard_7_risk import get_dashboard_risk

# List of available dashboards for easy iteration in app.py
DASHBOARDS = {
    "Vaccine Field Overview": render_vaccine_dashboard,
    "Gender Distribution": get_gender_dashboard,
    "Age Distribution": get_dashboard_age,
    "Household Analysis": render_household_dashboard,
    "Time Series & Drop-off Trends": show_timeseries_dashboard,
    "Additional Insights": show_additional_dashboard,
    "Predictive Risk & Segmentation": get_dashboard_risk,
}

__all__ = [
    "render_vaccine_dashboard",
    "get_gender_dashboard",
    "get_dashboard_age",
    "render_household_dashboard",
    "show_timeseries_dashboard",
    "show_additional_dashboard",
    "get_dashboard_risk",
    "DASHBOARDS",
]
