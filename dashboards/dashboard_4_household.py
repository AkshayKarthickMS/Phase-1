import os
from datetime import datetime

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st


# === PATH RESOLUTION HELPERS ===
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


# === MAIN DASHBOARD LOGIC ===
@st.cache_resource
def get_household_dashboard(
    zerodose_path="zerodose.csv",
    visit_path="facility_visit.csv",
    top_n=30
):
    """
    Load data, compute settlement-level summaries and priority scores, build figures.
    Returns:
      - figs: list of (title, plotly_fig)
      - table_df: merged table (settlement-level)
      - summary: dict with simple metrics
    """
    # Resolve and validate input files
    zerodose_path = resolve_path(zerodose_path)
    visit_path = resolve_path(visit_path)
    missing = [p for p in (zerodose_path, visit_path) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing input files: {', '.join(missing)}")

    # Load
    zd = pd.read_csv(zerodose_path)
    vis = pd.read_csv(visit_path)

    # Normalize column names and basic fields
    zd.columns = zd.columns.str.strip()
    vis.columns = vis.columns.str.strip()

    zd['Settlement_std'] = zd.get('Settlement', '').astype(str).str.strip().str.title()
    zd['Woman_or_child_std'] = zd.get('Woman or child', '').astype(str).str.strip().str.lower()
    zd['Status_std'] = zd.get('Status', '').astype(str).str.strip().str.title()
    zd['LGA_std'] = zd.get('LGA', '').astype(str).str.strip().str.title()
    vis['lga_name_std'] = vis.get('lga_name', '').astype(str).str.strip().str.title()

    # Keep only child records for zero-dose counts
    zd_children = zd[zd['Woman_or_child_std'] == 'child'].copy()

    zero_dose_counts = (
        zd_children.groupby(['Settlement_std', 'LGA_std'])
        .agg(zero_dose_child_count=('ID', 'nunique'))
        .reset_index()
        .sort_values('zero_dose_child_count', ascending=False)
    )

    # Determine visited LGAs
    visited_lgas = set(vis['lga_name_std'].dropna().unique())
    zd['Visited_LGA_flag'] = zd['LGA_std'].isin(visited_lgas)

    dropouts = zd[(zd['Visited_LGA_flag']) & (zd['Status_std'].str.lower() != 'resolved')].copy()
    dropout_counts = (
        dropouts.groupby(['Settlement_std', 'LGA_std'])
        .agg(dropout_count=('ID', 'nunique'))
        .reset_index()
        .sort_values('dropout_count', ascending=False)
    )

    # Merge
    merged = pd.merge(
        zero_dose_counts,
        dropout_counts,
        on=['Settlement_std', 'LGA_std'],
        how='outer'
    ).fillna(0)

    merged['zero_dose_child_count'] = merged['zero_dose_child_count'].astype(int)
    merged['dropout_count'] = merged['dropout_count'].astype(int)

    # Priority score: 0.7 * normalized zero-dose + 0.3 * normalized dropout
    zmax = max(merged['zero_dose_child_count'].max(), 1)
    dmax = max(merged['dropout_count'].max(), 1)
    if zmax == 0 and dmax == 0:
        merged['priority_score'] = 0.0
    else:
        merged['priority_score'] = (
            0.7 * (merged['zero_dose_child_count'] / zmax) +
            0.3 * (merged['dropout_count'] / dmax)
        )

    merged = merged.sort_values('priority_score', ascending=False).reset_index(drop=True)

    # Build figures
    top_zero = merged.sort_values('zero_dose_child_count', ascending=False).head(top_n)
    fig_zero = px.bar(
        top_zero,
        x='zero_dose_child_count',
        y='Settlement_std',
        orientation='h',
        color='LGA_std',
        hover_data=['dropout_count', 'priority_score'],
        title=f"Top {top_n} Settlements — Number of Zero-dose Children",
        labels={'zero_dose_child_count': 'Zero-dose children', 'Settlement_std': 'Settlement'},
        height=600
    )
    fig_zero.update_layout(yaxis={'categoryorder': 'total ascending'})

    top_drop = merged.sort_values('dropout_count', ascending=False).head(top_n)
    fig_drop = px.bar(
        top_drop,
        x='dropout_count',
        y='Settlement_std',
        orientation='h',
        color='LGA_std',
        hover_data=['zero_dose_child_count', 'priority_score'],
        title=f"Top {top_n} Settlements — Recurring Dropout Cases (Unresolved)",
        labels={'dropout_count': 'Dropout cases', 'Settlement_std': 'Settlement'},
        height=600
    )
    fig_drop.update_layout(yaxis={'categoryorder': 'total ascending'})

    fig_priority = px.scatter(
        merged,
        x='zero_dose_child_count',
        y='dropout_count',
        size='priority_score',
        color='LGA_std',
        hover_name='Settlement_std',
        title='Settlement Prioritization (size ~ priority score)',
        labels={'zero_dose_child_count': 'Zero-dose children', 'dropout_count': 'Dropout cases'},
        height=600
    )

    # Small metrics summary
    summary = {
        "total_settlements": int(len(merged)),
        "settlements_gt1_zero_dose": int((merged['zero_dose_child_count'] > 1).sum()),
        "settlements_with_dropout": int((merged['dropout_count'] > 0).sum()),
        "generated_at": datetime.now().isoformat()
    }

    # Prepare table (top 200)
    table_df = merged.rename(columns={
        'Settlement_std': 'Settlement',
        'LGA_std': 'LGA',
        'zero_dose_child_count': 'ZeroDoseChildren',
        'dropout_count': 'DropoutCases',
        'priority_score': 'PriorityScore'
    }).head(200)

    figs = [
        ("Top Settlements by Zero-dose Children", fig_zero),
        ("Top Settlements by Dropout Cases", fig_drop),
        ("Settlement Prioritization (scatter)", fig_priority),
    ]

    return figs, table_df, summary


# === STREAMLIT RENDERER ===
def render_household_dashboard(
    zerodose_path=None,
    visit_path=None,
    top_n=30
):
    """Streamlit renderer: preloads cached data via get_household_dashboard and shows figures + table."""
    try:
        with st.spinner("Preloading household-level analytics..."):
            figs, table_df, summary = get_household_dashboard(
                resolve_path(zerodose_path or "zerodose.csv"),
                resolve_path(visit_path or "facility_visit.csv"),
                top_n
            )
    except FileNotFoundError as e:
        st.error(str(e))
        return
    except Exception as e:
        st.exception(e)
        return

    st.subheader("Household / Settlement Level Analysis")
    st.info(f"Total settlements: {summary['total_settlements']:,} | "
            f"Settlements with >1 zero-dose child: {summary['settlements_gt1_zero_dose']:,} | "
            f"Settlements with dropout cases: {summary['settlements_with_dropout']:,}")

    # Show figures
    for title, fig in figs:
        st.markdown(f"### {title}")
        st.plotly_chart(fig, use_container_width=True)

    # Table preview (Top 200)
    st.markdown("### Settlement Details (top 200)")
    st.dataframe(table_df)

    st.markdown(
        "Notes: Settlement is used as a household proxy. Priority score = "
        "`0.7 * normalized_zero_dose + 0.3 * normalized_dropout`. "
        "Tune weights per operational needs."
    )
