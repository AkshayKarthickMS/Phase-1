# dashboard_6_additional.py
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import re
import streamlit as st


# ---------- CONFIG ----------
ZERODOSE_CSV = "data/zerodose.csv"
VISIT_CSV = "data/facility_visit.csv"


# ---------- HELPERS ----------
def parse_age_to_months(age_str):
    """Convert age description (e.g. '1 year, 3 months') into total months."""
    if pd.isna(age_str):
        return 0
    years, months, weeks = 0, 0, 0
    try:
        parts = re.findall(r"(\d+)", str(age_str))
        if len(parts) >= 1:
            years = int(parts[0])
        if len(parts) >= 2:
            months = int(parts[1])
        if len(parts) >= 3:
            weeks = int(parts[2])
    except Exception:
        pass
    return years * 12 + months + weeks / 4


# ---------- CACHED DATA PREP ----------
@st.cache_data(show_spinner=False)
def load_and_prepare_data():
    facility_df = pd.read_csv(VISIT_CSV)
    zd_df = pd.read_csv(ZERODOSE_CSV)

    zd_df['Settlement'] = zd_df['Settlement'].fillna("Unknown")
    zd_df['LGA'] = zd_df['LGA'].fillna("Unknown")
    zd_df['AgeMonths'] = zd_df['Age at Enrollment'].apply(parse_age_to_months)

    facility_df['visit_date'] = pd.to_datetime(facility_df['visit_date'], errors='coerce')

    # Ward-wise counts
    ward_counts = zd_df.groupby(['LGA', 'Settlement']).size().reset_index(name='ZeroDoseCount')

    # Enrollment timeline
    zd_df['Enrollment Date'] = pd.to_datetime(zd_df['Enrollment Date'], errors='coerce')
    zd_df['EnrollMonth'] = zd_df['Enrollment Date'].dt.to_period("M").astype(str)
    trend_df = zd_df.groupby('EnrollMonth').size().reset_index(name='Cases')

    return zd_df, facility_df, ward_counts, trend_df


# ---------- CACHED VISUALS ----------
@st.cache_resource(show_spinner=False)
def build_figures():
    zd_df, facility_df, ward_counts, trend_df = load_and_prepare_data()

    # Visualization 1: Heatmap
    fig_heatmap = px.density_heatmap(
        ward_counts,
        x="Settlement",
        y="LGA",
        z="ZeroDoseCount",
        color_continuous_scale="Reds",
        title="Ward-wise Zero-Dose Density Heatmap"
    )

    # Visualization 2: Network Graph (Settlement-based)
    G = nx.Graph()
    for settlement in zd_df['Settlement'].unique():
        G.add_node(settlement, size=zd_df[zd_df['Settlement'] == settlement].shape[0])

    settlements_list = list(zd_df['Settlement'].unique())
    for i in range(len(settlements_list) - 1):
        G.add_edge(settlements_list[i], settlements_list[i + 1])

    pos = nx.spring_layout(G, seed=42)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x, node_y, node_size, node_text = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_size.append(G.nodes[node]['size'] * 5)
        node_text.append(f"{node}: {G.nodes[node]['size']} cases")

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text', text=node_text,
        marker=dict(size=node_size, color='red', opacity=0.8),
        hoverinfo='text'
    )

    fig_network = go.Figure(data=[edge_trace, node_trace])
    fig_network.update_layout(title="Household Network Graph (Settlement-based)")

    # Visualization 3: Drop-off Trend
    fig_trend = px.line(
        trend_df,
        x="EnrollMonth",
        y="Cases",
        markers=True,
        title="Drop-off Trend by Enrollment Month"
    )

    return fig_heatmap, fig_network, fig_trend


# ---------- DASHBOARD RENDER ----------
def show_additional_dashboard():
    st.header("🧭 Additional Analytics — Zero-Dose & Immunization")
    st.caption("This section combines supplementary analyses like heatmaps, network graphs, and trend analysis.")

    fig_heatmap, fig_network, fig_trend = build_figures()

    st.subheader("Ward-wise Zero-Dose Density Heatmap")
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.subheader("Household Network Graph (Settlement-based)")
    st.plotly_chart(fig_network, use_container_width=True)

    st.subheader("Drop-off Trend by Enrollment Month")
    st.plotly_chart(fig_trend, use_container_width=True)

    st.info("Network relationships here are simulated to represent settlement adjacency, not real household links.")


# ---------- FOR TESTING LOCALLY ----------
if __name__ == "__main__":
    import streamlit.web.cli as stcli
    stcli._main_run_clExplicit("dashboard_6_additional.py", "streamlit run")
