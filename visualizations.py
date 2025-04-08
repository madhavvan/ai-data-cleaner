import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import networkx as nx
import folium
from streamlit_folium import st_folium
from data_utils import forecast_time_series, perform_clustering, suggest_visualization
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import dask.dataframe as dd  # For lazy loading of large datasets
import io
import kaleido  # For exporting Plotly figures

def render_visualization_page(df):
    """Render the visualization page with advanced options, dashboards, and export capabilities."""
    if df is None or df.empty:
        st.warning("Please upload a dataset first on the Upload page.")
        return
    
    st.title("📊 Visualize Your Dataset")

    # Update progress
    st.session_state.progress["Visualize"] = "In Progress"

    # Convert to Dask DataFrame for lazy loading if dataset is large
    if len(df) > 10000:
        st.info("Large dataset detected. Using lazy loading for performance.")
        df = dd.from_pandas(df, npartitions=4)

    # Initialize session state for filtered data, clustering, and dashboard
    if 'filtered_df' not in st.session_state:
        st.session_state.filtered_df = df.copy()
    if 'clustering_labels' not in st.session_state:
        st.session_state.clustering_labels = None
    if 'cluster_cols' not in st.session_state:
        st.session_state.cluster_cols = []
    if 'dashboard_charts' not in st.session_state:
        st.session_state.dashboard_charts = []
    if 'dashboard_filters' not in st.session_state:
        st.session_state.dashboard_filters = {}

    # Compute Dask DataFrame if necessary
    if isinstance(st.session_state.filtered_df, dd.DataFrame):
        st.session_state.filtered_df = st.session_state.filtered_df.compute()

    with st.form("visualization_form"):
        viz_types = [
            "Bar", "Histogram", "Scatter", "Line", "Box", "Violin", "Heatmap (Correlation)", "Pie",
            "Time Series Forecast", "3D Scatter", "Geospatial Map", "Area Chart", "Strip Plot",
            "Swarm Plot", "Density Plot", "ECDF Plot", "Treemap", "Sunburst Chart", "Dendrogram",
            "Network Graph", "Choropleth Map", "Heatmap (Geospatial)", "Timeline", "Gantt Chart",
            "Calendar Heatmap", "Parallel Coordinates", "Radar Chart", "Bubble Chart", "Surface Plot",
            "Word Cloud", "Gauge Chart", "Funnel Chart", "Sankey Diagram", "Waterfall Chart",
            "Pair Plot", "Joint Plot", "Clustering"
        ]
        viz_type = st.selectbox("Select Visualization Type", viz_types, help="Choose the type of chart to display.")
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        all_cols = df.columns.tolist()
        time_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        object_cols = df.select_dtypes(include=['object']).columns.tolist()

        # Dynamic Filters (shared across dashboard)
        st.subheader("Filter Data")
        filter_col = st.selectbox("Filter By (Optional)", ["None"] + all_cols, key="global_filter_col")
        filtered_df = df.copy()
        if filter_col != "None":
            col_type = df[filter_col].dtype
            if pd.api.types.is_numeric_dtype(col_type):
                min_val, max_val = float(df[filter_col].min()), float(df[filter_col].max())
                if pd.isna(min_val) or pd.isna(max_val):
                    st.warning(f"Column {filter_col} contains missing values. Filtering may exclude these rows.")
                elif min_val == max_val:
                    st.warning(f"Column {filter_col} has identical values ({min_val}). Range filtering is not applicable.")
                else:
                    selected_range = st.slider(
                        f"Filter {filter_col} (range)", 
                        min_val, 
                        max_val, 
                        (min_val, max_val),
                        step=(max_val - min_val) / 100 if max_val != min_val else 1.0,
                        key=f"filter_{filter_col}"
                    )
                    filtered_df = filtered_df[
                        (filtered_df[filter_col] >= selected_range[0]) & (filtered_df[filter_col] <= selected_range[1])
                    ]
                    st.session_state.dashboard_filters[filter_col] = selected_range
            elif pd.api.types.is_datetime64_any_dtype(col_type):
                min_date, max_date = df[filter_col].min(), df[filter_col].max()
                if pd.isna(min_date) or pd.isna(max_date):
                    st.warning(f"Column {filter_col} contains missing values. Filtering may exclude these rows.")
                elif min_date == max_date:
                    st.warning(f"Column {filter_col} has identical dates ({min_date}). Date range filtering is not applicable.")
                else:
                    selected_dates = st.date_input(
                        f"Filter {filter_col} (date range)", 
                        [min_date, max_date],
                        min_value=min_date,
                        max_value=max_date,
                        key=f"filter_{filter_col}"
                    )
                    if len(selected_dates) == 2:
                        start_date, end_date = selected_dates
                        filtered_df = filtered_df[
                            (filtered_df[filter_col] >= pd.to_datetime(start_date)) & 
                            (filtered_df[filter_col] <= pd.to_datetime(end_date))
                        ]
                        st.session_state.dashboard_filters[filter_col] = (start_date, end_date)
            else:
                unique_vals = df[filter_col].dropna().unique().tolist()
                if len(unique_vals) == 1:
                    st.warning(f"Column {filter_col} has a single unique value ({unique_vals[0]}). Filtering is not applicable.")
                else:
                    filter_value = st.multiselect(f"Select {filter_col} values", unique_vals, default=unique_vals)
                    if filter_value:
                        filtered_df = filtered_df[filtered_df[filter_col].isin(filter_value)]
                        st.session_state.dashboard_filters[filter_col] = filter_value

        if not filtered_df.empty:
            st.session_state.filtered_df = filtered_df
            st.write(f"Filtered dataset: {filtered_df.shape[0]} rows, {filtered_df.shape[1]} columns")
        else:
            st.warning("Filters resulted in an empty dataset. Please adjust your filters.")
            st.session_state.filtered_df = df.copy()
            return

        # Reset clustering labels if not using Clustering visualization
        if viz_type != "Clustering" and 'Cluster' in st.session_state.filtered_df.columns:
            st.session_state.filtered_df = st.session_state.filtered_df.drop(columns=['Cluster'])
            st.session_state.clustering_labels = None
            st.session_state.cluster_cols = []

        # Visualization Configuration
        x_col = y_col = hue_col = z_col = lat_col = lon_col = size_col = color_col = None
        periods = None
        if viz_type in ["Bar", "Scatter", "Line", "3D Scatter", "Bubble Chart"]:
            x_col = st.selectbox("X-Axis Column", all_cols, help="Select column for the X-axis.")
            y_col = st.selectbox("Y-Axis Column", numeric_cols, help="Select column for the Y-axis.")
            hue_col = st.selectbox("Group By (Optional)", ["None"] + all_cols, help="Optional grouping column.")
            if viz_type == "3D Scatter":
                z_col = st.selectbox("Z-Axis Column", numeric_cols, help="Select column for the Z-axis.")
            if viz_type == "Bubble Chart":
                size_col = st.selectbox("Size By", numeric_cols, help="Select column for bubble size.")

        elif viz_type == "Histogram":
            x_col = st.selectbox("Column", numeric_cols, help="Select column to plot.")
            hue_col = st.selectbox("Group By (Optional)", ["None"] + all_cols, help="Optional grouping column.")

        elif viz_type in ["Box", "Violin", "Strip Plot", "Swarm Plot"]:
            x_col = st.selectbox("X-Axis Column (Optional)", ["None"] + all_cols, help="Optional X-axis column.")
            y_col = st.selectbox("Y-Axis Column", numeric_cols, help="Select column for the Y-axis.")
            hue_col = st.selectbox("Group By (Optional)", ["None"] + all_cols, help="Optional grouping column.")

        elif viz_type == "Heatmap (Correlation)":
            x_col = st.multiselect("Columns for Correlation", numeric_cols, default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols, help="Select numeric columns for correlation heatmap.")

        elif viz_type == "Pie":
            x_col = st.selectbox("Categories", all_cols, help="Select column for pie segments.")
            y_col = st.selectbox("Values", numeric_cols, help="Select column for pie values.")

        elif viz_type == "Time Series Forecast":
            x_col = st.selectbox("Time Column", time_cols, help="Select datetime column.")
            y_col = st.selectbox("Value Column", numeric_cols, help="Select column to forecast.")
            periods = st.slider("Forecast Periods", 1, 30, 5, help="Number of future periods to predict.")
            freq = st.selectbox("Frequency", ["D", "M", "Y"], help="Select the frequency of the time series data.")

        elif viz_type == "Geospatial Map":
            lat_col = st.selectbox("Latitude Column", numeric_cols, help="Select latitude column.")
            lon_col = st.selectbox("Longitude Column", numeric_cols, help="Select longitude column.")
            size_col = st.selectbox("Size By (Optional)", ["None"] + numeric_cols)
            color_col = st.selectbox("Color By (Optional)", ["None"] + all_cols)

        elif viz_type == "Choropleth Map":
            geo_col = st.selectbox("Geographic Column (e.g., country, state)", all_cols)
            value_col = st.selectbox("Values", numeric_cols)

        elif viz_type == "Heatmap (Geospatial)":
            lat_col = st.selectbox("Latitude Column", numeric_cols)
            lon_col = st.selectbox("Longitude Column", numeric_cols)

        elif viz_type == "Area Chart":
            x_col = st.selectbox("Time Column", time_cols)
            y_col = st.selectbox("Y-Axis Column", numeric_cols)
            hue_col = st.selectbox("Group By (Optional)", ["None"] + all_cols)

        elif viz_type in ["Density Plot", "ECDF Plot"]:
            x_col = st.selectbox("Column", numeric_cols)

        elif viz_type in ["Treemap", "Sunburst Chart"]:
            path_cols = st.multiselect("Hierarchy (select multiple columns)", all_cols)
            values_col = st.selectbox("Values", numeric_cols)

        elif viz_type == "Dendrogram":
            selected_cols = st.multiselect("Select numerical columns", numeric_cols, default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols)

        elif viz_type == "Network Graph":
            source_col = st.selectbox("Source Node", all_cols)
            target_col = st.selectbox("Target Node", [col for col in all_cols if col != source_col])
            weight_col = st.selectbox("Weight (Optional)", ["None"] + numeric_cols)

        elif viz_type == "Timeline":
            time_col = st.selectbox("Time Column", time_cols)
            event_col = st.selectbox("Event Column", all_cols)

        elif viz_type == "Gantt Chart":
            start_col = st.selectbox("Start Time", time_cols)
            end_col = st.selectbox("End Time", time_cols)
            task_col = st.selectbox("Task", all_cols)

        elif viz_type == "Calendar Heatmap":
            date_col = st.selectbox("Date Column", time_cols)
            value_col = st.selectbox("Values", numeric_cols)

        elif viz_type == "Parallel Coordinates":
            selected_cols = st.multiselect("Select numerical columns", numeric_cols, default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols)
            color_col = st.selectbox("Color By (Optional)", ["None"] + all_cols)

        elif viz_type == "Radar Chart":
            selected_cols = st.multiselect("Select numerical columns", numeric_cols, default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols)
            group_col = st.selectbox("Group", all_cols)

        elif viz_type == "Surface Plot":
            x_col = st.selectbox("X-Axis Column", numeric_cols)
            y_col = st.selectbox("Y-Axis Column", numeric_cols)
            z_col = st.selectbox("Z-Axis Column", numeric_cols)

        elif viz_type == "Word Cloud":
            text_col = st.selectbox("Text Column", object_cols)

        elif viz_type == "Gauge Chart":
            value_col = st.selectbox("Value", numeric_cols)
            max_value = st.number_input("Max Value", value=float(st.session_state.filtered_df[value_col].max()) * 1.2)

        elif viz_type == "Funnel Chart":
            stages_col = st.selectbox("Stages", all_cols)
            values_col = st.selectbox("Values", numeric_cols)

        elif viz_type == "Sankey Diagram":
            source_col = st.selectbox("Source", all_cols)
            target_col = st.selectbox("Target", [col for col in all_cols if col != source_col])
            value_col = st.selectbox("Value", numeric_cols)

        elif viz_type == "Waterfall Chart":
            measure_col = st.selectbox("Measure (e.g., 'relative', 'total')", all_cols)
            x_col = st.selectbox("X-Axis (categories)", all_cols)
            y_col = st.selectbox("Y-Axis (values)", numeric_cols)

        elif viz_type == "Pair Plot":
            selected_cols = st.multiselect("Select numerical columns", numeric_cols, default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols)

        elif viz_type == "Joint Plot":
            x_col = st.selectbox("X-Axis Column", numeric_cols)
            y_col = st.selectbox("Y-Axis Column", numeric_cols)

        elif viz_type == "Clustering":
            cluster_cols = st.multiselect("Select columns for clustering", numeric_cols, default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols)
            n_clusters = st.slider("Number of clusters", 2, 10, 3)

        title = st.text_input("Chart Title", f"{viz_type} Visualization", help="Customize the chart title.")
        add_to_dashboard = st.checkbox("Add to Dashboard", help="Add this chart to a dashboard for combined viewing.")
        submit_button = st.form_submit_button("Generate Visualization")

    if submit_button:
        try:
            df = st.session_state.filtered_df
            fig = None
            is_wordcloud = False
            is_jointplot = False
            is_clustering = False

            if viz_type == "Bar":
                fig = px.bar(df, x=x_col, y=y_col, color=None if hue_col == "None" else hue_col, title=title)
            elif viz_type == "Histogram":
                fig = px.histogram(df, x=x_col, color=None if hue_col == "None" else hue_col, title=title)
            elif viz_type == "Scatter":
                fig = px.scatter(df, x=x_col, y=y_col, color=None if hue_col == "None" else hue_col, title=title)
            elif viz_type == "Line":
                fig = px.line(df, x=x_col, y=y_col, color=None if hue_col == "None" else hue_col, title=title)
            elif viz_type == "Box":
                fig = px.box(df, x=None if x_col == "None" else x_col, y=y_col, color=None if hue_col == "None" else hue_col, title=title)
            elif viz_type == "Violin":
                fig = px.violin(df, x=None if x_col == "None" else x_col, y=y_col, color=None if hue_col == "None" else hue_col, title=title)
            elif viz_type == "Heatmap (Correlation)":
                if len(x_col) < 2:
                    st.error("Please select at least two numerical columns for the heatmap.")
                    return
                corr = df[x_col].corr()
                fig = px.imshow(corr, text_auto=True, title=title)
            elif viz_type == "Pie":
                fig = px.pie(df, names=x_col, values=y_col, title=title)
            elif viz_type == "Time Series Forecast":
                if not time_cols:
                    st.error("No datetime columns available for forecasting.")
                    return
                forecast_df = forecast_time_series(df, y_col, periods, time_col=x_col, freq=freq)
                historical = df[[x_col, y_col]].copy()
                historical['Type'] = 'Historical'
                forecast_df = forecast_df.reset_index().rename(columns={'index': x_col, y_col: y_col})
                forecast_df['Type'] = 'Forecast'
                combined_df = pd.concat([historical, forecast_df], ignore_index=True)
                fig = px.line(combined_df, x=x_col, y=y_col, color='Type', title=title)
                fig.add_vline(x=df[x_col].iloc[-1], line_dash="dash", line_color="red")
            elif viz_type == "3D Scatter":
                fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color=None if hue_col == "None" else hue_col, title=title)
            elif viz_type == "Geospatial Map":
                fig = px.scatter_mapbox(df, lat=lat_col, lon=lon_col, size=None if size_col == "None" else size_col, color=None if color_col == "None" else color_col, title=title, zoom=3)
                fig.update_layout(mapbox_style="open-street-map")
            elif viz_type == "Choropleth Map":
                fig = px.choropleth(df, locations=geo_col, locationmode="country names", color=value_col, title=title)
            elif viz_type == "Heatmap (Geospatial)":
                fig = px.density_mapbox(df, lat=lat_col, lon=lon_col, radius=10, center=dict(lat=df[lat_col].mean(), lon=df[lon_col].mean()), zoom=5, mapbox_style="open-street-map", title=title)
            elif viz_type == "Area Chart":
                if not time_cols:
                    st.error("No datetime columns available for area chart.")
                    return
                fig = px.area(df, x=x_col, y=y_col, color=None if hue_col == "None" else hue_col, title=title)
            elif viz_type == "Strip Plot":
                fig = px.strip(df, x=None if x_col == "None" else x_col, y=y_col, color=None if hue_col == "None" else hue_col, title=title)
            elif viz_type == "Swarm Plot":
                fig = px.strip(df, x=None if x_col == "None" else x_col, y=y_col, color=None if hue_col == "None" else hue_col, title=title)
                fig.update_traces(jitter=1)
            elif viz_type == "Density Plot":
                fig = px.density_contour(df, x=x_col, title=title)
                fig.update_traces(contours_coloring="fill", contours_showlabels=True)
            elif viz_type == "ECDF Plot":
                x = df[x_col].dropna()
                ecdf = np.arange(1, len(x) + 1) / len(x)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=np.sort(x), y=ecdf, mode='lines', name='ECDF'))
                fig.update_layout(title=title, xaxis_title=x_col, yaxis_title="Cumulative Probability")
            elif viz_type == "Treemap":
                if not path_cols:
                    st.error("Please select at least one column for the hierarchy.")
                    return
                fig = px.treemap(df, path=path_cols, values=values_col, title=title)
            elif viz_type == "Sunburst Chart":
                if not path_cols:
                    st.error("Please select at least one column for the hierarchy.")
                    return
                fig = px.sunburst(df, path=path_cols, values=values_col, title=title)
            elif viz_type == "Dendrogram":
                if len(selected_cols) < 2:
                    st.error("Please select at least two numerical columns.")
                    return
                X = df[selected_cols].dropna()
                Z = linkage(X, method='ward')
                fig = go.Figure()
                dendro = dendrogram(Z, no_plot=True)
                fig.add_trace(go.Scatter(x=dendro['icoord'][0], y=dendro['dcoord'][0], mode='lines', line=dict(color='white')))
                for i in range(1, len(dendro['icoord'])):
                    fig.add_trace(go.Scatter(x=dendro['icoord'][i], y=dendro['dcoord'][i], mode='lines', line=dict(color='white'), showlegend=False))
                fig.update_layout(title=title, xaxis_title="Sample Index", yaxis_title="Distance")
            elif viz_type == "Network Graph":
                G = nx.from_pandas_edgelist(df, source=source_col, target=target_col, edge_attr=None if weight_col == "None" else weight_col)
                pos = nx.spring_layout(G)
                edge_x, edge_y = [], []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                node_x, node_y = [], []
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='gray'), hoverinfo='none'))
                fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers', marker=dict(size=10, color='white'), text=list(G.nodes()), hoverinfo='text'))
                fig.update_layout(title=title, showlegend=False)
            elif viz_type == "Timeline":
                fig = px.scatter(df, x=time_col, y=[0] * len(df), text=event_col, title=title)
                fig.update_traces(textposition="top center")
                fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
            elif viz_type == "Gantt Chart":
                fig = px.timeline(df, x_start=start_col, x_end=end_col, y=task_col, title=title)
            elif viz_type == "Calendar Heatmap":
                data = df.groupby(date_col)[value_col].sum().reset_index()
                fig = px.density_heatmap(data, x=data[date_col].dt.day, y=data[date_col].dt.month, z=value_col, title=title)
            elif viz_type == "Parallel Coordinates":
                if len(selected_cols) < 2:
                    st.error("Please select at least two numerical columns.")
                    return
                fig = px.parallel_coordinates(df, dimensions=selected_cols, color=None if color_col == "None" else color_col, title=title)
            elif viz_type == "Radar Chart":
                if len(selected_cols) < 2:
                    st.error("Please select at least two numerical columns.")
                    return
                grouped = df.groupby(group_col)[selected_cols].mean().reset_index()
                fig = go.Figure()
                for _, row in grouped.iterrows():
                    fig.add_trace(go.Scatterpolar(r=[row[col] for col in selected_cols], theta=selected_cols, fill='toself', name=row[group_col]))
                fig.update_layout(title=title)
            elif viz_type == "Bubble Chart":
                fig = px.scatter(df, x=x_col, y=y_col, size=size_col, color=None if hue_col == "None" else hue_col, title=title)
            elif viz_type == "Surface Plot":
                data = df.pivot_table(index=x_col, columns=y_col, values=z_col).fillna(0)
                fig = go.Figure(data=[go.Surface(z=data.values, x=data.columns, y=data.index)])
                fig.update_layout(title=title, scene=dict(xaxis_title=x_col, yaxis_title=y_col, zaxis_title=z_col))
            elif viz_type == "Word Cloud":
                text = " ".join(df[text_col].dropna().astype(str))
                wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text)
                plt.figure(figsize=(10, 5), facecolor='black')
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                st.pyplot(plt)
                is_wordcloud = True
            elif viz_type == "Gauge Chart":
                value = df[value_col].mean()
                fig = go.Figure(go.Indicator(mode="gauge+number", value=value, domain={'x': [0, 1], 'y': [0, 1]}, title={'text': title}, gauge={'axis': {'range': [0, max_value]}, 'bar': {'color': "white"}}))
            elif viz_type == "Funnel Chart":
                fig = px.funnel(df, x=values_col, y=stages_col, title=title)
            elif viz_type == "Sankey Diagram":
                label_list = list(set(df[source_col].tolist() + df[target_col].tolist()))
                label_dict = {label: idx for idx, label in enumerate(label_list)}
                source = df[source_col].map(label_dict)
                target = df[target_col].map(label_dict)
                value = df[value_col]
                fig = go.Figure(data=[go.Sankey(node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=label_list), link=dict(source=source, target=target, value=value))])
                fig.update_layout(title=title)
            elif viz_type == "Waterfall Chart":
                fig = go.Figure(go.Waterfall(x=df[x_col], measure=df[measure_col], y=df[y_col], textposition="auto"))
                fig.update_layout(title=title)
            elif viz_type == "Pair Plot":
                if len(selected_cols) < 2:
                    st.error("Please select at least two numerical columns.")
                    return
                fig = px.scatter_matrix(df, dimensions=selected_cols, title=title)
            elif viz_type == "Joint Plot":
                fig = sns.jointplot(data=df, x=x_col, y=y_col, kind="scatter")
                st.pyplot(fig)
                is_jointplot = True
            elif viz_type == "Clustering":
                if len(cluster_cols) < 2:
                    st.error("Please select at least two numerical columns for clustering.")
                    return
                labels = perform_clustering(df, cluster_cols, n_clusters)
                st.session_state.clustering_labels = labels
                st.session_state.cluster_cols = cluster_cols
                df['Cluster'] = labels
                st.session_state.filtered_df = df
                if len(cluster_cols) >= 2:
                    fig_2d = px.scatter(df, x=cluster_cols[0], y=cluster_cols[1], color=labels.astype(str), labels={'color': 'Cluster'}, title="Clustering Results (2D Scatter Plot)", hover_data=cluster_cols)
                    st.plotly_chart(fig_2d, use_container_width=True)
                if len(cluster_cols) >= 3:
                    fig_3d = px.scatter_3d(df, x=cluster_cols[0], y=cluster_cols[1], z=cluster_cols[2], color=labels.astype(str), labels={'color': 'Cluster'}, title="Clustering Results (3D Scatter Plot)", hover_data=cluster_cols)
                    st.plotly_chart(fig_3d, use_container_width=True)
                cluster_counts = pd.Series(labels).value_counts().sort_index()
                fig_dist = px.bar(x=cluster_counts.index.astype(str), y=cluster_counts.values, labels={'x': 'Cluster', 'y': 'Number of Points'}, title="Cluster Distribution", color=cluster_counts.index.astype(str))
                st.plotly_chart(fig_dist, use_container_width=True)
                st.write("Dataset with Cluster Labels:")
                st.dataframe(df, use_container_width=True)
                is_clustering = True

            if not is_wordcloud and not is_jointplot and not is_clustering:
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white', title_font_color='white', showlegend=True)
                st.plotly_chart(fig, use_container_width=True)

                # Export Visualization
                st.subheader("Export Visualization")
                export_format = st.selectbox("Select Export Format", ["PNG", "SVG", "PDF"])
                if st.button("Export"):
                    with st.spinner("Exporting visualization..."):
                        buffer = io.BytesIO()
                        fig.write_image(buffer, format=export_format.lower())
                        st.download_button(
                            label=f"Download as {export_format}",
                            data=buffer,
                            file_name=f"{title}.{export_format.lower()}",
                            mime=f"image/{export_format.lower()}"
                        )

                # Add to Dashboard
                if add_to_dashboard:
                    chart_config = {
                        "type": viz_type,
                        "title": title,
                        "fig": fig,
                        "x_col": x_col,
                        "y_col": y_col,
                        "hue_col": hue_col,
                        "z_col": z_col,
                        "lat_col": lat_col,
                        "lon_col": lon_col,
                        "size_col": size_col,
                        "color_col": color_col,
                        "periods": periods,
                        "freq": freq if viz_type == "Time Series Forecast" else None
                    }
                    st.session_state.dashboard_charts.append(chart_config)
                    st.success("Chart added to dashboard!")

                # Dynamic Visualization Suggestions
                st.subheader("Suggested Follow-Up Visualizations")
                with st.spinner("Generating suggestions..."):
                    suggested_viz, reason = suggest_visualization(df)
                    st.write(f"- **{suggested_viz}**: {reason}")
                    # Suggest additional visualizations based on current chart
                    if viz_type == "Scatter" and len(numeric_cols) >= 2:
                        st.write("- **Heatmap (Correlation)**: Explore correlations between numerical variables.")
                    elif viz_type == "Bar" and len(object_cols) > 0:
                        st.write("- **Pie Chart**: Visualize the distribution of a categorical column.")
                    elif viz_type == "Line" and time_cols:
                        st.write("- **Time Series Forecast**: Predict future values for this time series.")

            st.success("Visualization generated successfully!")
            st.session_state.progress["Visualize"] = "Done"
        except ValueError as e:
            st.error(f"Invalid input: {str(e)}. Please check your column selections and data types.")
            st.session_state.progress["Visualize"] = "Failed"
        except Exception as e:
            st.error(f"Error generating visualization: {str(e)}. Ensure columns have valid data and try again.")
            st.session_state.progress["Visualize"] = "Failed"

    # Dashboard Section
    st.subheader("Create Dashboard")
    if st.session_state.dashboard_charts:
        st.write("### Dashboard")
        for i, chart in enumerate(st.session_state.dashboard_charts):
            st.write(f"**Chart {i+1}: {chart['title']}**")
            st.plotly_chart(chart['fig'], use_container_width=True)
            if st.button(f"Remove Chart {i+1} from Dashboard", key=f"remove_chart_{i}"):
                st.session_state.dashboard_charts.pop(i)
                st.rerun()
    else:
        st.info("No charts added to dashboard yet. Check 'Add to Dashboard' to include charts.")