import streamlit as st
import plotly.express as px
import pandas as pd
from data_utils import forecast_time_series

def render_visualization_page(df):
    """Render the visualization page with advanced options."""
    if df is None:
        st.warning("Please upload a dataset first on the Upload page.")
        return
    
    st.title("📊 Visualize Your Dataset")
    
    with st.form("visualization_form"):
        viz_type = st.selectbox("Select Visualization Type", 
                              ["Bar", "Histogram", "Scatter", "Line", "Box", "Violin", "Heatmap", "Pie", "Time Series Forecast", "3D Scatter", "Geospatial Map"],
                              help="Choose the type of chart to display.")
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        all_cols = df.columns.tolist()
        time_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        
        # Dynamic filters
        filter_col = st.selectbox("Filter By (Optional)", ["None"] + all_cols)
        if filter_col != "None":
            filter_value = st.multiselect(f"Select {filter_col} values", df[filter_col].unique())
            if filter_value:
                df = df[df[filter_col].isin(filter_value)]

        if viz_type in ["Bar", "Scatter", "Line", "3D Scatter"]:
            x_col = st.selectbox("X-Axis Column", all_cols, help="Select column for the X-axis.")
            y_col = st.selectbox("Y-Axis Column", numeric_cols, help="Select column for the Y-axis.")
            hue_col = st.selectbox("Group By (Optional)", ["None"] + all_cols, help="Optional grouping column.")
            z_col = st.selectbox("Z-Axis Column (for 3D)", ["None"] + numeric_cols) if viz_type == "3D Scatter" else None
        
        elif viz_type == "Histogram":
            x_col = st.selectbox("Column", numeric_cols, help="Select column to plot.")
            hue_col = st.selectbox("Group By (Optional)", ["None"] + all_cols, help="Optional grouping column.")
            y_col = None
        
        elif viz_type in ["Box", "Violin"]:
            x_col = st.selectbox("X-Axis Column (Optional)", ["None"] + all_cols, help="Optional X-axis column.")
            y_col = st.selectbox("Y-Axis Column", numeric_cols, help="Select column for the Y-axis.")
            hue_col = st.selectbox("Group By (Optional)", ["None"] + all_cols, help="Optional grouping column.")
        
        elif viz_type == "Heatmap":
            x_col = st.multiselect("Columns for Correlation", numeric_cols, default=numeric_cols[:2], 
                                 help="Select numeric columns for correlation heatmap.")
            y_col = hue_col = None
        
        elif viz_type == "Pie":
            x_col = st.selectbox("Categories", all_cols, help="Select column for pie segments.")
            y_col = st.selectbox("Values", numeric_cols, help="Select column for pie values.")
            hue_col = None
        
        elif viz_type == "Time Series Forecast":
            x_col = st.selectbox("Time Column", time_cols, help="Select datetime column.")
            y_col = st.selectbox("Value Column", numeric_cols, help="Select column to forecast.")
            periods = st.slider("Forecast Periods", 1, 30, 5, help="Number of future periods to predict.")
            hue_col = None
        
        elif viz_type == "Geospatial Map":
            lat_col = st.selectbox("Latitude Column", numeric_cols, help="Select latitude column.")
            lon_col = st.selectbox("Longitude Column", numeric_cols, help="Select longitude column.")
            size_col = st.selectbox("Size By (Optional)", ["None"] + numeric_cols)
            color_col = st.selectbox("Color By (Optional)", ["None"] + all_cols)
            x_col = y_col = hue_col = None
        
        title = st.text_input("Chart Title", f"{viz_type} of {x_col or ''} vs {y_col or ''}", 
                            help="Customize the chart title.")
        
        submit_button = st.form_submit_button("Generate Visualization")
    
    if submit_button:
        try:
            if viz_type == "Bar":
                fig = px.bar(df, x=x_col, y=y_col, color=None if hue_col == "None" else hue_col, title=title)
            elif viz_type == "Histogram":
                fig = px.histogram(df, x=x_col, color=None if hue_col == "None" else hue_col, title=title)
            elif viz_type == "Scatter":
                fig = px.scatter(df, x=x_col, y=y_col, color=None if hue_col == "None" else hue_col, title=title)
            elif viz_type == "Line":
                fig = px.line(df, x=x_col, y=y_col, color=None if hue_col == "None" else hue_col, title=title)
            elif viz_type == "Box":
                fig = px.box(df, x=None if x_col == "None" else x_col, y=y_col, 
                           color=None if hue_col == "None" else hue_col, title=title)
            elif viz_type == "Violin":
                fig = px.violin(df, x=None if x_col == "None" else x_col, y=y_col, 
                              color=None if hue_col == "None" else hue_col, title=title)
            elif viz_type == "Heatmap":
                corr = df[x_col].corr()
                fig = px.imshow(corr, text_auto=True, title=title)
            elif viz_type == "Pie":
                fig = px.pie(df, names=x_col, values=y_col, title=title)
            elif viz_type == "Time Series Forecast":
                df.set_index(x_col, inplace=True)
                forecast_df = forecast_time_series(df, y_col, periods)
                combined_df = pd.concat([df[[y_col]], forecast_df])
                fig = px.line(combined_df, y=y_col, title=title)
                fig.add_vline(x=df.index[-1], line_dash="dash", line_color="red")
            elif viz_type == "3D Scatter":
                fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color=None if hue_col == "None" else hue_col, title=title)
            elif viz_type == "Geospatial Map":
                fig = px.scatter_mapbox(df, lat=lat_col, lon=lon_col, 
                                      size=None if size_col == "None" else size_col,
                                      color=None if color_col == "None" else color_col,
                                      title=title, zoom=3)
                fig.update_layout(mapbox_style="open-street-map")
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating visualization: {str(e)}")