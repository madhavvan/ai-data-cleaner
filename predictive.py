import streamlit as st
import plotly.express as px
import pandas as pd
from data_utils import train_ml_model, perform_clustering

def render_predictive_page(df):
    """Render the predictive analytics page."""
    if df is None:
        st.warning("Please upload a dataset first on the Upload page.")
        return
    
    st.title("🔮 Predictive Analytics Dashboard")
    
    st.subheader("Predictive Modeling")
    target_col = st.selectbox("Target Column", df.columns.tolist())
    feature_cols = st.multiselect("Feature Columns", df.columns.tolist())
    task_type = st.radio("Task Type", ["Classification", "Regression"])
    
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            model, score, explainer, shap_values, X_test = train_ml_model(df, target_col, feature_cols, task_type.lower())
            st.write(f"Model Accuracy: {score:.2f}")
            
            # Feature importance using SHAP
            st.subheader("Feature Importance")
            shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
            st.pyplot()
    
    st.subheader("Clustering")
    cluster_cols = st.multiselect("Columns for Clustering", df.columns.tolist())
    n_clusters = st.slider("Number of Clusters", 2, 10, 3)
    if st.button("Perform Clustering"):
        with st.spinner("Clustering data..."):
            labels = perform_clustering(df, cluster_cols, n_clusters)
            df["Cluster"] = labels
            st.write("Clustered Data:")
            st.dataframe(df.head(10))
            fig = px.scatter(df, x=cluster_cols[0], y=cluster_cols[1], color="Cluster", title="Clustering Results")
            st.plotly_chart(fig)