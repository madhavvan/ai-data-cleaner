import streamlit as st
import plotly.express as px
import pandas as pd
from data_utils import train_ml_model, perform_clustering
from datetime import datetime  # Added for download link
import base64  # Added for download link

def get_download_link(df, filename):
    """Generate a download link for the dataset."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download clustered dataset</a>'

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
            try:
                model, score, explainer, shap_values, X_test = train_ml_model(df, target_col, feature_cols, task_type.lower())
                if model is not None and score is not None:
                    st.write(f"Model Accuracy: {score:.2f}")
                    
                    # Feature importance using SHAP, if available
                    if explainer is not None and shap_values is not None and X_test is not None:
                        st.subheader("Feature Importance")
                        try:
                            import shap
                            shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
                            st.pyplot()
                        except ImportError:
                            st.warning("SHAP library not installed. Feature importance plots are unavailable.")
                    else:
                        st.warning("Feature importance plots are unavailable due to missing SHAP library.")
                else:
                    st.error("Model training failed. Check logs or ensure valid input data.")
            except Exception as e:
                st.error(f"Error during model training: {str(e)}")
    
    st.subheader("Clustering")
    cluster_cols = st.multiselect("Columns for Clustering", df.columns.tolist())
    n_clusters = st.slider("Number of Clusters", 2, 10, 3)
    if st.button("Perform Clustering", key="predictive_perform_clustering"):  # Added unique key
        with st.spinner("Clustering data..."):
            try:
                clustered_df = df.copy()  # Enhancement: Avoid mutating original df
                labels = perform_clustering(df, cluster_cols, n_clusters)
                clustered_df["Cluster"] = labels
                st.write("Clustered Data:")
                st.dataframe(clustered_df.head(10))
                fig = px.scatter(clustered_df, x=cluster_cols[0], y=cluster_cols[1], color="Cluster", title="Clustering Results")
                st.plotly_chart(fig)
                # Enhancement: Add download link for clustered data
                st.markdown(get_download_link(clustered_df, f"clustered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error during clustering: {str(e)}")