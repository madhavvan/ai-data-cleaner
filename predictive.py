import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List
import plotly.express as px
import plotly.graph_objects as go
from data_utils import train_ml_model, perform_clustering
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import lime
import lime.lime_tabular

# Optional SHAP import with fallback (no top-level Streamlit command)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

def render_predictive_page(df: pd.DataFrame) -> None:
    """
    Render the predictive analytics page with ML model training, SHAP/LIME visualizations, fairness metrics, and clustering.

    Args:
        df (pd.DataFrame): Input DataFrame.
    """
    # Move the SHAP warning here
    if not SHAP_AVAILABLE:
        st.warning("SHAP library not installed. Feature importance visualizations will be unavailable.")

    if df is None or df.empty:
        st.error("No dataset available. Please upload a dataset on the Upload page.")
        return

    # Update progress
    st.session_state.progress["Predictive"] = "In Progress"

    # Model Training Section
    st.subheader("Train a Machine Learning Model")
    with st.expander("Model Training", expanded=True):
        task_type = st.radio("Task Type", ["classification", "regression"], help="Choose the type of ML task.")
        target_col = st.selectbox("Target Column", df.columns, help="Select the column to predict.")
        feature_cols = st.multiselect("Feature Columns", [col for col in df.columns if col != target_col], 
                                     help="Select columns to use as predictors.")
        
        model_type = st.selectbox("Model Type", ["RandomForest", "XGBoost", "LightGBM"], 
                                  help="Choose the ML model to train.")
        
        if st.button("Train Model"):
            if not target_col or not feature_cols:
                st.warning("Please select a target column and at least one feature column.")
            else:
                with st.spinner("Training model... This may take a few minutes."):
                    progress_bar = st.progress(0)
                    try:
                        # Simulate progress updates for better UX
                        for i in range(1, 101):
                            progress_bar.progress(i)
                            if i == 25:
                                st.text("Preprocessing data...")
                            elif i == 50:
                                st.text("Tuning hyperparameters...")
                            elif i == 75:
                                st.text("Finalizing model...")
                        # Manual model training
                        model, score, explainer, shap_values, X_test = train_ml_model(df, target_col, feature_cols, task_type, model_type=model_type)

                        if model is None:
                            st.error("Model training failed. Please check the dataset and try again.")
                        else:
                            st.session_state['model'] = model
                            st.session_state['explainer'] = explainer
                            st.session_state['shap_values'] = shap_values
                            st.session_state['X_test'] = X_test
                            st.session_state['feature_cols'] = feature_cols
                            st.session_state['task_type'] = task_type
                            st.success(f"Model trained successfully! {task_type.capitalize()} score: {score:.2f}")
                            
                            # Fairness Metrics
                            if task_type == "classification":
                                st.subheader("Fairness Metrics")
                                y_pred = model.predict(X_test)
                                # Compute confusion matrix and classification report
                                cm = confusion_matrix(y_test, y_pred)
                                fig_cm = px.imshow(cm, text_auto=True, title="Confusion Matrix")
                                st.plotly_chart(fig_cm, use_container_width=True)
                                st.write("Classification Report:")
                                st.text(classification_report(y_test, y_pred))
                                # Demographic parity (simplified, assuming binary classification and a sensitive attribute)
                                if "gender" in df.columns:  # Example sensitive attribute
                                    sensitive_attr = df.loc[X_test.index, "gender"]
                                    pred_df = pd.DataFrame({'prediction': y_pred, 'gender': sensitive_attr})
                                    parity = pred_df.groupby('gender')['prediction'].mean()
                                    st.write("Demographic Parity (Prediction Rates by Gender):")
                                    st.write(parity)
                                    if abs(parity.diff().iloc[-1]) > 0.1:
                                        st.warning("Potential fairness issue: Prediction rates differ significantly across groups.")
                    except Exception as e:
                        st.error(f"Error during model training: {str(e)}. Please ensure the selected columns are valid.")
                    finally:
                        progress_bar.empty()

    # SHAP Visualization Section
    if 'model' in st.session_state and SHAP_AVAILABLE and st.session_state.get('explainer') is not None:
        st.subheader("Feature Importance (SHAP)")
        with st.expander("SHAP Visualizations", expanded=True):
            try:
                shap_values = st.session_state['shap_values']
                X_test = st.session_state['X_test']
                feature_cols = st.session_state['feature_cols']

                # Summary Plot (Interactive with Plotly)
                st.write("### Feature Importance Summary")
                shap_df = pd.DataFrame(shap_values, columns=feature_cols)
                mean_shap = np.abs(shap_df).mean().sort_values(ascending=False)
                fig = px.bar(
                    x=mean_shap.values,
                    y=mean_shap.index,
                    orientation='h',
                    labels={'x': 'Mean |SHAP Value| (Impact on Model Output)', 'y': 'Feature'},
                    title="Feature Importance (Mean |SHAP Value|)",
                    color=mean_shap.values,
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    title_font_color='white',
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

                # Interactive Force Plot for a single prediction
                st.write("### Individual Prediction Explanation")
                sample_idx = st.slider("Select a sample to explain", 0, len(X_test) - 1, 0)
                shap.initjs()
                force_plot = shap.force_plot(
                    st.session_state['explainer'].expected_value,
                    shap_values[sample_idx],
                    X_test.iloc[sample_idx],
                    show=False,
                    matplotlib=False
                )
                st_shap(force_plot, height=200)

            except Exception as e:
                st.error(f"Error generating SHAP visualizations: {str(e)}. Please ensure SHAP is properly configured.")

    # LIME Visualization Section
    if 'model' in st.session_state:
        st.subheader("Local Model Interpretability (LIME)")
        with st.expander("LIME Explanations", expanded=True):
            try:
                X_test = st.session_state['X_test']
                feature_cols = st.session_state['feature_cols']
                model = st.session_state['model']
                task_type = st.session_state['task_type']

                # Initialize LIME explainer
                lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    X_test.values,
                    feature_names=feature_cols,
                    class_names=[str(i) for i in range(len(np.unique(df[target_col]))) if task_type == "classification"],
                    mode="classification" if task_type == "classification" else "regression"
                )

                # Select a sample to explain
                sample_idx = st.slider("Select a sample to explain (LIME)", 0, len(X_test) - 1, 0)
                instance = X_test.iloc[sample_idx].values

                # Generate LIME explanation
                if task_type == "classification":
                    exp = lime_explainer.explain_instance(
                        instance,
                        model.predict_proba,
                        num_features=len(feature_cols)
                    )
                else:
                    exp = lime_explainer.explain_instance(
                        instance,
                        lambda x: model.predict(x).reshape(-1),
                        num_features=len(feature_cols)
                    )

                # Display LIME explanation
                st.write("### LIME Explanation")
                fig, ax = plt.subplots()
                exp.as_pyplot_figure()
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error generating LIME explanations: {str(e)}. Please ensure the model and data are compatible.")

    # Clustering Section
    st.subheader("Clustering Results")
    with st.expander("Clustering", expanded=True):
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cluster_cols = st.multiselect("Select columns for clustering", numeric_cols, 
                                     help="Select at least two numerical columns for clustering.")
        n_clusters = st.slider("Number of clusters", 2, 10, 3, help="Choose the number of clusters to form.")
        
        if st.button("Run Clustering"):
            if len(cluster_cols) < 2:
                st.warning("Please select at least two columns for clustering.")
            else:
                with st.spinner("Performing clustering..."):
                    try:
                        labels = perform_clustering(df, cluster_cols, n_clusters)
                        df['Cluster'] = labels
                        st.session_state['cleaned_df'] = df
                        st.session_state['clustering_labels'] = labels
                        st.session_state['cluster_cols'] = cluster_cols
                        st.success("Clustering completed successfully!")
                    except Exception as e:
                        st.error(f"Error performing clustering: {str(e)}. Please ensure the selected columns contain valid numerical data.")

        # Interactive Clustering Visualization
        if 'clustering_labels' in st.session_state and st.session_state['clustering_labels'] is not None:
            try:
                cluster_cols = st.session_state['cluster_cols']
                labels = st.session_state['clustering_labels']
                
                if len(cluster_cols) >= 2:
                    # 2D Scatter Plot
                    fig_2d = px.scatter(
                        df,
                        x=cluster_cols[0],
                        y=cluster_cols[1],
                        color=labels.astype(str),
                        labels={'color': 'Cluster'},
                        title="Clustering Results (2D Scatter Plot)",
                        hover_data=cluster_cols
                    )
                    fig_2d.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='white',
                        title_font_color='white'
                    )
                    st.plotly_chart(fig_2d, use_container_width=True)

                if len(cluster_cols) >= 3:
                    # 3D Scatter Plot
                    fig_3d = px.scatter_3d(
                        df,
                        x=cluster_cols[0],
                        y=cluster_cols[1],
                        z=cluster_cols[2],
                        color=labels.astype(str),
                        labels={'color': 'Cluster'},
                        title="Clustering Results (3D Scatter Plot)",
                        hover_data=cluster_cols
                    )
                    fig_3d.update_layout(
                        scene=dict(
                            xaxis_title=cluster_cols[0],
                            yaxis_title=cluster_cols[1],
                            zaxis_title=cluster_cols[2],
                            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="white"),
                            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="white"),
                            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="white")
                        ),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='white',
                        title_font_color='white'
                    )
                    st.plotly_chart(fig_3d, use_container_width=True)

                # Cluster Distribution
                cluster_counts = pd.Series(labels).value_counts().sort_index()
                fig_dist = px.bar(
                    x=cluster_counts.index.astype(str),
                    y=cluster_counts.values,
                    labels={'x': 'Cluster', 'y': 'Number of Points'},
                    title="Cluster Distribution",
                    color=cluster_counts.index.astype(str),
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
                fig_dist.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    title_font_color='white',
                    showlegend=False
                )
                st.plotly_chart(fig_dist, use_container_width=True)

            except Exception as e:
                st.error(f"Error generating clustering visualizations: {str(e)}. Please try different columns or check the data.")

    st.session_state.progress["Predictive"] = "Done"

# Helper function to render SHAP plots in Streamlit
def st_shap(plot, height: Optional[int] = None) -> None:
    """
    Render SHAP plots in Streamlit.

    Args:
        plot: SHAP plot object.
        height (Optional[int]): Height of the plot in pixels.
    """
    try:
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        st.components.v1.html(shap_html, height=height)
    except Exception as e:
        st.error(f"Error rendering SHAP plot: {str(e)}.")