import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats
import re
import streamlit as st


# Function to load data
def load_data(file):
    try:
        if file is not None:
            if file.name.endswith('.csv'):
                data = pd.read_csv(file)
            elif file.name.endswith(('.xls', '.xlsx')):
                data = pd.read_excel(file)
            else:
                raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
            return data
        else:
            return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


# Function to summarize data
def summarize_data(df):
    with st.expander("Data Summary", expanded=False):  # Collapsible section for Data Summary, collapsed by default
        st.write(f"Shape: {df.shape}")
        st.write("\nData Types:")
        st.write(df.dtypes)
        st.write("\nMissing Values:")
        st.write(df.isnull().sum())
        st.write("\nDescriptive Statistics:")
        st.write(df.describe(include='all'))


# Function to detect anomalies using Z-score
def detect_anomalies(df, threshold=3):
    with st.expander("Anomaly Detection (Z-score Method)", expanded=False):  # Collapsible section for Anomaly Detection, collapsed by default
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outliers = df[col][(z_scores > threshold)]
            if not outliers.empty:
                st.write(f"Anomalies detected in '{col}':")
                st.write(outliers)


# Function to generate automated insights
def generate_insights(df):
    with st.expander("Automated Insights", expanded=False):  # Collapsible section for Automated Insights, collapsed by default
        numeric_cols = df.select_dtypes(include=['number']).columns
        correlations = df[numeric_cols].corr()
        high_corr = correlations[(correlations > 0.8) & (correlations < 1.0)]
        if not high_corr.empty:
            st.write("Highly correlated pairs:")
            st.write(high_corr.dropna(how='all').dropna(axis=1, how='all'))

        for col in numeric_cols:
            skewness = df[col].skew()
            if abs(skewness) > 1:
                st.write(f"'{col}' has high skewness ({skewness:.2f}).")


# Function to visualize data
def visualize_data(df):
    with st.expander("Visualizations", expanded=False):  # Collapsible section for Visualizations, collapsed by default
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns

        # Histograms for numeric columns
        for col in numeric_cols:
            fig = px.histogram(df, x=col, title=f'Distribution of {col}', marginal="box")
            st.plotly_chart(fig)

        # Bar plots for categorical columns
        for col in categorical_cols:
            value_counts = df[col].value_counts().reset_index()
            value_counts.columns = [col, 'Count']  # Renaming columns
            fig = px.bar(value_counts, x=col, y='Count',
                         title=f'Value Counts of {col}', labels={col: col, 'Count': 'Count'})
            st.plotly_chart(fig)

        # Correlation heatmap
        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
            plt.title('Correlation Heatmap')
            st.pyplot(plt)

        # Box plots for outlier detection
        for col in numeric_cols:
            fig = px.box(df, y=col, title=f'Box Plot of {col}')
            st.plotly_chart(fig)

        # Scatter plots for numerical relationships
        if len(numeric_cols) >= 2:
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    fig = px.scatter(df, x=numeric_cols[i], y=numeric_cols[j],
                                     title=f'Scatter Plot of {numeric_cols[i]} vs {numeric_cols[j]}',
                                     trendline='ols')
                    st.plotly_chart(fig)

        # Time series plots for datetime columns
        for col in datetime_cols:
            for num_col in numeric_cols:
                fig = px.line(df, x=col, y=num_col, title=f'Time Series Plot with {col} vs {num_col}')
                st.plotly_chart(fig)


# Main function for Streamlit
def main():
    st.title("Deep Statistical Analysis")

    # File upload
    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xls", "xlsx"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)

        if df is not None:
            # Display analysis and visualizations after file upload
            summarize_data(df)
            detect_anomalies(df)
            generate_insights(df)
            visualize_data(df)


if __name__ == "__main__":
    main()
