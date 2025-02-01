import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats
import re

# Function to load data
def load_data(file_path):
    try:
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

# Function to summarize data
def summarize_data(df):
    print("\n--- Data Summary ---")
    print(f"Shape: {df.shape}")
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nDescriptive Statistics:")
    print(df.describe(include='all'))

# Function to detect anomalies using Z-score
def detect_anomalies(df, threshold=3):
    print("\n--- Anomaly Detection (Z-score Method) ---")
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        z_scores = np.abs(stats.zscore(df[col].dropna()))
        outliers = df[col][(z_scores > threshold)]
        if not outliers.empty:
            print(f"Anomalies detected in '{col}':")
            print(outliers)

# Function to generate automated insights
def generate_insights(df):
    print("\n--- Automated Insights ---")
    numeric_cols = df.select_dtypes(include=['number']).columns
    correlations = df[numeric_cols].corr()
    high_corr = correlations[(correlations > 0.8) & (correlations < 1.0)]
    if not high_corr.empty:
        print("Highly correlated pairs:")
        print(high_corr.dropna(how='all').dropna(axis=1, how='all'))

    for col in numeric_cols:
        skewness = df[col].skew()
        if abs(skewness) > 1:
            print(f"'{col}' has high skewness ({skewness:.2f}).")

# Function to visualize data
def visualize_data(df):
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns

    # Histograms for numeric columns
    for col in numeric_cols:
        fig = px.histogram(df, x=col, title=f'Distribution of {col}', marginal="box")
        fig.show()

    # Bar plots for categorical columns
    for col in categorical_cols:
        value_counts = df[col].value_counts().reset_index()
        value_counts.columns = [col, 'Count']  # Renaming columns
        fig = px.bar(value_counts, x=col, y='Count',
                     title=f'Value Counts of {col}', labels={col: col, 'Count': 'Count'})
        fig.show()

    # Correlation heatmap
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.show()

    # Box plots for outlier detection
    for col in numeric_cols:
        fig = px.box(df, y=col, title=f'Box Plot of {col}')
        fig.show()

    # Scatter plots for numerical relationships
    if len(numeric_cols) >= 2:
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                fig = px.scatter(df, x=numeric_cols[i], y=numeric_cols[j],
                                 title=f'Scatter Plot of {numeric_cols[i]} vs {numeric_cols[j]}',
                                 trendline='ols')
                fig.show()

    # Time series plots for datetime columns
    for col in datetime_cols:
        for num_col in numeric_cols:
            fig = px.line(df, x=col, y=num_col, title=f'Time Series Plot with {col} vs {num_col}')
            fig.show()

# Function to interpret user queries
def interpret_query(query, df):
    query = query.lower()

    # Query 1: "What is the average of [column]?"
    avg_match = re.search(r"average of (\w+)", query)
    if avg_match:
        column = avg_match.group(1)
        if column in df.columns:
            avg_value = df[column].mean()
            print(f"Average of '{column}': {avg_value}")
        else:
            print(f"Column '{column}' not found.")

    # Query 2: "Are there any anomalies in [column]?"
    anomaly_match = re.search(r"anomalies in (\w+)", query)
    if anomaly_match:
        column = anomaly_match.group(1)
        if column in df.columns:
            z_scores = np.abs(stats.zscore(df[column].dropna()))
            outliers = df[column][(z_scores > 3)]
            if not outliers.empty:
                print(f"Anomalies detected in '{column}':")
                print(outliers)
            else:
                print(f"No anomalies found in '{column}'.")
        else:
            print(f"Column '{column}' not found.")

    # Query 3: "What is the correlation between [column1] and [column2]?"
    correlation_match = re.search(r"correlation between (\w+) and (\w+)", query)
    if correlation_match:
        col1, col2 = correlation_match.groups()
        if col1 in df.columns and col2 in df.columns:
            corr_value = df[[col1, col2]].corr().iloc[0, 1]
            print(f"Correlation between '{col1}' and '{col2}': {corr_value}")
        else:
            print(f"One or both columns '{col1}' and '{col2}' not found.")

    # Query 4: "What is the trend over time for [column]?"
    trend_match = re.search(r"trend over time for (\w+)", query)
    if trend_match:
        column = trend_match.group(1)
        if column in df.columns and pd.api.types.is_datetime64_any_dtype(df[column]):
            fig = px.line(df, x=df[column], y=df[column], title=f'Trend over Time for {column}')
            fig.show()
        else:
            print(f"Column '{column}' not found or it is not a valid datetime column.")

# Main function
def main():
    file_path = input("Enter the path to your CSV or Excel file: ")
    data = load_data(file_path)
    if data is not None:
        summarize_data(data)
        detect_anomalies(data)
        generate_insights(data)
        visualize_data(data)

        while True:
            query = input("\nAsk a question about your data (or type 'exit' to quit): ")
            if query.lower() == 'exit':
                break
            interpret_query(query, data)

if __name__ == "__main__":
    main()
