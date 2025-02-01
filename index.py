import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Function to visualize data
def visualize_data(df):
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns

    # Histograms for numeric columns
    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()

    # Bar plots for categorical columns
    for col in categorical_cols:
        plt.figure(figsize=(8, 4))
        df[col].value_counts().plot(kind='bar')
        plt.title(f'Value Counts of {col}')
        plt.show()

    # Correlation heatmap
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.show()

    # Box plots for outlier detection
    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[col])
        plt.title(f'Box Plot of {col}')
        plt.show()

    # Scatter plots for numerical relationships
    if len(numeric_cols) >= 2:
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                plt.figure(figsize=(8, 4))
                sns.scatterplot(x=df[numeric_cols[i]], y=df[numeric_cols[j]])
                plt.title(f'Scatter Plot of {numeric_cols[i]} vs {numeric_cols[j]}')
                plt.show()

    # Time series plots for datetime columns
    for col in datetime_cols:
        plt.figure(figsize=(10, 5))
        for num_col in numeric_cols:
            plt.plot(df[col], df[num_col], label=num_col)
        plt.title(f'Time Series Plot with {col}')
        plt.xlabel(col)
        plt.ylabel('Values')
        plt.legend()
        plt.show()

    # Pair plots for multi-variable relationships
    if len(numeric_cols) > 1:
        sns.pairplot(df[numeric_cols].dropna())
        plt.suptitle('Pair Plot for Numerical Variables', y=1.02)
        plt.show()

# Main function
def main():
    file_path = input("Enter the path to your CSV or Excel file: ")
    data = load_data(file_path)
    if data is not None:
        summarize_data(data)
        visualize_data(data)

if __name__ == "__main__":
    main()
