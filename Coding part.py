import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_and_clean_data(filepath):
    """Load the dataset and perform initial cleaning."""
    df = pd.read_csv(filepath)
    df['Acidity'] = pd.to_numeric(df['Acidity'], errors='coerce')
    df_cleaned = df.dropna()
    return df_cleaned

def encode_quality(df):
    """Encode the 'Quality' column."""
    df['Quality'] = df['Quality'].map({'good': 1, 'bad': 0})
    return df

def statistical_summary(df):
    """Print statistical summaries of the dataset."""
    print("Description:\n", df.describe())
    print("\nCorrelation:\n", df.corr())
    print("\nSkewness:\n", df.skew())
    print("\nKurtosis:\n", df.kurtosis())

def plot_histogram(df, column):
    """Generate a histogram."""
    plt.figure(figsize=(8, 6))
    sns.histplot(df[column], bins=30, kde=True, color='blue')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

def plot_scatter(df, x_column, y_column):
    """Generate a scatter plot."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=x_column, y=y_column, data=df, hue='Quality', palette='viridis', alpha=0.6)
    plt.title(f'Relationship between {x_column} and {y_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.legend(title='Quality', labels=['Bad', 'Good'])
    plt.show()

def plot_heatmap(df):
    """Generate a heatmap of correlations."""
    corr_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()

def plot_box(df, column):
    """Generate a box plot."""
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Quality', y=column, data=df, palette='pastel')
    plt.title(f'Box Plot of {column} by Quality')
    plt.xlabel('Quality')
    plt.ylabel(column)
    plt.xticks([0, 1], ['Bad', 'Good'])
    plt.show()

def plot_violin(df, column):
    """Generate a violin plot."""
    plt.figure(figsize=(8, 6))
    sns.violinplot(x='Quality', y=column, data=df, palette='pastel', inner='quartile')
    plt.title(f'Violin Plot of {column} by Quality')
    plt.xlabel('Quality')
    plt.ylabel(column)
    plt.xticks([0, 1], ['Bad', 'Good'])
    plt.show()

# Update this with your dataset path
data_path = 'apple_quality.csv'
df_cleaned = load_and_clean_data(data_path)
df_encoded = encode_quality(df_cleaned)

# Statistical Summary
statistical_summary(df_encoded)

# Generate Plots
plot_histogram(df_encoded, 'Sweetness')
plot_scatter(df_encoded, 'Sweetness', 'Weight')
plot_heatmap(df_encoded.drop('A_id', axis=1))  # Drop 'A_id' for the heatmap
plot_box(df_encoded, 'Juiciness')
plot_violin(df_encoded, 'Juiciness')
