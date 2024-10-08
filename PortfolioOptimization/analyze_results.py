import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Set the style for seaborn
sns.set_theme(style="whitegrid")

# Determine the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, 'results')

# List of algorithms and their corresponding result files
algorithms = {
    'EP Basic': 'ep_basic_results.csv',
    'EP Advanced': 'ep_advanced_results.csv',
    'ES Basic': 'es_basic_results.csv',
    'ES Advanced': 'es_advanced_results.csv',
    'ES (μ, λ)': 'es_mu_comma_lambda_results.csv',
    'ES (μ + λ)': 'es_mu_plus_lambda_results.csv'
}

# DataFrames to store results from all algorithms
all_results = {}

# Read the results for each algorithm
for algo_name, file_name in algorithms.items():
    file_path = os.path.join(results_dir, file_name)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['Algorithm'] = algo_name  # Add a column for algorithm name
        all_results[algo_name] = df
    else:
        print(f"File not found: {file_path}")

# Proceed only if all files are found
if len(all_results) < len(algorithms):
    print("Some result files are missing. Please ensure all result files are present.")
    exit()

# Concatenate all results into a single DataFrame
combined_results = pd.concat(all_results.values(), ignore_index=True)

# Create a directory for saving figures
figures_dir = os.path.join(script_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)

# Function to calculate mean values for each algorithm
def calculate_means(combined_results):
    # Exclude non-numeric columns and columns not needed for mean calculation
    cols_to_exclude = ['Run', 'Weights', 'Algorithm']  # Exclude 'Algorithm' here
    # Select columns to include
    cols_to_include = [col for col in combined_results.columns if col not in cols_to_exclude]
    # Compute the mean
    means = combined_results.groupby('Algorithm')[cols_to_include].mean().reset_index()
    return means




# Calculate mean values
means_df = calculate_means(combined_results)

# --- Visualization 1: Mean Fitness Comparison ---

plt.figure(figsize=(10, 6))
sns.barplot(data=means_df, x='Algorithm', y='Mean Fitness', palette='viridis')
plt.title('Mean Fitness Comparison')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'mean_fitness_comparison.png'))
plt.show()

# --- Visualization 1a: Best Fitness Comparison ---

plt.figure(figsize=(10, 6))
sns.barplot(data=means_df, x='Algorithm', y='Best Fitness', palette='coolwarm')
plt.title('Best Fitness Comparison')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'best_fitness_comparison.png'))
plt.show()

# --- Visualization 2: Expected Return vs. Portfolio Variance ---

plt.figure(figsize=(10, 6))
sns.scatterplot(data=means_df, x='Portfolio Variance', y='Expected Return', hue='Algorithm', style='Algorithm', s=100)
plt.title('Expected Return vs. Portfolio Variance')
plt.xlabel('Portfolio Variance')
plt.ylabel('Expected Return')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'return_vs_variance.png'))
plt.show()

# --- Visualization 3: Weight Allocation Heatmap ---

# Prepare data for heatmap
# We will compute the average weight allocation for each algorithm
def extract_weights(df):
    weights_list = []
    for weights_str in df['Weights']:
        # Convert string representation of list to actual list
        weights = [float(w.strip()) for w in weights_str.strip('[]').split(',')]
        weights_list.append(weights)
    # Compute average weights across runs
    avg_weights = np.mean(weights_list, axis=0)
    return avg_weights

weights_dict = {}
for algo_name, df in all_results.items():
    avg_weights = extract_weights(df)
    weights_dict[algo_name] = avg_weights

# Determine the number of assets
num_assets = len(next(iter(weights_dict.values())))
asset_indices = [f'Asset {i+1}' for i in range(num_assets)]

# Create DataFrame for heatmap
weights_df = pd.DataFrame(weights_dict, index=asset_indices)

plt.figure(figsize=(12, 8))
sns.heatmap(weights_df, annot=True, cmap='YlGnBu')
plt.title('Average Weight Allocation Heatmap')
plt.xlabel('Algorithm')
plt.ylabel('Assets')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'weight_allocation_heatmap.png'))
plt.show()

# --- Visualization 4: Training Time Comparison ---

plt.figure(figsize=(10, 6))
sns.barplot(data=means_df, x='Algorithm', y='Training Time', palette='magma')
plt.title('Average Training Time Comparison')
plt.xticks(rotation=45)
plt.ylabel('Training Time (seconds)')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'training_time_comparison.png'))
plt.show()

# --- Statistical Analysis: Comparing Algorithms ---

# For the statistical test, we will use ANOVA to compare the 'Best Fitness' across algorithms
from scipy.stats import f_oneway

# Prepare data for ANOVA
fitness_data = [df['Best Fitness'] for df in all_results.values()]
algo_names = list(all_results.keys())

# Perform one-way ANOVA
f_stat, p_value = f_oneway(*fitness_data)
print(f"ANOVA Results:\nF-statistic: {f_stat:.4f}, p-value: {p_value:.4f}")

# Interpret the p-value
alpha = 0.05
if p_value < alpha:
    print("Result: The differences between algorithms are statistically significant.")
else:
    print("Result: The differences between algorithms are not statistically significant.")

# --- Visualization 5: Boxplot of Best Fitness per Algorithm ---

plt.figure(figsize=(12, 6))
sns.boxplot(data=combined_results, x='Algorithm', y='Best Fitness', palette='Set3')
plt.title('Distribution of Best Fitness per Algorithm')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'best_fitness_boxplot.png'))
plt.show()

# --- Visualization 6: Training Time per Run per Algorithm ---

plt.figure(figsize=(12, 6))
sns.boxplot(data=combined_results, x='Algorithm', y='Training Time', palette='Set2')
plt.title('Training Time per Run per Algorithm')
plt.xticks(rotation=45)
plt.ylabel('Training Time (seconds)')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'training_time_boxplot.png'))
plt.show()

# --- Additional Statistical Test: Pairwise Comparisons ---

# If ANOVA is significant, perform pairwise t-tests
if p_value < alpha:
    from itertools import combinations
    import statsmodels.stats.multicomp as mc

    # Perform Tukey's HSD test
    tukey = mc.pairwise_tukeyhsd(combined_results['Best Fitness'], combined_results['Algorithm'], alpha=0.05)
    print(tukey)

    # Plotting the results of Tukey's test
    tukey_plot = tukey.plot_simultaneous()
    plt.title('Tukey HSD Test for Best Fitness')
    plt.xlabel('Best Fitness Difference')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'tukey_hsd_best_fitness.png'))
    plt.show()
