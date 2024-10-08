import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Determine the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the results directory
results_dir = os.path.join(script_dir, 'results')

# List of algorithms and their result files
algorithms = {
    'EP Basic': 'ep_basic_results.csv',
    'EP Advanced': 'ep_advanced_results.csv',
    'ES Basic': 'es_basic_results.csv',
    'ES Advanced': 'es_advanced_results.csv',
    'ES (μ + λ)': 'es_mu_plus_lambda_results.csv',
    'ES (μ, λ)': 'es_mu_comma_lambda_results.csv'
}

# Initialize dictionaries to store metrics
expected_returns = {}
portfolio_variances = {}
training_times = {}

# Iterate over each algorithm
for algo_name, file_name in algorithms.items():
    file_path = os.path.join(results_dir, file_name)
    if not os.path.exists(file_path):
        print(f"File {file_path} not found. Skipping {algo_name}.")
        continue
    
    data = pd.read_csv(file_path)
    
    expected_returns[algo_name] = data['Expected Return'].values
    portfolio_variances[algo_name] = data['Portfolio Variance'].values
    training_times[algo_name] = data['Training Time'].values

# Calculate summary statistics for all algorithms
summary_stats = []
for algo_name in algorithms.keys():
    mean_return = np.mean(expected_returns[algo_name])
    std_return = np.std(expected_returns[algo_name])
    mean_variance = np.mean(portfolio_variances[algo_name])
    std_variance = np.std(portfolio_variances[algo_name])
    mean_time = np.mean(training_times[algo_name])
    std_time = np.std(training_times[algo_name])
    
    summary_stats.append({
        'Algorithm': algo_name,
        'Mean Return': mean_return,
        'Std Return': std_return,
        'Mean Variance': mean_variance,
        'Std Variance': std_variance,
        'Mean Training Time': mean_time,
        'Std Training Time': std_time
    })

summary_df = pd.DataFrame(summary_stats)
print(summary_df)
summary_csv_file = os.path.join(results_dir, 'algorithm_summary_statistics.csv')
summary_df.to_csv(summary_csv_file, index=False)

# Boxplot for expected returns across all algorithms
returns_data = [expected_returns[algo_name] for algo_name in algorithms.keys()]
labels = list(algorithms.keys())
plt.figure(figsize=(12, 8))
plt.boxplot(returns_data, tick_labels=labels)
plt.title('Distribution of Expected Returns Across Algorithms')
plt.ylabel('Expected Return')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

boxplot_file = os.path.join(results_dir, 'expected_returns_boxplot.png')
plt.savefig(boxplot_file)
plt.show()

# Stability Statistics Calculation
stability_stats = []
for algo_name in algorithms.keys():
    mean_return = np.mean(expected_returns[algo_name])
    std_return = np.std(expected_returns[algo_name])
    cv_return = std_return / mean_return if mean_return != 0 else np.nan
    stability_stats.append({
        'Algorithm': algo_name,
        'Mean Return': mean_return,
        'Std Return': std_return,
        'Coefficient of Variation': cv_return
    })

stability_df = pd.DataFrame(stability_stats)
print(stability_df)
stability_csv_file = os.path.join(results_dir, 'algorithm_stability_statistics.csv')
stability_df.to_csv(stability_csv_file, index=False)
