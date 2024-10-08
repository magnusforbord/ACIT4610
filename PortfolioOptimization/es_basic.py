import numpy as np
import pandas as pd
import os
import csv
import time

# Determine the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data', 'processed')

# Define the results directory
results_dir = os.path.join(script_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

# Load monthly returns
monthly_returns = pd.read_csv(os.path.join(data_dir, 'monthly_returns.csv'), index_col=0)
mean_returns = monthly_returns.mean()
covariance_matrix = pd.read_csv(os.path.join(data_dir, 'covariance_matrix.csv'), index_col=0)
covariance_matrix = covariance_matrix.values  # Convert to numpy array

def objective_function(weights, mean_returns):
    expected_return = np.dot(weights, mean_returns)
    return expected_return

def calculate_portfolio_variance(weights, covariance_matrix):
    return np.dot(weights.T, np.dot(covariance_matrix, weights))

def initialize_population(pop_size, num_assets):
    population = []
    for _ in range(pop_size):
        weights = np.random.dirichlet(np.ones(num_assets), size=1)[0]
        population.append(weights)
    return np.array(population)

def evaluate_population(population, mean_returns):
    fitness = []
    for weights in population:
        expected_return = objective_function(weights, mean_returns)
        fitness.append(expected_return)
    return np.array(fitness)

def mutate(weights, mutation_strength):
    num_assets = len(weights)
    mutated_weights = weights + np.random.normal(0, mutation_strength, num_assets)
    # Ensure weights are non-negative and sum to 1
    mutated_weights = np.clip(mutated_weights, 0, None)
    mutated_weights /= np.sum(mutated_weights)
    return mutated_weights

def evolution_strategies(mean_returns, covariance_matrix, num_assets, pop_size=50, num_generations=100, mutation_strength=0.05):
    μ = pop_size // 2  # Number of parents selected
    λ = pop_size       # Number of offspring generated

    # Initialize population
    population = initialize_population(μ, num_assets)
    best_fitness_history = []

    for generation in range(num_generations):
        # Generate offspring
        offspring = [mutate(parent, mutation_strength) for parent in population]
        offspring = np.array(offspring)

        # Combine parents and offspring
        combined_population = np.vstack((population, offspring))

        # Evaluate fitness
        fitness = evaluate_population(combined_population, mean_returns)

        # Select the best μ individuals
        indices = np.argsort(fitness)[-μ:]
        population = combined_population[indices]

        # Record best fitness of the generation
        best_fitness = fitness[indices[-1]]
        best_fitness_history.append(best_fitness)

    # Calculate mean fitness over generations
    mean_fitness = np.mean(best_fitness_history)
    final_fitness = evaluate_population(population, mean_returns)
    best_index = np.argmax(final_fitness)
    best_weights = population[best_index]
    best_return = final_fitness[best_index]

    return best_weights, best_return, mean_fitness

if __name__ == "__main__":
    num_runs = 30  # Number of runs
    results = []   # List to store results from each run

    num_assets = len(mean_returns)

    for run in range(1, num_runs + 1):
        start_time = time.time()

        best_weights, best_return, mean_fitness = evolution_strategies(
            mean_returns.values,
            covariance_matrix,
            num_assets,
            pop_size=50,
            num_generations=100,
            mutation_strength=0.05
        )

        end_time = time.time()
        training_time = end_time - start_time

        # Calculate portfolio variance for the best weights
        portfolio_variance = calculate_portfolio_variance(best_weights, covariance_matrix)

        # Collect the results
        results.append([
            run,
            mean_fitness,  # Store mean fitness instead of entire history
            best_return,
            portfolio_variance,
            best_weights.tolist(),  # Convert numpy array to list for CSV
            mean_fitness,  # Store mean fitness over generations
            training_time
        ])

        print(f"Run {run}/{num_runs} completed. Mean Fitness over Generations: {mean_fitness:.6f}")

    # Save results to CSV
    csv_file_name = os.path.join(results_dir, 'es_basic_results.csv')
    with open(csv_file_name, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write header
        writer.writerow([
            'Run', 'Mean Fitness', 'Expected Return', 'Portfolio Variance',
            'Weights', 'Mean Fitness over Generations', 'Training Time'
        ])
        # Write data rows
        for result in results:
            writer.writerow(result)

    print(f"\nAll runs completed. Results saved to '{csv_file_name}'.")
