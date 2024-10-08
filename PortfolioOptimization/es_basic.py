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
mean_returns = monthly_returns.mean().values  # Convert to numpy array
covariance_matrix = pd.read_csv(os.path.join(data_dir, 'covariance_matrix.csv'), index_col=0).values

def objective_function(weights, mean_returns):
    return np.dot(weights, mean_returns)

def calculate_portfolio_variance(weights, covariance_matrix):
    return np.dot(weights.T, np.dot(covariance_matrix, weights))

def initialize_population(pop_size, num_assets):
    population = []
    for _ in range(pop_size):
        weights = np.random.dirichlet(np.ones(num_assets))
        population.append(weights)
    return np.array(population)

def evaluate_population(population, mean_returns):
    fitness = np.array([objective_function(weights, mean_returns) for weights in population])
    return fitness

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
    mean_fitness_history = []

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

        # Record best and mean fitness of the generation
        best_fitness = fitness[indices[-1]]
        mean_fitness = np.mean(fitness)
        best_fitness_history.append(best_fitness)
        mean_fitness_history.append(mean_fitness)

    # After evolution, get the best individual
    final_fitness = evaluate_population(population, mean_returns)
    best_index = np.argmax(final_fitness)
    best_weights = population[best_index]
    best_return = final_fitness[best_index]
    # Calculate portfolio variance
    portfolio_variance = calculate_portfolio_variance(best_weights, covariance_matrix)
    # Mean fitness of the final generation
    mean_fitness_final_gen = np.mean(final_fitness)

    return best_weights, best_return, portfolio_variance, best_fitness, mean_fitness_final_gen

if __name__ == "__main__":
    num_runs = 30  # Number of runs
    results = []   # List to store results from each run

    num_assets = len(mean_returns)

    for run in range(1, num_runs + 1):
        start_time = time.time()

        best_weights, expected_return, portfolio_variance, best_fitness, mean_fitness = evolution_strategies(
            mean_returns,
            covariance_matrix,
            num_assets,
            pop_size=50,
            num_generations=100,
            mutation_strength=0.05
        )

        end_time = time.time()
        training_time = end_time - start_time

        # Collect the results
        results.append([
            run,
            best_fitness,           # Best Fitness
            mean_fitness,           # Mean Fitness of final generation
            expected_return,
            portfolio_variance,
            best_weights.tolist(),  # Convert numpy array to list for CSV
            training_time
        ])

        print(f"Run {run}/{num_runs} completed. Best Fitness: {best_fitness:.6f}")

    # Save results to CSV
    csv_file_name = os.path.join(results_dir, 'es_basic_results.csv')
    with open(csv_file_name, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write header
        writer.writerow([
            'Run', 'Best Fitness', 'Mean Fitness', 'Expected Return', 'Portfolio Variance',
            'Weights', 'Training Time'
        ])
        # Write data rows
        for result in results:
            writer.writerow(result)

    print(f"\nAll runs completed. Results saved to '{csv_file_name}'.")
