import numpy as np
import pandas as pd
import os
import csv
import time

# Determine the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data', 'processed')

# Define the results directory and create it if it doesn't exist
results_dir = os.path.join(script_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

# Load monthly returns data and calculate mean returns
monthly_returns = pd.read_csv(os.path.join(data_dir, 'monthly_returns.csv'), index_col=0)
mean_returns = monthly_returns.mean().values

# Load the covariance matrix of the returns
covariance_matrix = pd.read_csv(os.path.join(data_dir, 'covariance_matrix.csv'), index_col=0).values

def objective_function(weights, mean_returns):
    """
    Calculate the expected return of the portfolio.
    """
    return np.dot(weights, mean_returns)

def calculate_portfolio_variance(weights, covariance_matrix):
    """
    Calculate the variance of the portfolio.
    """
    return np.dot(weights.T, np.dot(covariance_matrix, weights))

def initialize_population(pop_size, num_assets):
    """
    Initialize the population with random weights that sum to 1.
    """
    population = []
    for _ in range(pop_size):
        weights = np.random.dirichlet(np.ones(num_assets))
        population.append(weights)
    return np.array(population)

def evaluate_population(population, mean_returns):
    """
    Evaluate the fitness of each individual in the population.
    """
    fitness = np.array([objective_function(weights, mean_returns) for weights in population])
    return fitness

def mutate(weights, mutation_strength):
    """
    Mutate the weights of a portfolio with a specified mutation strength.
    """
    num_assets = len(weights)
    # Apply normal perturbation to weights
    mutated_weights = weights + np.random.normal(0, mutation_strength, num_assets)
    # Ensure weights are non-negative and sum to 1
    mutated_weights = np.clip(mutated_weights, 0, None)
    mutated_weights /= np.sum(mutated_weights)
    return mutated_weights

def evolution_strategies(mean_returns, covariance_matrix, num_assets, pop_size=50, num_generations=100, mutation_strength=0.05):
    """
    Evolution Strategies algorithm for portfolio optimization.
    """
    μ = pop_size // 2
    λ = pop_size

    # Initialize the population with μ individuals
    population = initialize_population(μ, num_assets)
    best_fitness_history = []
    mean_fitness_history = []

    # Run the algorithm for a specified number of generations
    for generation in range(num_generations):
        # Generate offspring by mutating each parent
        offspring = [mutate(parent, mutation_strength) for parent in population]
        offspring = np.array(offspring)

        # Combine parents and offspring into one population
        combined_population = np.vstack((population, offspring))

        # Evaluate the fitness of the combined population
        fitness = evaluate_population(combined_population, mean_returns)

        # Select the best μ individuals to form the next generation
        indices = np.argsort(fitness)[-μ:]  # Indices of top μ fitness values
        population = combined_population[indices]

        # Record the best and mean fitness of the generation
        best_fitness = fitness[indices[-1]]
        mean_fitness = np.mean(fitness)
        best_fitness_history.append(best_fitness)
        mean_fitness_history.append(mean_fitness)

    # After the final generation, select the best individual
    final_fitness = evaluate_population(population, mean_returns)
    best_index = np.argmax(final_fitness)
    best_weights = population[best_index]
    best_return = final_fitness[best_index]

    # Calculate portfolio variance for the best individual
    portfolio_variance = calculate_portfolio_variance(best_weights, covariance_matrix)

    # Calculate mean fitness of the final generation
    mean_fitness_final_gen = np.mean(final_fitness)

    return best_weights, best_return, portfolio_variance, best_fitness, mean_fitness_final_gen

if __name__ == "__main__":
    num_runs = 30
    results = []

    num_assets = len(mean_returns)

    # Execute the algorithm for a specified number of runs
    for run in range(1, num_runs + 1):
        start_time = time.time()

        # Run the Evolution Strategies algorithm
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
            best_fitness,
            mean_fitness,
            expected_return,
            portfolio_variance,
            best_weights.tolist(),
            training_time
        ])

        print(f"Run {run}/{num_runs} completed. Best Fitness: {best_fitness:.6f}")

    # Save all results to a CSV file
    csv_file_name = os.path.join(results_dir, 'es_basic_results.csv')
    with open(csv_file_name, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write the CSV header
        writer.writerow([
            'Run', 'Best Fitness', 'Mean Fitness', 'Expected Return', 'Portfolio Variance',
            'Weights', 'Training Time'
        ])
        # Write the data rows for each run
        for result in results:
            writer.writerow(result)

    print(f"\nAll runs completed. Results saved to '{csv_file_name}'.")
