import numpy as np
import pandas as pd
import os
import time
import csv

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
    Initialize the population with random weights summing to 1.
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

def mutate(weights, mutation_rate):
    """
    Mutate the portfolio weights.
    """
    num_assets = len(weights)
    mutated_weights = weights.copy()
    for i in range(num_assets):
        if np.random.rand() < mutation_rate:
            mutated_weights[i] += np.random.normal(0, 0.1)
    # Ensure weights are positive and sum to 1
    mutated_weights = np.abs(mutated_weights)
    mutated_weights /= np.sum(mutated_weights)
    return mutated_weights

def select_population(population, fitness, num_selected):
    """
    Select the top individuals based on fitness.
    """
    indices = np.argsort(fitness)[-num_selected:]
    return population[indices]

def evolutionary_programming(mean_returns, covariance_matrix, num_assets, pop_size=50, num_generations=100, mutation_rate=0.1):
    """
    Basic Evolutionary Programming algorithm for portfolio optimization.
    """
    # Initialize population
    population = initialize_population(pop_size, num_assets)
    best_fitness_history = []
    mean_fitness_history = []

    for generation in range(num_generations):
        # Evaluate fitness
        fitness = evaluate_population(population, mean_returns)

        # Generate offspring through mutation
        offspring = [mutate(individual, mutation_rate) for individual in population]
        offspring = np.array(offspring)

        # Evaluate fitness of offspring
        offspring_fitness = evaluate_population(offspring, mean_returns)

        # Combine parents and offspring
        combined_population = np.vstack((population, offspring))
        combined_fitness = np.concatenate((fitness, offspring_fitness))

        # Select the next generation
        population = select_population(combined_population, combined_fitness, pop_size)

        # Record best and mean fitness for this generation
        best_fitness = np.max(combined_fitness)
        mean_fitness = np.mean(combined_fitness)
        best_fitness_history.append(best_fitness)
        mean_fitness_history.append(mean_fitness)

    # After the final generation, get the best individual
    final_fitness = evaluate_population(population, mean_returns)
    best_index = np.argmax(final_fitness)
    best_weights = population[best_index]
    best_return = final_fitness[best_index]
    # Calculate portfolio variance
    portfolio_variance = calculate_portfolio_variance(best_weights, covariance_matrix)
    # Mean fitness of the final generation
    mean_fitness_final_gen = np.mean(final_fitness)

    return best_weights, best_return, portfolio_variance, best_return, mean_fitness_final_gen

if __name__ == "__main__":
    num_assets = len(mean_returns)
    num_runs = 30  # Number of runs
    results = []

    for run in range(1, num_runs + 1):
        start_time = time.time()
        best_weights, best_fitness, portfolio_variance, expected_return, mean_fitness = evolutionary_programming(
            mean_returns,
            covariance_matrix,
            num_assets,
            pop_size=50,
            num_generations=100,
            mutation_rate=0.1
        )
        end_time = time.time()
        training_time = end_time - start_time

        # Append results to the list
        results.append([
            run,
            best_fitness,           # Best Fitness (best expected return)
            mean_fitness,           # Mean Fitness of final generation
            expected_return,        # Expected Return (same as best fitness in this case)
            portfolio_variance,     # Portfolio Variance
            best_weights.tolist(),  # Weights
            training_time           # Training Time
        ])

        print(f"Run {run}/{num_runs} completed. Best Fitness: {best_fitness:.6f}")

    # Save results to CSV
    csv_file_name = os.path.join(results_dir, 'ep_basic_results.csv')
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

    print("\nAll runs completed. Results saved to 'ep_basic_results.csv'.")
