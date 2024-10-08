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

def evolution_strategies_mu_plus_lambda(mean_returns, covariance_matrix, num_assets, mu=20, lambda_=80, num_generations=100, mutation_strength=0.05):
    # Initialize population of parents
    population = initialize_population(mu, num_assets)
    best_fitness_history = []
    mean_fitness_history = []

    for generation in range(num_generations):
        offspring = []
        for _ in range(lambda_):
            parent_idx = np.random.randint(0, mu)
            parent = population[parent_idx]
            child = mutate(parent, mutation_strength)
            offspring.append(child)
        offspring = np.array(offspring)

        combined_population = np.vstack((population, offspring))
        fitness = evaluate_population(combined_population, mean_returns)

        # Record best and mean fitness
        best_fitness = np.max(fitness)
        mean_fitness = np.mean(fitness)
        best_fitness_history.append(best_fitness)
        mean_fitness_history.append(mean_fitness)

        indices = np.argsort(fitness)[-mu:]
        population = combined_population[indices]

    mean_fitness_final_gen = mean_fitness_history[-1]
    final_fitness = evaluate_population(population, mean_returns)
    best_index = np.argmax(final_fitness)
    best_weights = population[best_index]
    best_return = final_fitness[best_index]

    return best_weights, best_return, best_fitness, mean_fitness_final_gen

if __name__ == "__main__":
    num_runs = 30
    results = []

    num_assets = len(mean_returns)
    mu = 20
    lambda_ = 80
    num_generations = 100
    mutation_strength = 0.05

    for run in range(1, num_runs + 1):
        start_time = time.time()

        best_weights, expected_return, best_fitness, mean_fitness = evolution_strategies_mu_plus_lambda(
            mean_returns,
            covariance_matrix,
            num_assets,
            mu=mu,
            lambda_=lambda_,
            num_generations=num_generations,
            mutation_strength=mutation_strength
        )

        end_time = time.time()
        training_time = end_time - start_time

        portfolio_variance = calculate_portfolio_variance(best_weights, covariance_matrix)

        results.append([
            run,
            best_fitness,
            mean_fitness,
            expected_return,
            portfolio_variance,
            best_weights.tolist(),  # Convert numpy array to list for CSV
            training_time
        ])

        print(f"Run {run}/{num_runs} completed. Best Fitness: {best_fitness:.6f}")

    csv_file_name = os.path.join(results_dir, 'es_mu_plus_lambda_results.csv')
    with open(csv_file_name, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            'Run', 'Best Fitness', 'Mean Fitness', 'Expected Return', 'Portfolio Variance',
            'Weights', 'Training Time'
        ])
        for result in results:
            writer.writerow(result)

    print(f"\nAll runs completed. Results saved to '{csv_file_name}'.")
