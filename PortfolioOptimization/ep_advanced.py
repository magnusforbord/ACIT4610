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

def fitness_function(weights, mean_returns, covariance_matrix, risk_aversion):
    expected_return = np.dot(weights, mean_returns)
    portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
    fitness = expected_return - risk_aversion * portfolio_variance
    return fitness

def initialize_population(pop_size, num_assets):
    population = []
    for _ in range(pop_size):
        weights = np.random.dirichlet(np.ones(num_assets))
        mutation_rates = np.random.uniform(0.01, 0.1, num_assets)
        individual = {'weights': weights, 'mutation_rates': mutation_rates}
        population.append(individual)
    return population

def mutate(individual, tau, tau_prime):
    num_assets = len(individual['weights'])
    mutation_rates = individual['mutation_rates']
    global_mutation = np.random.normal(0, tau_prime)
    local_mutation = np.random.normal(0, tau, num_assets)
    new_mutation_rates = mutation_rates * np.exp(global_mutation + local_mutation)
    new_mutation_rates = np.clip(new_mutation_rates, 1e-5, 0.5)
    weights = individual['weights']
    weight_mutation = np.random.normal(0, new_mutation_rates)
    new_weights = weights + weight_mutation
    new_weights = np.clip(new_weights, 0, 1)
    new_weights /= new_weights.sum()
    individual['weights'] = new_weights
    individual['mutation_rates'] = new_mutation_rates
    return individual

def tournament_selection(population, fitnesses, tournament_size):
    selected = []
    pop_size = len(population)
    for _ in range(pop_size):
        participants = np.random.choice(pop_size, tournament_size, replace=False)
        best = max(participants, key=lambda idx: fitnesses[idx])
        selected.append(population[best])
    return selected

def elitism(population, fitnesses, num_elites):
    sorted_indices = np.argsort(fitnesses)[::-1]
    elites = [population[i] for i in sorted_indices[:num_elites]]
    return elites

def evolutionary_programming(mean_returns, covariance_matrix, num_assets, pop_size=50, num_generations=100, risk_aversion=3, tau=None, tau_prime=None, tournament_size=3, num_elites=2):
    if tau is None:
        tau = 1 / np.sqrt(2 * np.sqrt(num_assets))
    if tau_prime is None:
        tau_prime = 1 / np.sqrt(2 * num_assets)
    population = initialize_population(pop_size, num_assets)
    best_fitness_history = []

    for generation in range(1, num_generations + 1):
        fitnesses = np.array([fitness_function(ind['weights'], mean_returns, covariance_matrix, risk_aversion) for ind in population])
        elites = elitism(population, fitnesses, num_elites)
        selected_population = tournament_selection(population, fitnesses, tournament_size)
        offspring = [mutate(ind.copy(), tau, tau_prime) for ind in selected_population]
        population = elites + offspring[:pop_size - num_elites]
        best_fitness = np.max(fitnesses)
        best_fitness_history.append(best_fitness)

    mean_fitness = np.mean(best_fitness_history)
    final_fitnesses = [fitness_function(ind['weights'], mean_returns, covariance_matrix, risk_aversion) for ind in population]
    best_individual = population[np.argmax(final_fitnesses)]
    return best_individual, mean_fitness

if __name__ == "__main__":
    num_runs = 30
    results = []

    num_assets = len(mean_returns)

    for run in range(1, num_runs + 1):
        start_time = time.time()

        best_individual, mean_fitness = evolutionary_programming(
            mean_returns.values, covariance_matrix, num_assets, 
            pop_size=50, num_generations=100, risk_aversion=3, 
            tournament_size=3, num_elites=2
        )

        end_time = time.time()
        training_time = end_time - start_time

        weights = best_individual['weights']
        expected_return = np.dot(weights, mean_returns.values)
        portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))

        results.append([
            run,
            mean_fitness,
            expected_return,
            portfolio_variance,
            weights.tolist(),
            training_time
        ])

        print(f"Run {run}/{num_runs} completed. Mean Fitness: {mean_fitness:.6f}")

    csv_file_name = os.path.join(results_dir, 'ep_advanced_results.csv')
    with open(csv_file_name, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            'Run', 'Mean Fitness', 'Expected Return', 'Portfolio Variance', 
            'Weights', 'Training Time'
        ])
        for result in results:
            writer.writerow(result)

    print(f"\nAll runs completed. Results saved to '{csv_file_name}'.")
