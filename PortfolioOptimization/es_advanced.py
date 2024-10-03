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

def initialize_population(mu, num_assets):
    population = []
    for _ in range(mu):
        # Random weights summing to 1
        weights = np.random.dirichlet(np.ones(num_assets))
        # Initial mutation strengths for each weight
        sigma = np.random.uniform(0.05, 0.2, num_assets)
        individual = {'weights': weights, 'sigma': sigma}
        population.append(individual)
    return population

def recombine(parent1, parent2):
    num_assets = len(parent1['weights'])
    # Intermediate recombination (averaging)
    child_weights = (parent1['weights'] + parent2['weights']) / 2
    child_sigma = (parent1['sigma'] + parent2['sigma']) / 2
    # Ensure weights sum to 1
    child_weights /= np.sum(child_weights)
    return {'weights': child_weights, 'sigma': child_sigma}

def mutate(individual, tau, tau_prime):
    num_assets = len(individual['weights'])
    # Update mutation strengths
    sigma = individual['sigma']
    global_mutation = np.random.normal(0, tau_prime)
    local_mutation = np.random.normal(0, tau, num_assets)
    new_sigma = sigma * np.exp(global_mutation + local_mutation)
    # Ensure sigma is within bounds
    new_sigma = np.clip(new_sigma, 1e-5, 0.5)
    # Mutate weights
    weights = individual['weights']
    weight_mutation = np.random.normal(0, new_sigma)
    new_weights = weights + weight_mutation
    # Ensure weights are non-negative and sum to 1
    new_weights = np.clip(new_weights, 0, None)
    new_weights /= np.sum(new_weights)
    # Update individual
    individual['weights'] = new_weights
    individual['sigma'] = new_sigma
    return individual

def select_population(population, fitnesses, mu):
    # Sort individuals based on fitness
    sorted_indices = np.argsort(fitnesses)[::-1]
    selected = [population[i] for i in sorted_indices[:mu]]
    return selected

def evolution_strategies(mean_returns, covariance_matrix, num_assets, mu=20, lambda_=80, num_generations=100, risk_aversion=3):
    # Learning rates for mutation strengths
    tau = 1 / np.sqrt(2 * np.sqrt(num_assets))
    tau_prime = 1 / np.sqrt(2 * num_assets)
    # Initialize population
    population = initialize_population(mu, num_assets)
    best_fitness_history = []
    for generation in range(1, num_generations + 1):
        # Generate offspring
        offspring = []
        for _ in range(lambda_):
            # Select two parents randomly
            parents = np.random.choice(population, 2, replace=False)
            # Recombine to create a child
            child = recombine(parents[0], parents[1])
            # Mutate the child
            child = mutate(child, tau, tau_prime)
            offspring.append(child)
        # Combine parents and offspring
        combined_population = population + offspring
        # Evaluate fitness
        fitnesses = []
        for individual in combined_population:
            fitness = fitness_function(individual['weights'], mean_returns, covariance_matrix, risk_aversion)
            fitnesses.append(fitness)
        fitnesses = np.array(fitnesses)
        # Select the next generation
        population = select_population(combined_population, fitnesses, mu)
        # Record best fitness
        best_fitness = np.max(fitnesses)
        best_fitness_history.append(best_fitness)
        # Optionally, print progress
        # print(f"Generation {generation}/{num_generations}, Best Fitness: {best_fitness:.6f}")
    # After evolution, select the best individual
    final_fitnesses = []
    for individual in population:
        fitness = fitness_function(individual['weights'], mean_returns, covariance_matrix, risk_aversion)
        final_fitnesses.append(fitness)
    best_index = np.argmax(final_fitnesses)
    best_individual = population[best_index]
    return best_individual, best_fitness_history

if __name__ == "__main__":
    num_runs = 30  # Number of runs
    results = []   # List to store results from each run

    num_assets = len(mean_returns)
    mu = 20       # Number of parents
    lambda_ = 80  # Number of offspring
    num_generations = 100
    risk_aversion = 3

    for run in range(1, num_runs + 1):
        start_time = time.time()

        best_individual, fitness_history = evolution_strategies(
            mean_returns.values, covariance_matrix, num_assets,
            mu=mu, lambda_=lambda_, num_generations=num_generations, risk_aversion=risk_aversion
        )

        end_time = time.time()
        training_time = end_time - start_time

        # Collect the results
        weights = best_individual['weights']
        expected_return = np.dot(weights, mean_returns.values)
        portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
        best_fitness = fitness_history[-1]

        # Append results to the list
        results.append([
            run,
            best_fitness,
            expected_return,
            portfolio_variance,
            weights.tolist(),        # Convert numpy array to list for CSV
            fitness_history,         # List of best fitness values over generations
            training_time
        ])

        print(f"Run {run}/{num_runs} completed. Best Fitness: {best_fitness:.6f}")

    # Save results to CSV
    csv_file_name = os.path.join(results_dir, 'es_advanced_results.csv')
    with open(csv_file_name, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write header
        writer.writerow([
            'Run', 'Best Fitness', 'Expected Return', 'Portfolio Variance',
            'Weights', 'Fitness History', 'Training Time'
        ])
        # Write data rows
        for result in results:
            writer.writerow(result)

    print(f"\nAll runs completed. Results saved to '{csv_file_name}'.")
