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

# Load monthly returns and calculate mean returns and covariance matrix
monthly_returns = pd.read_csv(os.path.join(data_dir, 'monthly_returns.csv'), index_col=0)
mean_returns = monthly_returns.mean().values
covariance_matrix = pd.read_csv(os.path.join(data_dir, 'covariance_matrix.csv'), index_col=0).values

def fitness_function(weights, mean_returns):
    """
    Calculate the expected return of the portfolio.
    """
    expected_return = np.dot(weights, mean_returns)
    return expected_return

def initialize_population(pop_size, num_assets):
    """
    Initialize the population with random weights that sum to 1 and random mutation rates.
    """
    population = []
    for _ in range(pop_size):
        weights = np.random.dirichlet(np.ones(num_assets))
        mutation_rates = np.random.uniform(0.01, 0.1, num_assets)
        individual = {'weights': weights, 'mutation_rates': mutation_rates}
        population.append(individual)
    return population

def mutate(individual, tau, tau_prime):
    """
    Mutate the portfolio weights and mutation rates using self-adaptive mutation.
    """
    num_assets = len(individual['weights'])
    mutation_rates = individual['mutation_rates']

    # Apply global and local mutations to mutation rates
    global_mutation = np.random.normal(0, tau_prime)
    local_mutation = np.random.normal(0, tau, num_assets)
    new_mutation_rates = mutation_rates * np.exp(global_mutation + local_mutation)

    # Ensure mutation rates are within reasonable bounds
    new_mutation_rates = np.clip(new_mutation_rates, 1e-5, 0.5)

    # Mutate weights using the updated mutation rates
    weights = individual['weights']
    weight_mutation = np.random.normal(0, new_mutation_rates)
    new_weights = weights + weight_mutation

    # Ensure weights are non-negative and normalize them to sum to 1
    new_weights = np.clip(new_weights, 0, 1)
    new_weights /= new_weights.sum()

    # Update individual with new weights and mutation rates
    individual['weights'] = new_weights
    individual['mutation_rates'] = new_mutation_rates
    return individual

def tournament_selection(population, fitnesses, tournament_size):
    """
    Select individuals based on a tournament selection strategy.
    """
    selected = []
    pop_size = len(population)
    for _ in range(pop_size):
        participants = np.random.choice(pop_size, tournament_size, replace=False)
        best = max(participants, key=lambda idx: fitnesses[idx])
        selected.append(population[best])
    return selected

def elitism(population, fitnesses, num_elites):
    """
    Retain the top-performing individuals (elites) based on fitness.
    """
    sorted_indices = np.argsort(fitnesses)[::-1]
    elites = [population[i] for i in sorted_indices[:num_elites]]
    return elites

def evolutionary_programming(mean_returns, num_assets, pop_size=50, num_generations=100, tau=None, tau_prime=None, tournament_size=3, num_elites=2):
    """
    Evolutionary Programming algorithm for portfolio optimization with self-adaptive mutation.
    """
    # Set mutation parameters if not provided
    if tau is None:
        tau = 1 / np.sqrt(2 * np.sqrt(num_assets))
    if tau_prime is None:
        tau_prime = 1 / np.sqrt(2 * num_assets)

    # Initialize population
    population = initialize_population(pop_size, num_assets)
    best_fitness_history = []
    mean_fitness_history = []

    # Evolution process over generations
    for generation in range(1, num_generations + 1):
        # Calculate fitness for each individual
        fitnesses = np.array([fitness_function(ind['weights'], mean_returns) for ind in population])

        # Record best and mean fitness for the generation
        best_fitness = np.max(fitnesses)
        mean_fitness = np.mean(fitnesses)
        best_fitness_history.append(best_fitness)
        mean_fitness_history.append(mean_fitness)

        # Perform elitism to retain top individuals
        elites = elitism(population, fitnesses, num_elites)

        # Select individuals for the next generation through tournament selection
        selected_population = tournament_selection(population, fitnesses, tournament_size)

        # Apply mutation to create offspring
        offspring = [mutate(ind.copy(), tau, tau_prime) for ind in selected_population]

        # Form the new population with elites and offspring
        population = elites + offspring[:pop_size - num_elites]

    # After all generations, identify the best individual
    final_fitnesses = np.array([fitness_function(ind['weights'], mean_returns) for ind in population])
    best_index = np.argmax(final_fitnesses)
    best_individual = population[best_index]
    best_fitness = final_fitnesses[best_index]
    mean_fitness_final_gen = np.mean(final_fitnesses)

    return best_individual, best_fitness, mean_fitness_final_gen

if __name__ == "__main__":
    # Define the number of runs and results storage
    num_runs = 30
    results = []

    num_assets = len(mean_returns)

    # Perform multiple runs of the evolutionary programming algorithm
    for run in range(1, num_runs + 1):
        start_time = time.time()

        # Run the algorithm and capture the best individual, fitness, and mean fitness
        best_individual, best_fitness, mean_fitness = evolutionary_programming(
            mean_returns, num_assets,
            pop_size=50, num_generations=100,
            tournament_size=3, num_elites=2
        )

        end_time = time.time()
        training_time = end_time - start_time

        # Extract the weights and calculate expected return and portfolio variance
        weights = best_individual['weights']
        expected_return = np.dot(weights, mean_returns)
        portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))

        # Store the results for this run
        results.append([
            run,
            best_fitness,
            mean_fitness,
            expected_return,
            portfolio_variance,
            weights.tolist(),
            training_time
        ])

        print(f"Run {run}/{num_runs} completed. Best Fitness: {best_fitness:.6f}")

    # Save all results to a CSV file
    csv_file_name = os.path.join(results_dir, 'ep_advanced_results.csv')
    with open(csv_file_name, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write header
        writer.writerow([
            'Run', 'Best Fitness', 'Mean Fitness', 'Expected Return', 'Portfolio Variance',
            'Weights', 'Training Time'
        ])
        # Write each row of results
        for result in results:
            writer.writerow(result)

    print(f"\nAll runs completed. Results saved to '{csv_file_name}'.")
