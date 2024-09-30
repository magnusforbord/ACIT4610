import numpy as np
import pandas as pd
import os
import time

# Determine the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data', 'processed')

# Load monthly returns
monthly_returns = pd.read_csv(os.path.join(data_dir, 'monthly_returns.csv'), index_col=0)
mean_returns = monthly_returns.mean()

def objective_function(weights, mean_returns):
    """
    Calculate the expected return of the portfolio.

    Parameters:
    - weights (numpy.ndarray): Portfolio weights.
    - mean_returns (pandas.Series): Mean returns for each asset.

    Returns:
    - float: Expected portfolio return.
    """
    return np.dot(weights, mean_returns)

def initialize_population(pop_size, num_assets):
    """
    Initialize the population with random weights summing to 1.

    Parameters:
    - pop_size (int): Population size.
    - num_assets (int): Number of assets.

    Returns:
    - numpy.ndarray: Initialized population.
    """
    population = []
    for _ in range(pop_size):
        weights = np.random.dirichlet(np.ones(num_assets), size=1)[0]
        population.append(weights)
    return np.array(population)

def evaluate_population(population, mean_returns):
    """
    Evaluate the fitness of each individual in the population.

    Parameters:
    - population (numpy.ndarray): Array of individuals (portfolio weights).
    - mean_returns (pandas.Series): Mean returns for each asset.

    Returns:
    - numpy.ndarray: Array of fitness values for each individual.
    """
    fitness = []
    for weights in population:
        expected_return = objective_function(weights, mean_returns)
        fitness.append(expected_return)
    return np.array(fitness)

def mutate(weights, mutation_rate):
    """
    Mutate the portfolio weights.

    Parameters:
    - weights (numpy.ndarray): Current portfolio weights.
    - mutation_rate (float): Mutation rate.

    Returns:
    - numpy.ndarray: Mutated portfolio weights.
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

    Parameters:
    - population (numpy.ndarray): Current population.
    - fitness (numpy.ndarray): Fitness values for the population.
    - num_selected (int): Number of individuals to select.

    Returns:
    - numpy.ndarray: Selected population.
    """
    indices = np.argsort(fitness)[-num_selected:]
    return population[indices]

def evolutionary_programming(mean_returns, num_assets, pop_size=50, num_generations=100, mutation_rate=0.1):
    """
    Basic Evolutionary Programming algorithm for portfolio optimization.

    Parameters:
    - mean_returns (pandas.Series): Mean returns for each asset.
    - num_assets (int): Number of assets.
    - pop_size (int): Population size.
    - num_generations (int): Number of generations.
    - mutation_rate (float): Mutation rate.

    Returns:
    - tuple: Best weights, best return, fitness history.
    """
    # Initialize population
    population = initialize_population(pop_size, num_assets)
    fitness_history = []

    for generation in range(num_generations):
        # Evaluate fitness
        fitness = evaluate_population(population, mean_returns)

        # Generate offspring through mutation
        offspring = []
        for individual in population:
            mutated_individual = mutate(individual, mutation_rate)
            offspring.append(mutated_individual)
        offspring = np.array(offspring)

        # Evaluate fitness of offspring
        offspring_fitness = evaluate_population(offspring, mean_returns)

        # Combine parents and offspring
        combined_population = np.vstack((population, offspring))
        combined_fitness = np.concatenate((fitness, offspring_fitness))

        # Select the next generation
        population = select_population(combined_population, combined_fitness, pop_size)

        # Record the best fitness
        best_fitness = np.max(combined_fitness)
        fitness_history.append(best_fitness)

    # After the final generation, return the best solution
    final_fitness = evaluate_population(population, mean_returns)
    best_index = np.argmax(final_fitness)
    best_weights = population[best_index]
    best_return = final_fitness[best_index]
    return best_weights, best_return, fitness_history

if __name__ == "__main__":
    num_assets = len(mean_returns)
    num_runs = 30  # Number of runs
    run_results = []

    for run in range(1, num_runs + 1):
        start_time = time.time()
        best_weights, best_return, fitness_history = evolutionary_programming(
            mean_returns,
            num_assets,
            pop_size=50,
            num_generations=100,
            mutation_rate=0.1
        )
        end_time = time.time()
        execution_time = end_time - start_time

        # Store the results
        run_results.append({
            'Run': run,
            'Best_Return': best_return,
            'Best_Weights': best_weights,
            'Execution_Time': execution_time,
            'Fitness_History': fitness_history
        })

        print(f"Run {run}/{num_runs} completed. Best Return: {best_return:.6f}")

    # Save results to CSV
    results_df = pd.DataFrame(run_results)
    # Convert numpy arrays to strings for CSV storage
    results_df['Best_Weights'] = results_df['Best_Weights'].apply(lambda x: ','.join(map(str, x)))
    results_df['Fitness_History'] = results_df['Fitness_History'].apply(lambda x: ','.join(map(str, x)))
    results_df.to_csv('ep_basic_results.csv', index=False)

    print("\nAll runs completed. Results saved to 'ep_basic_results.csv'.")
