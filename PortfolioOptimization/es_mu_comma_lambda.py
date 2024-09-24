import numpy as np
import pandas as pd
import os

# Determine the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data', 'processed')

# Load monthly returns
monthly_returns = pd.read_csv(os.path.join(data_dir, 'monthly_returns.csv'), index_col=0)
mean_returns = monthly_returns.mean()

def objective_function(weights, mean_returns):
    return np.dot(weights, mean_returns)

def initialize_population(pop_size, num_assets):
    population = []
    for _ in range(pop_size):
        weights = np.random.dirichlet(np.ones(num_assets))
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

def evolution_strategies_mu_plus_lambda(mean_returns, num_assets, mu=20, lambda_=80, num_generations=100, mutation_strength=0.05):
    # Initialize population of parents
    population = initialize_population(mu, num_assets)

    for generation in range(num_generations):
        # Generate offspring
        offspring = []
        for _ in range(lambda_):
            # Select a parent randomly
            parent_idx = np.random.randint(0, mu)
            parent = population[parent_idx]
            # Mutate to create an offspring
            child = mutate(parent, mutation_strength)
            offspring.append(child)
        offspring = np.array(offspring)

        # Combine parents and offspring
        combined_population = np.vstack((population, offspring))

        # Evaluate fitness
        fitness = evaluate_population(combined_population, mean_returns)

        # Select the best μ individuals
        indices = np.argsort(fitness)[-mu:]
        population = combined_population[indices]

        # Optionally, print progress
        best_fitness = fitness[indices[-1]]
        print(f"Generation {generation+1}/{num_generations}, Best Expected Return: {best_fitness:.6f}")

    # After the final generation, return the best solution
    final_fitness = evaluate_population(population, mean_returns)
    best_index = np.argmax(final_fitness)
    best_weights = population[best_index]
    best_return = final_fitness[best_index]
    return best_weights, best_return

if __name__ == "__main__":
    num_assets = len(mean_returns)
    mu = 20       # Number of parents
    lambda_ = 80  # Number of offspring

    best_weights, best_return = evolution_strategies_mu_plus_lambda(
        mean_returns.values,
        num_assets,
        mu=mu,
        lambda_=lambda_,
        num_generations=100,
        mutation_strength=0.05
    )

    print("\nOptimal Portfolio Weights:")
    for ticker, weight in zip(mean_returns.index, best_weights):
        print(f"{ticker}: {weight:.4f}")
    print(f"\nExpected Portfolio Return: {best_return:.6f}")
