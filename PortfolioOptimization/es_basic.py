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

def evolution_strategies(mean_returns, num_assets, pop_size=50, num_generations=100, mutation_strength=0.05):
    μ = pop_size // 2  # Number of parents selected
    λ = pop_size       # Number of offspring generated

    # Initialize population
    population = initialize_population(μ, num_assets)

    for generation in range(num_generations):
        # Generate offspring
        offspring = []
        for parent in population:
            child = mutate(parent, mutation_strength)
            offspring.append(child)
        offspring = np.array(offspring)

        # Combine parents and offspring
        combined_population = np.vstack((population, offspring))

        # Evaluate fitness
        fitness = evaluate_population(combined_population, mean_returns)

        # Select the best μ individuals
        indices = np.argsort(fitness)[-μ:]
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
    best_weights, best_return = evolution_strategies(
        mean_returns.values,
        num_assets,
        pop_size=50,
        num_generations=100,
        mutation_strength=0.05
    )

    print("\nOptimal Portfolio Weights:")
    for ticker, weight in zip(mean_returns.index, best_weights):
        print(f"{ticker}: {weight:.4f}")
    print(f"\nExpected Portfolio Return: {best_return:.6f}")
