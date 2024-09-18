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

def mutate(weights, mutation_rate):
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
    # Select individuals with the highest fitness
    indices = np.argsort(fitness)[-num_selected:]
    return population[indices]

def evolutionary_programming(mean_returns, num_assets, pop_size=50, num_generations=100, mutation_rate=0.1):
    # Initialize population
    population = initialize_population(pop_size, num_assets)
    
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
        
        # Optionally, print progress
        best_fitness = np.max(combined_fitness)
        print(f"Generation {generation+1}/{num_generations}, Best Expected Return: {best_fitness:.6f}")
    
    # After the final generation, return the best solution
    final_fitness = evaluate_population(population, mean_returns)
    best_index = np.argmax(final_fitness)
    best_weights = population[best_index]
    best_return = final_fitness[best_index]
    return best_weights, best_return

if __name__ == "__main__":
    num_assets = len(mean_returns)
    best_weights, best_return = evolutionary_programming(
        mean_returns,
        num_assets,
        pop_size=50,
        num_generations=100,
        mutation_rate=0.1
    )
    
    print("\nOptimal Portfolio Weights:")
    for ticker, weight in zip(mean_returns.index, best_weights):
        print(f"{ticker}: {weight:.4f}")
    print(f"\nExpected Portfolio Return: {best_return:.6f}")
