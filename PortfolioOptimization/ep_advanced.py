import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Determine the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data', 'processed')

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
        # Random weights summing to 1
        weights = np.random.dirichlet(np.ones(num_assets))
        # Mutation rates for each weight
        mutation_rates = np.random.uniform(0.01, 0.1, num_assets)
        individual = {'weights': weights, 'mutation_rates': mutation_rates}
        population.append(individual)
    return population

def mutate(individual, tau, tau_prime):
    num_assets = len(individual['weights'])
    # Update mutation rates
    mutation_rates = individual['mutation_rates']
    global_mutation = np.random.normal(0, tau_prime)
    local_mutation = np.random.normal(0, tau, num_assets)
    new_mutation_rates = mutation_rates * np.exp(global_mutation + local_mutation)
    # Ensure mutation rates are within bounds
    new_mutation_rates = np.clip(new_mutation_rates, 1e-5, 0.5)
    # Mutate weights
    weights = individual['weights']
    weight_mutation = np.random.normal(0, new_mutation_rates)
    new_weights = weights + weight_mutation
    # Ensure weights are non-negative and sum to 1
    new_weights = np.clip(new_weights, 0, 1)
    new_weights /= new_weights.sum()
    # Update individual
    individual['weights'] = new_weights
    individual['mutation_rates'] = new_mutation_rates
    return individual

def tournament_selection(population, fitnesses, tournament_size):
    selected = []
    pop_size = len(population)
    for _ in range(pop_size):
        # Randomly select individuals for the tournament
        participants = np.random.choice(pop_size, tournament_size, replace=False)
        # Select the best among them
        best = participants[0]
        for participant in participants[1:]:
            if fitnesses[participant] > fitnesses[best]:
                best = participant
        selected.append(population[best])
    return selected

def elitism(population, fitnesses, num_elites):
    # Sort individuals based on fitness
    sorted_indices = np.argsort(fitnesses)[::-1]
    elites = [population[i] for i in sorted_indices[:num_elites]]
    return elites

def evolutionary_programming(mean_returns, covariance_matrix, num_assets, pop_size=50, num_generations=100, risk_aversion=0.1, tau=None, tau_prime=None, tournament_size=3, num_elites=2):
    if tau is None:
        tau = 1 / np.sqrt(2 * np.sqrt(num_assets))
    if tau_prime is None:
        tau_prime = 1 / np.sqrt(2 * num_assets)
    # Initialize population
    population = initialize_population(pop_size, num_assets)
    best_fitness_history = []
    for generation in range(1, num_generations + 1):
        # Evaluate fitness
        fitnesses = []
        for individual in population:
            fitness = fitness_function(individual['weights'], mean_returns, covariance_matrix, risk_aversion)
            fitnesses.append(fitness)
        fitnesses = np.array(fitnesses)
        # Elitism
        elites = elitism(population, fitnesses, num_elites)
        # Selection
        selected_population = tournament_selection(population, fitnesses, tournament_size)
        # Mutation
        offspring = []
        for individual in selected_population:
            mutated_individual = mutate(individual.copy(), tau, tau_prime)
            offspring.append(mutated_individual)
        # Create new population
        population = elites + offspring[:pop_size - num_elites]
        # Record best fitness
        best_fitness = np.max(fitnesses)
        best_fitness_history.append(best_fitness)
        print(f"Generation {generation}/{num_generations}, Best Fitness: {best_fitness:.6f}")
    # After evolution, select the best individual
    final_fitnesses = []
    for individual in population:
        fitness = fitness_function(individual['weights'], mean_returns, covariance_matrix, risk_aversion)
        final_fitnesses.append(fitness)
    best_index = np.argmax(final_fitnesses)
    best_individual = population[best_index]
    return best_individual, best_fitness_history

if __name__ == "__main__":
    num_assets = len(mean_returns)
    best_individual, fitness_history = evolutionary_programming(
        mean_returns.values, covariance_matrix, num_assets, 
        pop_size=50, num_generations=100, risk_aversion=0.1, 
        tournament_size=3, num_elites=2
    )
    # Output the results
    weights = best_individual['weights']
    print("\nOptimal Portfolio Weights:")
    for ticker, weight in zip(mean_returns.index, weights):
        print(f"{ticker}: {weight:.4f}")
    expected_return = np.dot(weights, mean_returns.values)
    portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
    print(f"\nExpected Portfolio Return: {expected_return:.6f}")
    print(f"Portfolio Variance: {portfolio_variance:.6f}")