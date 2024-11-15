from src.utils.data_loader import load_solomon_instance
from src.utils.distance_util import create_distance_matrix, create_time_matrix
from src.algorithms.aco.aco_optimizer import ACOOptimizer
import numpy as np

def main():
    # Load problem
    
    problem = load_solomon_instance('data/c101.txt')
    print(f"Loaded problem with {len(problem.customers)} customers")
    
    # Create distance and time matrices
    dist_matrix = create_distance_matrix(problem.customers, problem.depot)
    time_matrix = create_time_matrix(problem, dist_matrix)
    
    # Initialize and run ACO
    aco = ACOOptimizer(
        problem=problem,
        distance_matrix=dist_matrix,
        time_matrix=time_matrix,
        n_ants=50,  
        alpha=1.0,  #Importance of pheromones
        beta=2.0,   #Importance of heuristic
        rho=0.1     #Pheromone evaporation rate
    )
    
    solution,distances, time = aco.optimize(max_iterations=50) #Number of iterations 
    
    print("\nResults:")
    print(f"Feasible: {solution.feasible}")
    print(f"Total distance: {solution.total_distance:.2f}")
    print(f"Number of routes: {len(solution.routes)}")
    
        
    print("\nProblem constraints:")
    print(f"Vehicle capacity: {problem.capacity}")
    print(f"Number of vehicles available: {problem.vehicles}")



if __name__ == "__main__":
    main()