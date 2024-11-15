# main_pso.py
from src.utils.data_loader import load_solomon_instance
from src.utils.distance_util import create_distance_matrix, create_time_matrix
from src.algorithms.pso.pso_optimizer import PSOOptimizer
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Load problem
    problem = load_solomon_instance('data/c101.txt')
    print(f"Loaded problem with {len(problem.customers)} customers")
    
    # Create matrices
    dist_matrix = create_distance_matrix(problem.customers, problem.depot)
    time_matrix = create_time_matrix(problem, dist_matrix)
    
    # Initialize and run PSO
    pso = PSOOptimizer(
        problem=problem,
        distance_matrix=dist_matrix,
        time_matrix=time_matrix,
        n_particles=200, #Number of particles
        w=0.9,    #Initial intertia weight
        c1=2.5,   #Initial cognitive component value
        c2=1.5    #Initial social component value
    )
    
    solution, times, distances = pso.optimize(max_iterations=50)
    

    # Print final statistics
    print("\nFinal Statistics:")
    print(f"Total Distance: {solution.total_distance:.2f}")
    print(f"Number of Vehicles Used: {len(solution.routes)}")
    print(f"Average Route Length: {solution.total_distance/len(solution.routes):.2f}")
    
    # Print individual route details
    print("\nRoute Details:")
    for i, route in enumerate(solution.routes):
        route_distance = pso.calculate_route_distance(route)
        route_load = sum(problem.customers[c-1].demand for c in route)
        print(f"\nRoute {i+1}:")
        print(f"Distance: {route_distance:.2f}")
        print(f"Load: {route_load}/{problem.capacity}")
        print(f"Customers: {len(route)}")

if __name__ == "__main__":
    main()