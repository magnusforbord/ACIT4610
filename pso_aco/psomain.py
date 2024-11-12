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
        n_particles=200,
        w=0.9,
        c1=2,
        c2=2
    )
    
    solution = pso.optimize(max_iterations=100)
    
    # Plot solution
    depot = problem.depot
    customers = problem.customers
    plt.figure(figsize=(10, 6))
    plt.scatter(depot.x, depot.y, c='red', marker='s', s=100, label='Depot')
    plt.scatter([c.x for c in customers], [c.y for c in customers], 
                c='blue', marker='o', s=50, label='Customers')
    
    if solution.routes:
        cmap = plt.colormaps.get_cmap('tab10')
        colors = [cmap(i / len(solution.routes)) for i in range(len(solution.routes))]
        for i, route in enumerate(solution.routes):
            route_points = [(depot.x, depot.y)] + \
                        [(customers[c-1].x, customers[c-1].y) for c in route] + \
                        [(depot.x, depot.y)]
            x, y = zip(*route_points)
            plt.plot(x, y, color=colors[i], linestyle='-', linewidth=2, 
                    label=f'Vehicle {i+1}')
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('PSO Solution - Vehicle Routes')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()