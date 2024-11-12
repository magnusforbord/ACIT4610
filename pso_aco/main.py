from src.utils.data_loader import load_solomon_instance
from src.utils.distance_util import create_distance_matrix, create_time_matrix
from algorithms.aco.aco_optimizer import ACOOptimizer
import matplotlib.pyplot as plt
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
        alpha=1.0, #Importance of pheromones
        beta=5.0,   #Importance of heuristic
        rho=0.1     #Pheromone evaporation rate
    )
    
    solution = aco.optimize(max_iterations=50) #Number of iterations 
    
    print("\nResults:")
    print(f"Feasible: {solution.feasible}")
    print(f"Total distance: {solution.total_distance:.2f}")
    print(f"Number of routes: {len(solution.routes)}")
    
    if solution.routes:
        print(f"First route: {solution.routes[0][:10]}...")
    else:
        print("No feasible routes found!")
        

    print("\nProblem constraints:")
    print(f"Vehicle capacity: {problem.capacity}")
    print(f"Depot time window: {problem.depot.ready_time} - {problem.depot.due_time}")
    print(f"Number of vehicles available: {problem.vehicles}")

    depot = problem.depot
    customers = problem.customers
    plt.scatter(depot.x, depot.y, c='red', marker='s', s=100, label='Depot')
    plt.scatter([c.x for c in customers], [c.y for c in customers], c='blue', marker='o', s=50, label='Customers')

    colors = plt.cm.get_cmap('tab10', len(solution.routes))
    for i, route in enumerate(solution.routes):
        route_points = [(depot.x, depot.y)] + [(customers[c-1].x, customers[c-1].y) for c in route] + [(depot.x, depot.y)]
        x, y = zip(*route_points)
        plt.plot(x, y, color=colors(i), linestyle='-', linewidth=2, label=f'Vehicle {i + 1}')
        plt.pause(1)  # Pause to show each route being added sequentially

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Vehicle Routes')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot the pheromone strength as lines between nodes
    pheromone = aco.colony.pheromone
    plt.figure(figsize=(10, 6))
    plt.scatter(depot.x, depot.y, c='red', marker='s', s=100, label='Depot')
    plt.scatter([c.x for c in customers], [c.y for c in customers], c='blue', marker='o', s=50, label='Customers')

    max_pheromone = np.max(pheromone)
    min_pheromone = np.min(pheromone)
    for i in range(len(customers) + 1):
        for j in range(i + 1, len(customers) + 1):
            pheromone_strength = pheromone[i][j]
            if pheromone_strength > min_pheromone:  # Only plot significant pheromone levels
                x_coords = [depot.x if i == 0 else customers[i-1].x, depot.x if j == 0 else customers[j-1].x]
                y_coords = [depot.y if i == 0 else customers[i-1].y, depot.y if j == 0 else customers[j-1].y]
                line_width = 1 + 4 * (pheromone_strength - min_pheromone) / (max_pheromone - min_pheromone)
                plt.plot(x_coords, y_coords, 'k-', alpha=0.6, linewidth=line_width)

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Pheromone Strength Between Nodes')
    plt.grid(True)
    plt.show()

    # Explanation of the pheromone plot
    print("\nPheromone Strength Heatmap Explanation:")
    print("The pheromone strength plot shows the pheromone levels as lines between nodes (customers and depot).")
    print("- Thicker lines represent higher pheromone levels, indicating that these routes have been more frequently used or are considered more optimal by the ants.")
    print("- Thinner lines represent lower pheromone levels, suggesting that these routes are less attractive.")
    print("This visualization helps understand which paths are favored by the algorithm over the course of optimization.")


if __name__ == "__main__":
    main()