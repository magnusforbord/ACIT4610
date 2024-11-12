from src.utils.data_loader import load_solomon_instance
from src.utils.distance_util import create_distance_matrix, create_time_matrix
from src.algorithms.pso.pso_optimizer import PSOOptimizer
import matplotlib.pyplot as plt
import numpy as np

def check_customer_coverage(solution_routes, num_customers):
    """Check which customers are visited and which are missing."""
    visited = set()
    for route in solution_routes:
        for customer in route:
            visited.add(customer)
    
    all_customers = set(range(1, num_customers + 1))
    missing = all_customers - visited
    
    print(f"\nCustomer Coverage Analysis:")
    print(f"Total customers: {num_customers}")
    print(f"Customers visited: {len(visited)}")
    print(f"Customers missing: {len(missing)}")
    if missing:
        print(f"Missing customer IDs: {sorted(list(missing))}")
    
    return visited, missing

def main():
    # Load problem
    problem = load_solomon_instance('data/c101.txt')
    print(f"Loaded problem with {len(problem.customers)} customers")
    
    # Create matrices
    dist_matrix = create_distance_matrix(problem.customers, problem.depot)
    time_matrix = create_time_matrix(problem, dist_matrix)
    
    # Initialize and run PSO with adjusted parameters
    pso = PSOOptimizer(
        problem=problem,
        distance_matrix=dist_matrix,
        time_matrix=time_matrix,
        n_particles=100,  # More particles
        w=0.9,     # Higher initial inertia for exploration
        c1=2.0,    # Balanced cognitive parameter
        c2=2.0     # Balanced social parameter
    )

    solution = pso.optimize(max_iterations=20)  # More iterations
        
    print("\nResults:")
    print(f"Feasible: {solution.feasible}")
    print(f"Total distance: {solution.total_distance:.2f}")
    print(f"Number of routes: {len(solution.routes)}")
    
    # Check customer coverage
    visited, missing = check_customer_coverage(solution.routes, len(problem.customers))
    
    # Print route details
    print("\nRoute Details:")
    for i, route in enumerate(solution.routes):
        total_demand = sum(problem.customers[c-1].demand for c in route)
        print(f"Route {i+1}: {len(route)} customers, Total demand: {total_demand}")
        print(f"Customers: {route}")
    
    # Visualization
    depot = problem.depot
    customers = problem.customers
    
    plt.figure(figsize=(12, 8))
    
    # Plot all customers
    plt.scatter([c.x for c in customers], [c.y for c in customers], 
                c='lightgray', marker='o', s=50, label='Unvisited Customers')
    
    # Plot visited customers in blue
    visited_customers = [customers[i-1] for i in visited]
    if visited_customers:
        plt.scatter([c.x for c in visited_customers], [c.y for c in visited_customers], 
                    c='blue', marker='o', s=50, label='Visited Customers')
    
    # Plot depot
    plt.scatter(depot.x, depot.y, c='red', marker='s', s=100, label='Depot')
    
    # Plot routes
    colors = plt.cm.get_cmap('tab10', len(solution.routes))
    for i, route in enumerate(solution.routes):
        route_points = [(depot.x, depot.y)] + [(customers[c-1].x, customers[c-1].y) for c in route] + [(depot.x, depot.y)]
        x, y = zip(*route_points)
        plt.plot(x, y, color=colors(i), linestyle='-', linewidth=2, label=f'Route {i + 1}')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('PSO Vehicle Routes')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()