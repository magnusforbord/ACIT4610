import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from aco import AntColonyOptimizer  # Adjust the import based on your project structure
from Data_parser import dataset_parser  # Ensure this points to your dataset parser

def test_all_customers_visited(routes, customers):
    # Extract all customer IDs from customers (excluding the depot)
    customer_ids = set(customer.customer_no for customer in customers)  
    
    # Collect all customer IDs visited in the routes
    visited_customers = set()
    for route in routes:
        visited_customers.update(route)  # Add all customers from each route
    
    # Remove the depot (0) from visited_customers since we're only interested in customer visits
    visited_customers.discard(0)
    
    # Check if all customers have been visited
    if customer_ids == visited_customers:
        print("Test passed: All customers are visited.")
    else:
        missing_customers = customer_ids - visited_customers
        print(f"Test failed: The following customers were not visited: {missing_customers}")


# Function to plot final routes in a static plot
def plot_final_routes(routes, customers, depot, title="Optimized Customer Routes"):
    plt.figure()
    cmap = plt.get_cmap('tab10')  # Use a colormap for different vehicles
    for idx, route in enumerate(routes):
        x_coords = [depot.x_coord if customer == 0 else customers[customer - 1].x_coord for customer in route]
        y_coords = [depot.y_coord if customer == 0 else customers[customer - 1].y_coord for customer in route]
        plt.plot(x_coords, y_coords, marker='o', color=cmap(idx / len(routes)), label=f'Vehicle {idx + 1}')
    
    # Mark the depot distinctly
    plt.plot(depot.x_coord, depot.y_coord, 'ks', markersize=10, label='Depot')

    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.legend()
    plt.show()

# Function to animate routes over iterations
def animate_routes(i, optimizer, customers, depot, ax):
    ax.clear()
    ax.set_xlim(min(customer.x_coord for customer in customers) - 10,
                max(customer.x_coord for customer in customers) + 10)
    ax.set_ylim(min(customer.y_coord for customer in customers) - 10,
                max(customer.y_coord for customer in customers) + 10)

    cmap = plt.get_cmap('tab10')
    
    # Plot the depot
    ax.plot(depot.x_coord, depot.y_coord, 'ks', markersize=10, label='Depot')
    
    # Get the current best routes to animate the final solution
    routes = optimizer.best_routes
    max_steps = sum(len(route) - 1 for route in routes)  # Total route segments across all vehicles
    step = min(i, max_steps - 1)  # Avoid overflow
    
    # Track the cumulative steps for each vehicle to know which route segment to animate
    cumulative_steps = 0
    for vehicle_id, route in enumerate(routes):
        color = cmap(vehicle_id / len(routes))
        
        for j in range(1, len(route)):  # Skip the first depot visit
            if cumulative_steps < step:
                # Draw completed segments for each vehicle up to the current step
                start_customer = route[j - 1]
                end_customer = route[j]
                x_coords = [
                    depot.x_coord if start_customer == 0 else customers[start_customer - 1].x_coord,
                    depot.x_coord if end_customer == 0 else customers[end_customer - 1].x_coord
                ]
                y_coords = [
                    depot.y_coord if start_customer == 0 else customers[start_customer - 1].y_coord,
                    depot.y_coord if end_customer == 0 else customers[end_customer - 1].y_coord
                ]
                ax.plot(x_coords, y_coords, marker='o', color=color, label=f'Vehicle {vehicle_id + 1}' if j == 1 else "")
                cumulative_steps += 1
            else:
                break

    # Set plot labels and title
    ax.set_title(f"Step-by-Step Route Construction (Step {step + 1})")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.grid(True)
    ax.legend(loc='upper right')
    return ax

# Function to animate pheromone trails over iterations
def animate_pheromones(i, optimizer, customers, depot, ax):
    ax.clear()  # Clear the previous plot to refresh it for the current frame
    ax.set_xlim(min(customer.x_coord for customer in customers) - 10,
                max(customer.x_coord for customer in customers) + 10)
    ax.set_ylim(min(customer.y_coord for customer in customers) - 10,
                max(customer.y_coord for customer in customers) + 10)
    
    # Get the pheromone matrix from the history at the current iteration
    pheromone_matrix = optimizer.pheromone_matrix_history[i]
    max_pheromone = np.max(pheromone_matrix)  # Get the maximum pheromone value for scaling

    # Plot the customers (nodes)
    ax.plot(depot.x_coord, depot.y_coord, 'ks', markersize=10, label='Depot')
    for customer in customers:
        ax.plot(customer.x_coord, customer.y_coord, 'ro', markersize=6)

    # Draw the pheromone trails between customers
    for j in range(len(customers) + 1):
        for k in range(j + 1, len(customers) + 1):  # Only draw edges once
            pheromone_level = pheromone_matrix[j][k]
            if pheromone_level > 0:
                line_width = (pheromone_level / max_pheromone) * 5  # Adjust the scaling factor as needed
                x_coords = [
                    depot.x_coord if j == 0 else customers[j - 1].x_coord,
                    depot.x_coord if k == 0 else customers[k - 1].x_coord
                ]
                y_coords = [
                    depot.y_coord if j == 0 else customers[j - 1].y_coord,
                    depot.y_coord if k == 0 else customers[k - 1].y_coord
                ]
                ax.plot(x_coords, y_coords, color='blue', linewidth=line_width)

    ax.set_title(f"Pheromone Trails at Iteration: {i + 1}")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.grid(True)

    return ax

def main():
    # Parse the dataset
    vehicles, customers, depot = dataset_parser('../dataset/c101.txt')
    print("dataset found")
    # Initialize the Ant Colony Optimizer
    optimizer = AntColonyOptimizer(vehicles, customers, depot, num_ants=20, num_iterations=100)
    print("starting optimization")
    # Run the optimization
    optimizer.optimize()
    print("\n--- Solution Validation ---")
    optimizer.validate_solution()

    # 1. Plot the final optimized routes in a static plot
    plot_final_routes(optimizer.best_routes, customers, depot, title="Final Optimized Customer Routes")

    # Test if all customers are visited
    test_all_customers_visited(optimizer.best_routes, customers)

    # 2. Animation for vehicle routes over iterations
    fig, ax1 = plt.subplots()
    ani_routes = FuncAnimation(fig, animate_routes, frames=len(optimizer.history), fargs=(optimizer, customers, depot, ax1), interval=500, repeat=False)
    plt.show()

    # 3. Animation for pheromone trails over iterations
    fig, ax2 = plt.subplots()
    ani_pheromones = FuncAnimation(fig, animate_pheromones, frames=len(optimizer.pheromone_matrix_history), fargs=(optimizer, customers, depot, ax2), interval=500, repeat=False)
    plt.show()

if __name__ == "__main__":
    main()