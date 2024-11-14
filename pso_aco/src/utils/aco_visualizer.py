import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List, Dict
import matplotlib.animation as animation

class ACOVisualizer:
    def __init__(self, problem, distance_matrix):
        self.problem = problem
        self.distance_matrix = distance_matrix
        self.pheromone_history = []
        self.best_distances = []
        self.route_history = []
        self.iteration_times = []
        
    def record_state(self, pheromone_matrix, best_distance, current_routes):
        """Record state for visualization"""
        self.pheromone_history.append(pheromone_matrix.copy())
        self.best_distances.append(best_distance)
        self.route_history.append([route.copy() for route in current_routes])
    
    def plot_pheromone_evolution(self, save_path=None):
        """Plot pheromone network at different iterations"""
        iterations = [0, len(self.pheromone_history)//2, -1]  # Start, middle, end
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, iter_num in enumerate(iterations):
            pheromone = self.pheromone_history[iter_num]
            ax = axes[idx]
            
            # Plot depot
            ax.scatter([self.problem.depot.x], [self.problem.depot.y], 
                    c='red', marker='s', s=100, label='Depot')
            
            # Plot customers
            customer_coords = [(c.x, c.y) for c in self.problem.customers]
            x_coords, y_coords = zip(*customer_coords)
            ax.scatter(x_coords, y_coords, c='blue', marker='o', s=50)
            
            # Draw edges with thickness based on pheromone strength
            max_pheromone = np.max(pheromone)
            min_width = 0.5
            max_width = 4.0
            
            # Draw edges between nodes where pheromone > threshold
            threshold = 0.1
            for i in range(len(pheromone)):
                for j in range(i+1, len(pheromone)):
                    if pheromone[i][j] > threshold:
                        # Get coordinates
                        start = (self.problem.depot.x, self.problem.depot.y) if i == 0 \
                            else (self.problem.customers[i-1].x, self.problem.customers[i-1].y)
                        end = (self.problem.depot.x, self.problem.depot.y) if j == 0 \
                            else (self.problem.customers[j-1].x, self.problem.customers[j-1].y)
                        
                        # Calculate line width based on pheromone strength
                        width = min_width + (pheromone[i][j]/max_pheromone) * (max_width - min_width)
                        
                        # Draw line with alpha proportional to pheromone strength
                        alpha = 0.3 + 0.7 * (pheromone[i][j]/max_pheromone)
                        ax.plot([start[0], end[0]], [start[1], end[1]], 
                            'b-', linewidth=width, alpha=alpha)
            
            ax.set_title(f'Iteration {iter_num if iter_num != -1 else len(self.pheromone_history)-1}')
            ax.grid(True)
            
        plt.suptitle('Pheromone Network Evolution')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_convergence(self, save_path=None):
        """Plot convergence of best distance over iterations"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.best_distances, 'b-', label='Best Distance')
        plt.title('ACO Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Total Distance')
        plt.grid(True)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def create_route_animation(self, save_path=None, interval=500):
        """Create animation of route evolution"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot depot and customers
        depot = self.problem.depot
        customers = self.problem.customers
        
        def update(frame):
            ax.clear()
            ax.scatter(depot.x, depot.y, c='red', marker='s', s=100, label='Depot')
            ax.scatter([c.x for c in customers], [c.y for c in customers], 
                      c='blue', marker='o', s=50, label='Customers')
            
            routes = self.route_history[frame]
            colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(routes)))
            
            for route, color in zip(routes, colors):
                route_coords = [(depot.x, depot.y)]
                for cust_id in route:
                    customer = customers[cust_id-1]
                    route_coords.append((customer.x, customer.y))
                route_coords.append((depot.x, depot.y))
                
                xs, ys = zip(*route_coords)
                ax.plot(xs, ys, c=color, linewidth=2)
            
            ax.set_title(f'Iteration {frame}')
            ax.grid(True)
            ax.legend()
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
        
        anim = animation.FuncAnimation(fig, update, 
                                     frames=len(self.route_history), 
                                     interval=interval, 
                                     repeat=False)
        
        if save_path:
            anim.save(save_path, writer='pillow')
        
        plt.show()
        return anim
        
    def animate_pheromone_evolution(self, save_path=None, interval=200):
        """Animate the evolution of pheromone network over iterations"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        def update(frame):
            ax.clear()
            pheromone = self.pheromone_history[frame]
            
            # Plot depot
            ax.scatter([self.problem.depot.x], [self.problem.depot.y], 
                    c='red', marker='s', s=100, label='Depot')
            
            # Plot customers
            customer_coords = [(c.x, c.y) for c in self.problem.customers]
            x_coords, y_coords = zip(*customer_coords)
            ax.scatter(x_coords, y_coords, c='blue', marker='o', s=50, label='Customers')
            
            # Draw edges with thickness based on pheromone strength
            max_pheromone = np.max(pheromone)
            min_width = 0.5
            max_width = 4.0
            
            # Draw edges between nodes where pheromone > threshold
            threshold = 0.1
            for i in range(len(pheromone)):
                for j in range(i+1, len(pheromone)):
                    if pheromone[i][j] > threshold:
                        # Get coordinates
                        start = (self.problem.depot.x, self.problem.depot.y) if i == 0 \
                            else (self.problem.customers[i-1].x, self.problem.customers[i-1].y)
                        end = (self.problem.depot.x, self.problem.depot.y) if j == 0 \
                            else (self.problem.customers[j-1].x, self.problem.customers[j-1].y)
                        
                        # Calculate line width based on pheromone strength
                        width = min_width + (pheromone[i][j]/max_pheromone) * (max_width - min_width)
                        
                        # Draw line with alpha proportional to pheromone strength
                        alpha = 0.3 + 0.7 * (pheromone[i][j]/max_pheromone)
                        ax.plot([start[0], end[0]], [start[1], end[1]], 
                            'b-', linewidth=width, alpha=alpha)
            
            ax.set_title(f'Iteration {frame}')
            ax.grid(True)
            ax.legend()
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            
            # Set consistent axes limits
            ax.set_xlim(min(x_coords) - 5, max(x_coords) + 5)
            ax.set_ylim(min(y_coords) - 5, max(y_coords) + 5)
        
        anim = animation.FuncAnimation(fig, update,
                                    frames=len(self.pheromone_history),
                                    interval=interval,
                                    repeat=False)
        
        if save_path:
            anim.save(save_path, writer='pillow')
        
        plt.show()
        return anim