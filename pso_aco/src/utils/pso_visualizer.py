import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List, Dict
import matplotlib.animation as animation

class PSOVisualizer:
    def __init__(self, problem, distance_matrix):
        self.problem = problem
        self.distance_matrix = distance_matrix
        self.particle_positions_history = []  # List of particle positions per iteration
        self.particle_velocities_history = []  # List of velocities per iteration
        self.best_positions_history = []  # Global best positions per iteration
        self.best_distances = []  # Best distances found per iteration
        self.diversity_history = []  # Swarm diversity per iteration
        self.route_history = []  # Best routes found per iteration
        
    def record_state(self, particles, global_best_position, best_distance, current_routes):
        """Record state for visualization"""
        # Record particle positions and velocities
        positions = np.array([p.position for p in particles])
        velocities = np.array([p.velocity if hasattr(p, 'velocity') else np.zeros_like(p.position) for p in particles])
        
        self.particle_positions_history.append(positions.copy())
        self.particle_velocities_history.append(velocities.copy())
        self.best_positions_history.append(global_best_position.copy())
        self.best_distances.append(best_distance)
        self.route_history.append([route.copy() for route in current_routes])
        
        # Calculate and record swarm diversity
        diversity = np.mean(np.std(positions, axis=0))
        self.diversity_history.append(diversity)
    
    def plot_convergence(self, save_path=None):
        """Plot convergence metrics"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot best distance
        ax1.plot(self.best_distances, 'b-', label='Global Best Distance')
        ax1.set_title('PSO Convergence')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Total Distance')
        ax1.grid(True)
        ax1.legend()
        
        # Plot swarm diversity
        ax2.plot(self.diversity_history, 'r-', label='Swarm Diversity')
        ax2.set_title('Swarm Diversity')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Diversity')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def create_route_animation(self, save_path=None, interval=500):
        """Animate evolution of best route found"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        def update(frame):
            ax.clear()
            
            # Plot depot and customers
            ax.scatter(self.problem.depot.x, self.problem.depot.y, 
                      c='red', marker='s', s=100, label='Depot')
            ax.scatter([c.x for c in self.problem.customers], 
                      [c.y for c in self.problem.customers],
                      c='blue', marker='o', s=50, label='Customers')
            
            # Plot current best routes
            routes = self.route_history[frame]
            colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(routes)))
            
            for route, color in zip(routes, colors):
                route_coords = [(self.problem.depot.x, self.problem.depot.y)]
                for cust_id in route:
                    customer = self.problem.customers[cust_id-1]
                    route_coords.append((customer.x, customer.y))
                route_coords.append((self.problem.depot.x, self.problem.depot.y))
                
                xs, ys = zip(*route_coords)
                ax.plot(xs, ys, c=color, linewidth=2)
            
            ax.set_title(f'Iteration {frame}\nDistance: {self.best_distances[frame]:.2f}')
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