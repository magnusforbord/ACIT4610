
from src.algorithms.pso.particle import Particle
from src.utils.data_loader import Problem
import numpy as np

class Swarm:
    def __init__(self, 
                 problem: Problem,
                 n_particles: int,
                 distance_matrix: np.ndarray,
                 time_matrix: np.ndarray,
                 w: float = 0.6,
                 c1: float = 1.8,
                 c2: float = 1.8):
        self.particles = []
        n_heuristic = int(n_particles * 0.7)  # 70% use heuristic
        n_random = n_particles - n_heuristic

        for _ in range(n_heuristic):
            p = Particle(problem, distance_matrix, time_matrix)
            p.position = p._nearest_neighbor_init()
            self.particles.append(p)
        
        # Create particles with random initialization
        for _ in range(n_random):
            p = Particle(problem, distance_matrix, time_matrix)
            self.particles.append(p)

        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.global_best_routes = None

        
    def optimize(self, iterations: int = 1):
        for it in range(iterations):
            
            w = self.w - (self.w - 0.4) * it / iterations
            
            # Update each particle
            for i, particle in enumerate(self.particles):

                # Evaluate current position
                fitness = particle.evaluate()

                
                # Update global best if needed
                if fitness < self.global_best_fitness:
                    print(f"New global best found: {fitness}")
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position.copy()
                    self.global_best_routes = particle.best_routes.copy()
                    
                # Update particle velocity and position
                if self.global_best_position is not None:
                    particle.update_velocity(w, self.c1, self.c2, self.global_best_position)
                    particle.update_position()


        return self.global_best_fitness, self.global_best_routes