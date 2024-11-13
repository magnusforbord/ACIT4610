import time
import numpy as np
from src.algorithms.pso.particle import Particle
from src.utils.data_loader import Problem

class Swarm:
    def __init__(self, 
            problem: Problem,
            n_particles: int,
            distance_matrix: np.ndarray,
            time_matrix: np.ndarray,
            w: float = 0.9,
            c1: float = 2.0,
            c2: float = 2.0):
        np.random.seed(int(time.time() * 1000) % 2**32)
        
        self.problem = problem
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix
        self.n_particles = n_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        self.particles = []
        
        # Structured initialization
        n_nearest = int(n_particles * 0.5)  # Increased nearest neighbor portion
        n_savings = int(n_particles * 0.3)
        n_random = n_particles - n_nearest - n_savings
        
        # Create base positions using each strategy
        base_nearest = self._nearest_neighbor_init()
        base_savings = self._savings_based_init()
        
        # Initialize nearest neighbor based particles with controlled perturbation
        for i in range(n_nearest):
            p = Particle(problem, distance_matrix, time_matrix)
            perturbation = np.random.normal(0, 0.05 + (i/n_nearest)*0.1, size=len(base_nearest))
            p.position = np.clip(base_nearest + perturbation, 0, 1)
            self.particles.append(p)
        
        # Initialize savings based particles with controlled perturbation
        for i in range(n_savings):
            p = Particle(problem, distance_matrix, time_matrix)
            perturbation = np.random.normal(0, 0.05 + (i/n_savings)*0.1, size=len(base_savings))
            p.position = np.clip(base_savings + perturbation, 0, 1)
            self.particles.append(p)
        
        # Random particles with bias towards good regions
        for _ in range(n_random):
            p = Particle(problem, distance_matrix, time_matrix)
            if np.random.random() < 0.7:
                # Blend of nearest and savings
                blend = np.random.random()
                p.position = np.clip(
                    blend * base_nearest + (1-blend) * base_savings + 
                    np.random.normal(0, 0.1, size=len(base_nearest)), 
                    0, 1
                )
            else:
                # Pure random with slight bias towards middle values
                p.position = np.random.beta(2, 2, size=len(base_nearest))
            self.particles.append(p)

        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.global_best_routes = None


    def _nearest_neighbor_init(self) -> np.ndarray:
        """Initialize using nearest neighbor with randomization"""
        n_customers = len(self.problem.customers)
        position = np.zeros(n_customers)
        unvisited = set(range(n_customers))
        current = 0
        current_time = 0
        
        while unvisited:
            feasible_next = []
            feasible_scores = []
            
            for next_cust in unvisited:
                cust = self.problem.customers[next_cust]
                travel_time = self.time_matrix[current][next_cust + 1]
                arrival_time = current_time + travel_time
                
                if arrival_time <= cust.due_time:
                    score = (travel_time + 
                            0.5 * (cust.due_time - arrival_time) +
                            0.3 * cust.demand +
                            np.random.normal(0, 20))  # Add random noise to score
                    feasible_next.append(next_cust)
                    feasible_scores.append(score)
            
            if feasible_next:
                # Sometimes pick random feasible customer instead of best
                if np.random.random() < 0.2:  # 20% chance
                    next_cust = np.random.choice(feasible_next)
                else:
                    next_cust = feasible_next[np.argmin(feasible_scores)]
                    
                position[next_cust] = 1.0 - (len(position) - len(unvisited)) / len(position)
                position[next_cust] *= (1 + np.random.normal(0, 0.1))  # Add noise to position value
                unvisited.remove(next_cust)
                current = next_cust + 1
                current_time = max(current_time + self.time_matrix[current-1][current],
                                self.problem.customers[next_cust].ready_time) + \
                            self.problem.customers[next_cust].service_time
            else:
                # Randomly assign remaining
                for cust in unvisited:
                    position[cust] = np.random.random()
                break
        
        return np.clip(position, 0, 1)

    def _savings_based_init(self) -> np.ndarray:
        """Initialize using savings algorithm with time windows"""
        n_customers = len(self.problem.customers)
        position = np.zeros(n_customers)
        
        # Calculate savings for all customer pairs
        savings = []
        for i in range(n_customers):
            cust_i = i + 1
            for j in range(i + 1, n_customers):
                cust_j = j + 1
                # Basic savings calculation
                saving = (self.distance_matrix[0][cust_i] + 
                         self.distance_matrix[0][cust_j] - 
                         self.distance_matrix[cust_i][cust_j])
                
                # Adjust savings based on time windows compatibility
                cust1 = self.problem.customers[i]
                cust2 = self.problem.customers[j]
                time_compatibility = abs(cust1.due_time - cust2.due_time)
                adjusted_saving = saving * (1 + 1/time_compatibility if time_compatibility > 0 else saving)
                
                savings.append((adjusted_saving, i, j))
        
        # Sort savings in descending order
        savings.sort(reverse=True)
        
        # Assign position values based on savings rank
        max_rank = len(savings)
        seen = set()
        
        for rank, (_, i, j) in enumerate(savings):
            if i not in seen:
                position[i] = 1.0 - (rank / max_rank)
                seen.add(i)
            if j not in seen:
                position[j] = 1.0 - (rank / max_rank)
                seen.add(j)
        
        # Fill remaining positions
        for i in range(n_customers):
            if i not in seen:
                position[i] = np.random.random()
        
        return np.clip(position, 0, 1)
        
    def optimize(self, iterations: int = 1):
        diversity_threshold = 0.15
        n_stagnant = 0
        max_stagnant = 15  # Increased patience
        last_best = float('inf')
        
        # More aggressive parameter adaptation
        w_start, w_end = 0.95, 0.2
        c1_start, c1_end = 2.8, 1.2
        c2_start, c2_end = 1.2, 2.8

        for it in range(iterations):
            # Non-linear parameter adaptation
            progress = (it / iterations) ** 0.8  # Non-linear decay
            w = w_start - (w_start - w_end) * progress
            c1 = c1_start - (c1_start - c1_end) * progress
            c2 = c2_start + (c2_end - c2_start) * progress
            
            positions = np.array([p.position for p in self.particles])
            diversity = np.mean(np.std(positions, axis=0))
            
            # Enhanced diversity management
            if diversity < diversity_threshold:
                if n_stagnant > max_stagnant:
                    # Partial reset with memory retention
                    n_reset = len(self.particles) // 3  # Reset 33% of particles
                    sorted_particles = sorted(self.particles, key=lambda p: p.best_fitness)
                    for particle in sorted_particles[-n_reset:]:
                        # Mix random with best known positions
                        if np.random.random() < 0.7:
                            particle.position = np.random.random(size=len(particle.position))
                        else:
                            # Perturb global best
                            particle.position = self.global_best_position + np.random.normal(0, 0.3, size=len(particle.position))
                            particle.position = np.clip(particle.position, 0, 1)
                    n_stagnant = 0
                
                # Increase exploration for remaining particles
                w *= 1.2  # Boost inertia weight
                
            for particle in self.particles:
                fitness = particle.evaluate()
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position.copy()
                    self.global_best_routes = particle.best_routes.copy()
                    n_stagnant = 0
                
                particle.update_velocity(w, c1, c2, self.global_best_position)
            
            if abs(self.global_best_fitness - last_best) < 0.001:
                n_stagnant += 1
            last_best = self.global_best_fitness
                
    def _reinitialize_worst_particles(self):
        """Reinitialize worst performing particles"""
        n_reinit = max(3, len(self.particles) // 10)  # Reinit 10% of particles
        sorted_particles = sorted(self.particles, key=lambda p: p.best_fitness, reverse=True)
        
        for particle in sorted_particles[:n_reinit]:
            if np.random.random() < 0.5:
                particle.position = particle._nearest_neighbor_init()
            else:
                particle.position = self._savings_based_init()
            particle.velocity = np.random.uniform(-0.4, 0.4, size=len(particle.position))

