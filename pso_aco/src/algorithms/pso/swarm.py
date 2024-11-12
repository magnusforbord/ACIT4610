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
                 c1: float = 2.05,  
                 c2: float = 2.05): 
        self.problem = problem
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix
        self.n_particles = n_particles
        
        # Save PSO parameters
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        self.particles = []
        
        # Multi-strategy initialization
        n_nearest = int(n_particles * 0.4)  # 40% nearest neighbor
        n_savings = int(n_particles * 0.3)  # 30% savings
        n_random = n_particles - n_nearest - n_savings  # 30% random
        
        # Nearest neighbor initialization
        for _ in range(n_nearest):
            p = Particle(problem, distance_matrix, time_matrix)
            p.position = self._nearest_neighbor_init()
            self.particles.append(p)
            
        # Savings-based initialization
        for _ in range(n_savings):
            p = Particle(problem, distance_matrix, time_matrix)
            p.position = self._savings_based_init()
            self.particles.append(p)
            
        # Random initialization
        for _ in range(n_random):
            p = Particle(problem, distance_matrix, time_matrix)
            self.particles.append(p)

        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.global_best_routes = None

    def _nearest_neighbor_init(self) -> np.ndarray:
        """Initialize using nearest neighbor with time windows"""
        n_customers = len(self.problem.customers)
        position = np.zeros(n_customers)
        unvisited = set(range(n_customers))
        current = 0  # Start at depot
        current_time = 0
        
        while unvisited:
            best_next = None
            best_value = float('inf')
            
            for next_cust in unvisited:
                cust = self.problem.customers[next_cust]
                travel_time = self.time_matrix[current][next_cust + 1]
                arrival_time = current_time + travel_time
                
                if arrival_time <= cust.due_time:
                    # Score combines distance and time window urgency
                    score = (travel_time + 
                            0.5 * (cust.due_time - arrival_time) +
                            0.3 * cust.demand)
                    if score < best_value:
                        best_value = score
                        best_next = next_cust
            
            if best_next is not None:
                position[best_next] = 1.0 - (len(position) - len(unvisited)) / len(position)
                unvisited.remove(best_next)
                current = best_next + 1
                current_time = max(current_time + self.time_matrix[current-1][current],
                                 self.problem.customers[best_next].ready_time) + \
                             self.problem.customers[best_next].service_time
            else:
                # Assign remaining randomly
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
        diversity_threshold = 0.1
        
        for it in range(iterations):
            # Calculate swarm diversity
            positions = np.array([p.position for p in self.particles])
            diversity = np.mean(np.std(positions, axis=0))
            
            # Reinitialize worst particles if diversity is too low
            if diversity < diversity_threshold:
                self._reinitialize_worst_particles()

        for it in range(iterations):
            # Update each particle
            for particle in self.particles:
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
                    w = self.w - (self.w - 0.4) * it / iterations
                    particle.update_velocity(w, self.c1, self.c2, self.global_best_position)
                    particle.update_position()

        return self.global_best_fitness, self.global_best_routes
    
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
