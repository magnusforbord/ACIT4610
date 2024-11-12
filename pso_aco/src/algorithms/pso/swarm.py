import numpy as np
from typing import List, Tuple
from .particle import Particle
from src.utils.data_loader import Problem

class Swarm:
    def __init__(self,
                 problem: Problem,
                 n_particles: int,
                 distance_matrix: np.ndarray,
                 time_matrix: np.ndarray,
                 w: float = 0.7,
                 c1: float = 1.5,
                 c2: float = 1.5):
        """Initialize swarm with parameters."""
        self.problem = problem
        self.n_particles = n_particles
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix
        
        # PSO parameters
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive parameter
        self.c2 = c2  # Social parameter
        
        # Initialize particles
        self.particles = [
            Particle(problem, distance_matrix, time_matrix)
            for _ in range(n_particles)
        ]
        
        # Initialize global best with first particle's position
        self.global_best_position = [route.copy() for route in self.particles[0].position]
        self.global_best_cost = self._calculate_total_cost(self.global_best_position)
        
        # Update global best based on all initial positions
        self.update_global_best()
        
    def update_global_best(self):
        """Update global best solution based on all particles."""
        for particle in self.particles:
            cost = self._calculate_total_cost(particle.position)
            
            # Update particle's personal best
            if cost < particle.personal_best_cost:
                particle.personal_best_position = [route.copy() for route in particle.position]
                particle.personal_best_cost = cost
                
                # Update global best if needed
                if cost < self.global_best_cost and self._is_solution_feasible(particle.position):
                    self.global_best_position = [route.copy() for route in particle.position]
                    self.global_best_cost = cost
    
    def move_particles(self):
        """Update velocity and position of all particles."""
        for particle in self.particles:
            particle.update_velocity(
                self.global_best_position,
                self.w,
                self.c1,
                self.c2
            )
            particle.update_position()
    
    def _calculate_total_cost(self, solution: List[List[int]]) -> float:
        """Calculate total distance of all routes in solution."""
        if not solution:
            return float('inf')
            
        total_cost = 0
        for route in solution:
            if not route:
                continue
                
            # Distance from depot to first customer
            cost = self.distance_matrix[0][route[0]]
            
            # Distances between consecutive customers
            for i in range(len(route) - 1):
                cost += self.distance_matrix[route[i]][route[i + 1]]
            
            # Distance from last customer back to depot
            cost += self.distance_matrix[route[-1]][0]
            
            total_cost += cost
            
        return total_cost
    
    def _is_solution_feasible(self, solution: List[List[int]]) -> bool:
        """Check if complete solution satisfies all constraints."""
        if not solution:
            return False
            
        # Check if all customers are visited exactly once
        visited = set()
        for route in solution:
            for customer in route:
                if customer in visited:
                    return False
                visited.add(customer)
        
        if len(visited) != len(self.problem.customers):
            return False
            
        # Check vehicle limit
        if len(solution) > self.problem.vehicles:
            return False
            
        # Check capacity and time windows for each route
        for route in solution:
            # Check capacity
            total_demand = sum(self.problem.customers[c-1].demand for c in route)
            if total_demand > self.problem.capacity:
                return False
                
            # Check time windows
            current_time = 0
            current_pos = 0
            
            for customer_id in route:
                customer = self.problem.customers[customer_id-1]
                travel_time = self.time_matrix[current_pos][customer_id]
                arrival_time = current_time + travel_time
                
                if arrival_time > customer.due_time:
                    return False
                    
                service_start = max(arrival_time, customer.ready_time)
                current_time = service_start + customer.service_time
                current_pos = customer_id
                
            # Check return to depot
            final_time = current_time + self.time_matrix[current_pos][0]
            if final_time > self.problem.depot.due_time:
                return False
                
        return True
    
    def get_best_solution(self) -> Tuple[List[List[int]], float]:
        """Return the best solution found by the swarm."""
        return self.global_best_position, self.global_best_cost