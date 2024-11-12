import numpy as np
from src.algorithms.base import BaseOptimizer, Solution
from .swarm import Swarm
import time
from typing import List, Tuple

class PSOOptimizer(BaseOptimizer):
    def __init__(self,
                 problem,
                 distance_matrix,
                 time_matrix,
                 n_particles: int = 50,
                 w: float = 0.7,
                 c1: float = 1.5,
                 c2: float = 1.5):
        super().__init__(problem, distance_matrix, time_matrix)
        self.swarm = Swarm(
            problem=problem,
            n_particles=n_particles,
            distance_matrix=distance_matrix,
            time_matrix=time_matrix,
            w=w,
            c1=c1,
            c2=c2
        )
        
    def optimize(self, max_iterations: int) -> Solution:
        """Run PSO optimization."""
        best_solution = None
        best_cost = float('inf')
        iterations_without_improvement = 0
        
        print("\nStarting PSO optimization...")
        start_time = time.time()
        
        # Check initial solutions
        for particle in self.swarm.particles:
            if (self._is_feasible(particle.position) and 
                particle.personal_best_cost < best_cost):
                best_solution = [route.copy() for route in particle.position]
                best_cost = particle.personal_best_cost
                print(f"New best solution found!")
                print(f"Cost: {best_cost:.2f}")
                print(f"Number of routes: {len(best_solution)}")
                print("-" * 40)
        
        # Run iterations
        for iteration in range(max_iterations):
            improved = False
            
            # Move particles
            for particle in self.swarm.particles:
                particle.update_velocity(
                    self.swarm.global_best_position,
                    self.swarm.w,
                    self.swarm.c1,
                    self.swarm.c2
                )
                particle.update_position()
                
                # Check new position
                if (self._is_feasible(particle.position) and 
                    particle.personal_best_cost < best_cost):
                    best_solution = [route.copy() for route in particle.position]
                    best_cost = particle.personal_best_cost
                    self.swarm.global_best_position = [route.copy() for route in particle.position]
                    self.swarm.global_best_cost = particle.personal_best_cost
                    improved = True
                    iterations_without_improvement = 0
                    
                    print(f"\nNew best solution found in iteration {iteration}!")
                    print(f"Cost: {best_cost:.2f}")
                    print(f"Number of routes: {len(best_solution)}")
                    print("-" * 40)

            if not improved:
                iterations_without_improvement += 1
            
        
        end_time = time.time()
        time_taken = end_time - start_time

        print("\nOptimization Summary:")
        print("-" * 40)
        print(f"Time taken: {time_taken:.2f} seconds")
        print(f"Total iterations: {max_iterations}")
        print(f"Final best cost: {best_cost:.2f}")
        print(f"Number of routes: {len(best_solution) if best_solution else 0}")
        print(f"Iterations without improvement: {iterations_without_improvement}")
        
        if best_solution is None:
            print("WARNING: No feasible solution found!")
            return Solution(routes=[], total_distance=float('inf'), feasible=False)
        
        return Solution(
            routes=best_solution,
            total_distance=best_cost,
            feasible=True
        )

    def _is_feasible(self, solution: List[List[int]]) -> bool:
        """Check if solution is feasible."""
        feasible, _ = self._check_feasibility_detailed(solution)
        return feasible

    def _check_feasibility_detailed(self, solution: List[List[int]]) -> Tuple[bool, str]:
        """Detailed feasibility check with reason for failure."""
        if not solution:
            return False, "Empty solution"
            
        # Check vehicle limit
        if len(solution) > self.problem.vehicles:
            return False, f"Too many vehicles: {len(solution)} > {self.problem.vehicles}"
        
        # Check customer coverage
        visited = set()
        for route in solution:
            for customer in route:
                if customer in visited:
                    return False, f"Customer {customer} visited multiple times"
                visited.add(customer)
        
        if len(visited) != len(self.problem.customers):
            missing = set(range(1, len(self.problem.customers) + 1)) - visited
            return False, f"Missing customers: {missing}"
        
        # Check each route
        for i, route in enumerate(solution):
            # Check capacity
            total_demand = sum(self.problem.customers[c-1].demand for c in route)
            if total_demand > self.problem.capacity:
                return False, f"Route {i} exceeds capacity: {total_demand} > {self.problem.capacity}"
            
            # Check time windows
            current_time = 0
            current_pos = 0
            
            for customer_id in route:
                customer = self.problem.customers[customer_id-1]
                travel_time = self.time_matrix[current_pos][customer_id]
                arrival_time = current_time + travel_time
                
                if arrival_time > customer.due_time:
                    return False, f"Route {i}: Late arrival for customer {customer_id} at time {arrival_time} > {customer.due_time}"
                
                service_start = max(arrival_time, customer.ready_time)
                current_time = service_start + customer.service_time
                current_pos = customer_id
            
            # Check return to depot
            final_time = current_time + self.time_matrix[current_pos][0]
            if final_time > self.problem.depot.due_time:
                return False, f"Route {i}: Late return to depot at time {final_time} > {self.problem.depot.due_time}"
        
        return True, "Solution is feasible"

    def _calculate_cost(self, solution: List[List[int]]) -> float:
        """Calculate total distance of a solution."""
        if not solution:
            return float('inf')
            
        total = 0
        for route in solution:
            if not route:
                continue
            # Depot to first customer
            total += self.distance_matrix[0][route[0]]
            # Between customers
            for i in range(len(route) - 1):
                total += self.distance_matrix[route[i]][route[i + 1]]
            # Last customer back to depot
            total += self.distance_matrix[route[-1]][0]
        return total