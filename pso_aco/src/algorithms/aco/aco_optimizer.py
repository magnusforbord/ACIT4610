from src.algorithms.base import BaseOptimizer, Solution
from .colony import Colony
import numpy as np
from typing import List, Tuple
import time

class ACOOptimizer(BaseOptimizer):
    def __init__(self, 
                 problem, 
                 distance_matrix, 
                 time_matrix,
                 n_ants: int = 50,
                 alpha: float = 1.0,
                 beta: float = 5.0,
                 rho: float = 0.1):
        super().__init__(problem, distance_matrix, time_matrix)
        self.colony = Colony(
            problem=problem,
            n_ants=n_ants,
            distance_matrix=distance_matrix,
            time_matrix=time_matrix,
            alpha=alpha,
            beta=beta,
            rho=rho
        )
        
    def optimize(self, max_iterations: int) -> Solution:
        best_solution = None
        best_distance = float('inf')
        no_improvement = 0
        
        start_time = time.time()
        for iteration in range(max_iterations):
            # Construct solutions with all ants
            solutions = self.colony.construct_solutions()
            
            # Evaluate all solutions
            for routes in solutions:
                if not self.is_solution_feasible(routes):
                    continue
                    
                distance = self.calculate_total_distance(routes)
                if distance < best_distance:
                    best_distance = distance
                    best_solution = routes.copy()
                    no_improvement = 0
                    print(f"Iteration {iteration}: New best distance = {best_distance:.2f}, Routes = {len(routes)}")
                else:
                    no_improvement += 1
            
            # Update pheromone trails
            solution_costs = [self.calculate_total_distance(s) for s in solutions]
            self.colony.update_pheromone(solutions, solution_costs)
            
        end_time = time.time()
        print(f"\nOptimization completed in {end_time - start_time:.2f} seconds")
        
        return Solution(
            routes=best_solution if best_solution else [],
            total_distance=best_distance if best_solution else float('inf'),
            feasible=True if best_solution else False
        )
    
    def calculate_total_distance(self, routes: List[List[int]]) -> float:
        """Calculate total distance for all routes."""
        if not routes:
            return float('inf')
            
        total = 0
        for route in routes:
            if route:
                total += self.calculate_route_distance(route)
        return total
    
    def is_solution_feasible(self, routes: List[List[int]]) -> bool:
        """Check if complete solution satisfies all constraints."""
        if not routes:
            return False
            
        # Check if all customers are visited exactly once
        visited = set()
        for route in routes:
            for customer in route:
                if customer in visited:
                    return False
                visited.add(customer)
        
        if len(visited) != len(self.problem.customers):
            return False
            
        # Check each route's constraints
        for route in routes:
            if not self.is_route_feasible(route):
                return False
                
        return True
    
    def is_route_feasible(self, route: List[int]) -> bool:
        """Check if single route satisfies capacity and time constraints."""
        if not route:
            return True
            
        # Check capacity constraint
        total_demand = sum(self.problem.customers[c-1].demand for c in route)
        if total_demand > self.problem.capacity:
            return False
            
        # Check time windows
        current_time = 0
        current_pos = 0  # Start at depot
        
        for customer_id in route:
            customer = self.problem.customers[customer_id - 1]
            
            # Add travel time
            travel_time = self.time_matrix[current_pos][customer_id]
            arrival_time = current_time + travel_time
            
            # Check if we arrived too late
            if arrival_time > customer.due_time:
                return False
                
            # Update current time (wait if arrived too early)
            current_time = max(arrival_time, customer.ready_time) + customer.service_time
            current_pos = customer_id
            
        # Check return to depot
        final_travel_time = self.time_matrix[current_pos][0]
        final_arrival = current_time + final_travel_time
        
        return final_arrival <= self.problem.depot.due_time