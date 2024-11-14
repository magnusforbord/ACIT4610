import numpy as np
from typing import List, Set
from src.utils.data_loader import Problem

class Ant:
    def __init__(self, problem: Problem, distance_matrix: np.ndarray, time_matrix: np.ndarray):
        """Creates an ant agent for solving VRPTW.
        Each ant represents a potential solution constructor that builds routes
        while respecting vehicle capacity and time window constraints."""
        self.problem = problem
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix
        self.reset()
        
    def reset(self):
        """Reset ant for new solution construction."""
        self.routes: List[List[int]] = []
        self.current_route: List[int] = []
        self.unvisited: Set[int] = set(range(1, len(self.problem.customers) + 1))
        self.current_pos = 0  # depot
        self.current_time = 0
        self.current_capacity = self.problem.capacity

    def construct_solution(self, pheromone: np.ndarray, alpha: float, beta: float) -> list[list[int]]:
        """ Constructs a complete solution (set of routes) using ACO principles.
            Uses pheromone levels and heuristic information to probabilistically build routes
            that satisfy problem constraints."""
        self.reset()
        available_vehicles = self.problem.vehicles
        
        while self.unvisited and available_vehicles > 0:
            # Start new route
            self.current_pos = 0
            self.current_time = 0
            self.current_capacity = self.problem.capacity
            self.current_route = []
            available_vehicles -= 1
            
            while self.unvisited:
                next_customer = self.select_next_customer(pheromone, alpha, beta)
                if next_customer == 0:
                    break
                    
                # Check if adding this customer would violate time windows
                if not self._is_time_feasible(next_customer):
                    break
                
                self.current_route.append(next_customer)
                self.unvisited.remove(next_customer)
                
                # Update state
                customer = self.problem.customers[next_customer-1]
                self.current_capacity -= customer.demand
                travel_time = self.time_matrix[self.current_pos][next_customer]
                self.current_time = max(self.current_time + travel_time, customer.ready_time) + customer.service_time
                self.current_pos = next_customer
            
            if self.current_route:
                self.routes.append(self.current_route)
        
        return self.routes
    
    def select_next_customer(self, pheromone: np.ndarray, alpha: float, beta: float) -> int:
        """Selects next customer using ACO probability rules.
        Combines pheromone levels with heuristic information about distance,
        time windows, and capacity to make selection."""
        if not self.unvisited:
            return 0
            
        candidates = []
        probabilities = []
        
        # Get earliest and latest due times for scaling
        due_times = [self.problem.customers[c-1].due_time for c in self.unvisited]
        earliest_due = min(due_times)
        latest_due = max(due_times)
        due_range = latest_due - earliest_due if latest_due > earliest_due else 1
        
        for next_customer in self.unvisited:
            customer = self.problem.customers[next_customer-1]
            
            # Skip if basic constraints would be violated
            if not self._is_feasible(next_customer):
                continue
                
            # Calculate temporal aspects
            travel_time = self.time_matrix[self.current_pos][next_customer]
            arrival_time = self.current_time + travel_time
            
            # Skip if we arrive after due time
            if arrival_time > customer.due_time:
                continue
                
            # Calculate urgency score (0 to 1)
            time_score = 1.0 - (customer.due_time - earliest_due) / due_range
            
            # Calculate distance score (closer is better)
            distance = self.distance_matrix[self.current_pos][next_customer]
            distance_score = 1.0 / (1.0 + distance)
            
            # Calculate capacity score
            capacity_score = 1.0 - (customer.demand / self.current_capacity)
            
            # Combine all factors
            pheromone_val = pheromone[self.current_pos][next_customer]
            attractiveness = (
                distance_score * 
                (1 + time_score) * 
                (1 + capacity_score)
            )
            
            prob = (pheromone_val ** alpha) * (attractiveness ** beta)
            
            candidates.append(next_customer)
            probabilities.append(prob)
        
        if not candidates:
            return 0
        
        total = sum(probabilities)
        if total == 0:
            return np.random.choice(candidates)
        
        probabilities = [p/total for p in probabilities]
        return np.random.choice(candidates, p=probabilities)
    
    def _is_feasible(self, next_customer: int) -> bool:
        """Checks if adding next_customer violates any constraints."""
        customer = self.problem.customers[next_customer-1]
        
        # Check capacity
        if customer.demand > self.current_capacity:
            return False
            
        # Check if we can reach customer in time
        travel_time = self.time_matrix[self.current_pos][next_customer]
        arrival_time = self.current_time + travel_time
        
        if arrival_time > customer.due_time:
            return False
            
        # Check if we can get back to depot
        service_end = max(arrival_time, customer.ready_time) + customer.service_time
        return_time = service_end + self.time_matrix[next_customer][0]
        
        if return_time > self.problem.depot.due_time:
            return False
            
        return True
    
    def _is_time_feasible(self, next_customer: int) -> bool:
        """Checks time window feasibility for next customer."""
        customer = self.problem.customers[next_customer-1]
        travel_time = self.time_matrix[self.current_pos][next_customer]
        arrival_time = self.current_time + travel_time
        
        # Must arrive before due time
        if arrival_time > customer.due_time:
            return False
            
        # Check if service can start within time window
        service_start = max(arrival_time, customer.ready_time)
        if service_start > customer.due_time:
            return False
            
        # Check if we can return to depot in time
        service_end = service_start + customer.service_time
        return_time = service_end + self.time_matrix[next_customer][0]
        
        return return_time <= self.problem.depot.due_time