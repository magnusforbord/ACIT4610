import numpy as np
from typing import List, Tuple
from .ant import Ant
from src.utils.data_loader import Problem

class Colony:
    def __init__(self, 
                 problem: Problem,
                 n_ants: int,
                 distance_matrix: np.ndarray,
                 time_matrix: np.ndarray,
                 alpha: float = 1.0,
                 beta: float = 2.0,
                 rho: float = 0.1):
        self.problem = problem
        self.n_ants = n_ants
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        
        # Initialize pheromone matrix
        n_nodes = len(problem.customers) + 1
        self.pheromone = np.ones((n_nodes, n_nodes)) * 0.1
        
    def construct_solutions(self) -> List[List[List[int]]]:
        """Generate solutions (multiple routes) using all ants."""
        solutions = []
        
        for _ in range(self.n_ants):
            ant = Ant(self.problem, self.distance_matrix, self.time_matrix)
            routes = ant.construct_solution(self.pheromone, self.alpha, self.beta)
            solutions.append(routes)
            
        return solutions
    
    def update_pheromone(self, solutions: List[List[List[int]]], solution_costs: List[float]):
        """Update pheromone levels based on complete solutions."""
        # Evaporation
        self.pheromone *= (1 - self.rho)
        
        # Add new pheromone for each solution
        for routes, cost in zip(solutions, solution_costs):
            if not routes:
                continue
                
            deposit = 1.0 / cost if cost > 0 else 1.0
            
            # Update pheromone for each route in the solution
            for route in routes:
                if not route:
                    continue
                    
                # Deposit pheromone on route edges
                prev = 0  # depot
                for customer in route:
                    self.pheromone[prev][customer] += deposit
                    self.pheromone[customer][prev] += deposit
                    prev = customer
                    
                # Return to depot
                self.pheromone[prev][0] += deposit
                self.pheromone[0][prev] += deposit

    def calculate_total_cost(self, solution: List[List[int]]) -> float:
        """Calculate total distance of all routes in solution."""
        total_cost = 0
        
        for route in solution:
            if not route:
                continue
                
            # Add distance from depot to first customer
            cost = self.distance_matrix[0][route[0]]
            
            # Add distances between customers
            for i in range(len(route) - 1):
                cost += self.distance_matrix[route[i]][route[i + 1]]
                
            # Add distance back to depot
            cost += self.distance_matrix[route[-1]][0]
            
            total_cost += cost
            
        return total_cost