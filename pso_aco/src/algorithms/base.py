from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple
from src.utils.data_loader import Problem
from dataclasses import dataclass

@dataclass
class Solution:
    routes: List[List[int]]
    total_distance: float
    feasible: bool

class BaseOptimizer(ABC):
    def __init__(self, 
                 problem: Problem, 
                 distance_matrix: np.ndarray, 
                 time_matrix: np.ndarray,
                 **kwargs):  
        self.problem = problem
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix
        self.best_solution = None
        
        
    @abstractmethod
    def optimize(self, max_iterations: int) -> Solution:
        """Run the optimization algorithm."""
        pass
    
    def calculate_route_distance(self, route: List[int]) -> float:
        """Calculate total distance of a route including return to depot."""
        if not route:
            return 0.0
            
        distance = self.distance_matrix[0][route[0]]  # Depot to first
        
        for i in range(len(route) - 1):
            distance += self.distance_matrix[route[i]][route[i + 1]]
            
        distance += self.distance_matrix[route[-1]][0]  # Last to depot
        return distance
    
    def is_capacity_feasible(self, route: List[int]) -> bool:
        """Check if route satisfies vehicle capacity constraint."""
        total_demand = sum(self.problem.customers[i-1].demand for i in route)
        return total_demand <= self.problem.capacity