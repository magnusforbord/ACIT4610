import numpy as np
from typing import List
from src.utils.data_loader import Customer, Problem

def create_distance_matrix(customers: List[Customer], depot: Customer) -> np.ndarray:
    """Create matrix of Euclidean distances between all points (including depot)."""
    all_points = [depot] + customers
    size = len(all_points)
    matrix = np.zeros((size, size))
    
    for i in range(size):
        for j in range(i + 1, size):
            dist = np.sqrt(
                (all_points[i].x - all_points[j].x) ** 2 + 
                (all_points[i].y - all_points[j].y) ** 2
            )
            matrix[i][j] = dist
            matrix[j][i] = dist
    
    return matrix

def create_time_matrix(problem: Problem, distance_matrix: np.ndarray) -> np.ndarray:
    """Create matrix of travel times between all points.
    Assumes unit velocity (time = distance)."""
    return distance_matrix.copy()  # For now assuming speed = 1, so time = distance

def is_time_feasible(route: List[int], problem: Problem, time_matrix: np.ndarray) -> bool:
    """Check if route satisfies all time windows."""
    if not route:
        return True
        
    current_time = 0
    current_pos = 0  # depot
    
    for next_pos in route:
        # Add travel time to next customer
        travel_time = time_matrix[current_pos][next_pos]
        arrival_time = current_time + travel_time
        
        customer = problem.customers[next_pos - 1]  # -1 because customer ids start at 1
        
        # Wait if arrived too early
        current_time = max(arrival_time, customer.ready_time)
        
        # Check if arrived too late
        if current_time > customer.due_time:
            return False
            
        # Add service time
        current_time += customer.service_time
        current_pos = next_pos
        
    # Check return to depot
    final_time = current_time + time_matrix[current_pos][0]
    return final_time <= problem.depot.due_time