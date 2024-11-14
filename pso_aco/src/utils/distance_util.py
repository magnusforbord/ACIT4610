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
    return distance_matrix.copy()  