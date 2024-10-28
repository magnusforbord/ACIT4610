import numpy as np
from problem_model import Customer

def euclidean_distance(customer_1: Customer, customer_2: Customer) -> float:
    """Calculate the Euclidean distance between two Customer objects."""
    return np.sqrt((customer_1.x_coord - customer_2.x_coord) ** 2 +
                   (customer_1.y_coord - customer_2.y_coord) ** 2)
