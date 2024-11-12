import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Customer:
    id: int
    x: float
    y: float
    demand: float
    ready_time: int
    due_time: int
    service_time: int

@dataclass
class Problem:
    vehicles: int
    capacity: float
    customers: List[Customer]
    depot: Customer

def load_solomon_instance(filepath: str) -> Problem:
    """Load Solomon VRPTW instance from file."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    # Skip header
    current_line = 4
    
    # Get vehicle info
    vehicle_info = lines[current_line].split()
    num_vehicles = int(vehicle_info[0])
    capacity = float(vehicle_info[1])
    
    # Skip to customer data
    current_line += 4
    
    customers = []
    for line in lines[current_line:]:
        data = line.strip().split()
        if not data:
            continue
            
        cust = Customer(
            id=int(data[0]),
            x=float(data[1]),
            y=float(data[2]),
            demand=float(data[3]),
            ready_time=int(data[4]),
            due_time=int(data[5]),
            service_time=int(data[6])
        )
        
        if cust.id == 0:
            depot = cust
        else:
            customers.append(cust)
    
    return Problem(
        vehicles=num_vehicles,
        capacity=capacity,
        customers=customers,
        depot=depot
    )