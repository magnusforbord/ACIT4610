from typing import List, Dict
from src.utils.data_loader import Problem
import numpy as np

def validate_solution(routes: List[List[int]], problem: Problem, time_matrix: np.ndarray) -> Dict[str, bool]:
    validation = {
        'all_customers_visited': True,
        'single_visit_per_customer': True,
        'capacity_constraints': True,
        'time_windows': True,
        'vehicle_limit': True,
        'depot_time_window': True
    }
    
    # Check vehicle limit
    if len(routes) > problem.vehicles:
        validation['vehicle_limit'] = False
        print(f"Vehicle limit exceeded: {len(routes)} > {problem.vehicles}")
    
    # Check all customers visited exactly once
    visited = set()
    for route in routes:
        for customer in route:
            if customer in visited:
                validation['single_visit_per_customer'] = False
                print(f"Customer {customer} visited multiple times")
            visited.add(customer)
    
    if len(visited) != len(problem.customers):
        validation['all_customers_visited'] = False
        missing = set(range(1, len(problem.customers) + 1)) - visited
        print(f"Missing customers: {missing}")
    
    # Check each route
    for i, route in enumerate(routes):
        # Check capacity
        total_demand = sum(problem.customers[c-1].demand for c in route)
        if total_demand > problem.capacity:
            validation['capacity_constraints'] = False
            print(f"Route {i} exceeds capacity: {total_demand} > {problem.capacity}")
        
        # Check time windows
        current_time = 0
        current_pos = 0  # depot
        
        for customer_id in route:
            customer = problem.customers[customer_id-1]
            travel_time = time_matrix[current_pos][customer_id]
            arrival_time = current_time + travel_time
            
            if arrival_time > customer.due_time:
                validation['time_windows'] = False
                print(f"Time window violated for customer {customer_id}:")
                print(f"Arrived at {arrival_time}, due time was {customer.due_time}")
            
            service_start = max(arrival_time, customer.ready_time)
            current_time = service_start + customer.service_time
            current_pos = customer_id
        
        # Check return to depot
        final_travel_time = time_matrix[current_pos][0]
        final_arrival = current_time + final_travel_time
        
        if final_arrival > problem.depot.due_time:
            validation['depot_time_window'] = False
            print(f"Route {i} returns to depot too late: {final_arrival} > {problem.depot.due_time}")
    
    return validation