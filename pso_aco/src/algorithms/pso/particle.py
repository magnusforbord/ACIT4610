import numpy as np
from typing import List, Tuple, Set
from src.utils.data_loader import Problem

class Particle:
    def __init__(self, problem: Problem, distance_matrix: np.ndarray, time_matrix: np.ndarray):
        self.problem = problem
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix
        
        self.position: List[List[int]] = []
        self.velocity: List[float] = np.zeros(len(problem.customers))
        self.personal_best_position: List[List[int]] = []
        self.personal_best_cost = float('inf')
        
        self.initialize_position()

    def initialize_position(self):
        """Initialize solution ensuring all customers are assigned exactly once."""
        unassigned = set(range(1, len(self.problem.customers) + 1))
        self.position = []
        vehicles_used = 0
        
        while unassigned and vehicles_used < self.problem.vehicles:
            # Start new route
            route = []
            capacity = self.problem.capacity
            current_time = 0
            
            # Try to build route
            candidates = list(unassigned)
            np.random.shuffle(candidates)
            
            for customer_id in candidates[:]:
                customer = self.problem.customers[customer_id-1]
                
                # Check if customer can be added to route
                if customer.demand <= capacity:
                    # Check time feasibility
                    if not route:  # First in route
                        arrival_time = self.time_matrix[0][customer_id]
                    else:
                        arrival_time = current_time + self.time_matrix[route[-1]][customer_id]
                    
                    if arrival_time <= customer.due_time:
                        service_start = max(arrival_time, customer.ready_time)
                        service_end = service_start + customer.service_time
                        return_time = service_end + self.time_matrix[customer_id][0]
                        
                        if return_time <= self.problem.depot.due_time:
                            route.append(customer_id)
                            unassigned.remove(customer_id)
                            capacity -= customer.demand
                            current_time = service_end
            
            if route:  # Only add non-empty routes
                self.position.append(route)
                vehicles_used += 1
        
        # If there are still unassigned customers, try to insert them into existing routes
        if unassigned:
            for customer_id in list(unassigned):
                inserted = False
                # Try each route
                for route in self.position:
                    # Check capacity
                    route_demand = sum(self.problem.customers[c-1].demand for c in route)
                    if route_demand + self.problem.customers[customer_id-1].demand <= self.problem.capacity:
                        # Try each position
                        for pos in range(len(route) + 1):
                            new_route = route[:pos] + [customer_id] + route[pos:]
                            if self._is_route_feasible(new_route):
                                route[:] = new_route
                                unassigned.remove(customer_id)
                                inserted = True
                                break
                    if inserted:
                        break
                        
                if not inserted and vehicles_used < self.problem.vehicles:
                    # Create new route for this customer
                    self.position.append([customer_id])
                    unassigned.remove(customer_id)
                    vehicles_used += 1
        
        # Verify solution
        assigned = set()
        for route in self.position:
            for customer in route:
                if customer in assigned:
                    raise ValueError(f"Customer {customer} assigned multiple times")
                assigned.add(customer)
        
        if len(assigned) != len(self.problem.customers):
            missing = set(range(1, len(self.problem.customers) + 1)) - assigned
            raise ValueError(f"Missing customers: {missing}")
        
        print(f"Created solution with {len(self.position)} routes")
        
        # Set initial personal best quietly
        self.personal_best_position = [route.copy() for route in self.position]
        self.personal_best_cost = self._calculate_total_cost()
        
        # Improve initial solution without logging
        self._local_search()

    def update_velocity(self, global_best_position: List[List[int]], w: float, c1: float, c2: float):
        """Update particle's velocity."""
        current_seq = [c for route in self.position for c in route]
        personal_best_seq = [c for route in self.personal_best_position for c in route]
        global_best_seq = [c for route in global_best_position for c in route]
        
        inertia = w * self.velocity
        cognitive = c1 * np.random.random() * self._sequence_difference(personal_best_seq, current_seq)
        social = c2 * np.random.random() * self._sequence_difference(global_best_seq, current_seq)
        
        self.velocity = inertia + cognitive + social

    def update_position(self):
        """Update particle's position based on velocity."""
        if not self.position:
            self.initialize_position()
            return
            
        # Get customers to move
        moves = np.where(np.random.random(len(self.velocity)) < np.abs(self.velocity))[0]
        if not moves.size:
            return
            
        # Create new solution
        current_seq = [c for route in self.position for c in route]
        for idx in moves:
            customer = idx + 1
            if customer in current_seq:
                curr_pos = current_seq.index(customer)
                new_pos = (curr_pos + np.random.randint(-2, 3)) % len(current_seq)
                current_seq.pop(curr_pos)
                current_seq.insert(new_pos, customer)
        
        # Convert back to routes
        new_position = []
        route = []
        capacity = self.problem.capacity
        current_time = 0
        current_pos = 0
        
        for customer_id in current_seq:
            customer = self.problem.customers[customer_id-1]
            
            if (customer.demand <= capacity and 
                self._is_route_feasible(route + [customer_id])):
                route.append(customer_id)
                capacity -= customer.demand
                if len(route) == 1:
                    travel_time = self.time_matrix[0][customer_id]
                else:
                    travel_time = self.time_matrix[route[-2]][customer_id]
                current_time = max(current_time + travel_time, customer.ready_time) + customer.service_time
                current_pos = customer_id
            else:
                if route:
                    new_position.append(route)
                if len(new_position) < self.problem.vehicles:
                    route = [customer_id]
                    capacity = self.problem.capacity - customer.demand
                    current_time = max(self.time_matrix[0][customer_id], customer.ready_time) + customer.service_time
                    current_pos = customer_id
                else:
                    break
        
        if route and len(new_position) < self.problem.vehicles:
            new_position.append(route)
        
        if (new_position and 
            len(new_position) <= self.problem.vehicles and 
            sum(len(r) for r in new_position) == len(self.problem.customers) and
            all(self._is_route_feasible(r) for r in new_position)):
            self.position = new_position
            cost = self._calculate_total_cost()
            if cost < self.personal_best_cost:
                self.personal_best_position = [r.copy() for r in new_position]
                self.personal_best_cost = cost

    def _sequence_difference(self, seq1: List[int], seq2: List[int]) -> np.ndarray:
        """Calculate difference between two sequences."""
        diff = np.zeros(len(self.problem.customers))
        for i, customer in enumerate(seq1):
            if i >= len(seq2) or customer != seq2[i]:
                diff[customer-1] = 1
        return diff

    def _calculate_route_cost(self, route: List[int]) -> float:
        """Calculate total distance of a single route."""
        if not route:
            return 0
        cost = self.distance_matrix[0][route[0]]  # Depot to first
        for i in range(len(route) - 1):
            cost += self.distance_matrix[route[i]][route[i + 1]]
        cost += self.distance_matrix[route[-1]][0]  # Last back to depot
        return cost

    def _calculate_total_cost(self) -> float:
        """Calculate total distance of all routes."""
        return sum(self._calculate_route_cost(route) for route in self.position)

    def _is_route_feasible(self, route: List[int]) -> bool:
        """Check if route satisfies capacity and time window constraints."""
        if not route:
            return True
            
        # Check capacity
        total_demand = sum(self.problem.customers[c-1].demand for c in route)
        if total_demand > self.problem.capacity:
            return False
            
        # Check time windows
        current_time = 0
        current_pos = 0
        
        for customer_id in route:
            customer = self.problem.customers[customer_id-1]
            travel_time = self.time_matrix[current_pos][customer_id]
            arrival_time = current_time + travel_time
            
            if arrival_time > customer.due_time:
                return False
                
            service_start = max(arrival_time, customer.ready_time)
            current_time = service_start + customer.service_time
            current_pos = customer_id
            
        # Check return to depot
        return_time = current_time + self.time_matrix[current_pos][0]
        return return_time <= self.problem.depot.due_time

    def _local_search(self):
        """Apply local search improvements."""
        improved = True
        while improved:
            improved = False
            
            # Try intra-route improvements
            for route in self.position:
                # 2-opt
                for i in range(len(route)-1):
                    for j in range(i+2, len(route)):
                        new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
                        if (self._is_route_feasible(new_route) and 
                            self._calculate_route_cost(new_route) < self._calculate_route_cost(route)):
                            route[:] = new_route
                            improved = True
            
            # Try inter-route improvements
            for i in range(len(self.position)):
                for j in range(i+1, len(self.position)):
                    route1, route2 = self.position[i], self.position[j]
                    
                    # Try swapping customers
                    for pos1 in range(len(route1)):
                        for pos2 in range(len(route2)):
                            # Swap customers
                            new_route1 = route1[:pos1] + [route2[pos2]] + route1[pos1+1:]
                            new_route2 = route2[:pos2] + [route1[pos1]] + route2[pos2+1:]
                            
                            if (self._is_route_feasible(new_route1) and 
                                self._is_route_feasible(new_route2)):
                                old_cost = (self._calculate_route_cost(route1) + 
                                          self._calculate_route_cost(route2))
                                new_cost = (self._calculate_route_cost(new_route1) + 
                                          self._calculate_route_cost(new_route2))
                                
                                if new_cost < old_cost:
                                    self.position[i] = new_route1
                                    self.position[j] = new_route2
                                    improved = True