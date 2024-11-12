from typing import List, Tuple
import numpy as np
from src.utils.data_loader import Problem

class Particle:
    def __init__(self, problem: Problem, distance_matrix: np.ndarray, time_matrix: np.ndarray):
        self.problem = problem
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix
        self.position = self._initialize_position()
        self.velocity = np.random.uniform(-0.4, 0.4, size=len(problem.customers))
        self.best_position = self.position.copy()
        self.best_routes = []
        self.best_fitness = float('inf')
        self.fitness = float('inf')  # Add this line
        
    def _initialize_position(self) -> np.ndarray:
        """Initialize particle position using modified nearest neighbor heuristic"""
        position = np.zeros(len(self.problem.customers))
        unvisited = set(range(len(self.problem.customers)))
        current = 0  # Start at depot
        current_time = 0
        current_capacity = self.problem.capacity
        
        for i in range(len(self.problem.customers)):
            if not unvisited:
                break
                
            best_next = None
            best_score = float('inf')
            
            for next_cust in unvisited:
                customer = self.problem.customers[next_cust]
                travel_time = self.time_matrix[current][next_cust + 1]
                arrival_time = current_time + travel_time
                
                if (arrival_time <= customer.due_time and 
                    customer.demand <= current_capacity):
                    
                    # Score combines distance, time window urgency, and demand
                    time_slack = customer.due_time - arrival_time
                    demand_ratio = customer.demand / self.problem.capacity
                    
                    score = (
                        self.distance_matrix[current][next_cust + 1] +
                        0.3 * time_slack +
                        0.2 * demand_ratio
                    )
                    
                    if score < best_score:
                        best_score = score
                        best_next = next_cust
            
            if best_next is not None:
                position[best_next] = 1.0 - (i / len(self.problem.customers))
                unvisited.remove(best_next)
                current = best_next + 1
                current_time = max(
                    current_time + self.time_matrix[current-1][current],
                    self.problem.customers[best_next].ready_time
                ) + self.problem.customers[best_next].service_time
                current_capacity -= self.problem.customers[best_next].demand
            else:
                # Assign remaining customers randomly
                remaining = list(unvisited)
                random_priorities = np.random.uniform(0, 0.5, size=len(remaining))
                for cust, pri in zip(remaining, random_priorities):
                    position[cust] = pri
                break
        
        # Add small random noise
        position += np.random.uniform(-0.05, 0.05, size=len(position))
        return np.clip(position, 0, 1)
        
    def _nearest_neighbor_init(self) -> np.ndarray:
        position = np.zeros(len(self.problem.customers))
        maxVal = float('inf')
        unvisited = set(range(len(self.problem.customers)))
        current = 0  # depot
        current_time = 0
        
        # Initialize heuristic scores
        heuristic = np.zeros(len(self.problem.customers))
        
        while unvisited:
            for city in unvisited:
                travel_time = self.time_matrix[current][city + 1]
                arrival_time = current_time + travel_time
                wait_time = max(self.problem.customers[city].ready_time - arrival_time, 0)
                
                # Combine multiple factors in scoring
                heuristic[city] = (
                    travel_time + 
                    wait_time + 
                    (self.problem.customers[city].due_time - arrival_time) * 0.3 +
                    (self.problem.customers[city].demand / self.problem.capacity) * 50
                )
            
            nearest = min((h, i) for i, h in enumerate(heuristic) 
                        if i in unvisited)[1]
            
            # Assign position value based on order and urgency
            due_time_factor = 1.0 - (self.problem.customers[nearest].due_time / 
                                    max(c.due_time for c in self.problem.customers))
            position[nearest] = 1.0 - (len(position) - len(unvisited)) / len(position)
            position[nearest] *= (1 + 0.2 * due_time_factor)  # Adjust by due time
            
            unvisited.remove(nearest)
            current = nearest + 1
            current_time = max(
                current_time + self.time_matrix[current-1][current],
                self.problem.customers[nearest].ready_time
            ) + self.problem.customers[nearest].service_time
            heuristic[nearest] = maxVal
            
        return np.clip(position, 0, 1)

    def _local_search(self, routes: List[List[int]]) -> List[List[int]]:
        if not routes:
            return routes
            
        improved = True
        while improved:
            improved = False
            
            # 1. Try moving single customers between routes
            for i in range(len(routes)):
                for j in range(len(routes)):
                    if i == j:
                        continue
                        
                    route1, route2 = routes[i], routes[j]
                    
                    # Try moving each customer
                    for pos1 in range(len(route1)):
                        customer = route1[pos1]
                        
                        # Try all possible positions in route2
                        best_pos = None
                        best_cost = float('inf')
                        
                        for pos2 in range(len(route2) + 1):
                            new_route2 = route2[:pos2] + [customer] + route2[pos2:]
                            new_route1 = route1[:pos1] + route1[pos1+1:]
                            
                            if (self._is_time_feasible(new_route1) and 
                                self._is_time_feasible(new_route2)):
                                
                                old_cost = (self._calculate_route_cost(route1) + 
                                        self._calculate_route_cost(route2))
                                new_cost = (self._calculate_route_cost(new_route1) + 
                                        self._calculate_route_cost(new_route2))
                                
                                if new_cost < best_cost:
                                    best_cost = new_cost
                                    best_pos = pos2
                        
                        if best_pos is not None:
                            route2.insert(best_pos, customer)
                            route1.pop(pos1)
                            improved = True
                            break
                    
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                # 2. Try swapping customers between routes
                for i in range(len(routes)):
                    for j in range(i + 1, len(routes)):
                        route1, route2 = routes[i], routes[j]
                        
                        for pos1 in range(len(route1)):
                            for pos2 in range(len(route2)):
                                # Try swapping
                                cust1, cust2 = route1[pos1], route2[pos2]
                                new_route1 = route1[:pos1] + [cust2] + route1[pos1+1:]
                                new_route2 = route2[:pos2] + [cust1] + route2[pos2+1:]
                                
                                if (self._is_time_feasible(new_route1) and 
                                    self._is_time_feasible(new_route2)):
                                    
                                    old_cost = (self._calculate_route_cost(route1) + 
                                            self._calculate_route_cost(route2))
                                    new_cost = (self._calculate_route_cost(new_route1) + 
                                            self._calculate_route_cost(new_route2))
                                    
                                    if new_cost < old_cost:
                                        route1[pos1], route2[pos2] = route2[pos2], route1[pos1]
                                        improved = True
                                        break
                        
                            if improved:
                                break
                        if improved:
                            break
        
        return [r for r in routes if r]  # Remove empty routes
    
    def _calculate_route_cost(self, route: List[int]) -> float:
        """Calculate total distance of route"""
        if not route:
            return 0
        cost = self.distance_matrix[0][route[0]]
        for i in range(len(route)-1):
            cost += self.distance_matrix[route[i]][route[i+1]]
        cost += self.distance_matrix[route[-1]][0]
        return cost


    def _is_time_feasible(self, route: List[int]) -> bool:
        if not route:
            return True
            
        current_time = 0
        current_pos = 0
        
        for customer_id in route:
            customer = self.problem.customers[customer_id-1]
            travel_time = self.time_matrix[current_pos][customer_id]
            arrival_time = current_time + travel_time
            
            if arrival_time > customer.due_time:
                return False
                
            current_time = max(arrival_time, customer.ready_time) + customer.service_time
            current_pos = customer_id
            
        # Check return to depot
        final_time = current_time + self.time_matrix[current_pos][0]
        return final_time <= self.problem.depot.due_time

    def _can_return_to_depot(self, current_pos: int, current_time: float) -> bool:
        return (current_time + self.time_matrix[current_pos][0] <= self.problem.depot.due_time)

    def _decode_position(self):
        """Decode particle position into routes with time window clustering"""
        n_customers = len(self.problem.customers)
        routes = []
        
        # Group customers by time window similarity
        customers = [(i+1, 
                    self.position[i], 
                    self.problem.customers[i].ready_time,
                    self.problem.customers[i].due_time,
                    self.problem.customers[i].demand) for i in range(n_customers)]
        
        # Sort by position value and time windows
        customers.sort(key=lambda x: (x[1], x[2]))  # Sort by position value and ready time
        
        current_route = []
        current_load = 0
        current_time = 0
        current_pos = 0
        
        while customers and len(routes) < self.problem.vehicles:
            found = False
            best_insertion = None
            best_cost = float('inf')
            
            # Try each remaining customer
            for i, (cust_id, _, ready_time, due_time, demand) in enumerate(customers):
                if current_load + demand > self.problem.capacity:
                    continue
                    
                # Calculate insertion cost
                travel_time = self.time_matrix[current_pos][cust_id]
                arrival_time = current_time + travel_time
                wait_time = max(0, ready_time - arrival_time)
                
                if arrival_time + wait_time <= due_time:
                    # Score based on distance and time window compatibility
                    cost = (travel_time + wait_time + 
                        0.5 * abs(current_time - ready_time))
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_insertion = (i, cust_id, ready_time, due_time, demand)
                        found = True
            
            if found:
                idx, cust_id, ready_time, due_time, demand = best_insertion
                current_route.append(cust_id)
                current_load += demand
                
                travel_time = self.time_matrix[current_pos][cust_id]
                current_time = max(current_time + travel_time, ready_time) + \
                            self.problem.customers[cust_id-1].service_time
                current_pos = cust_id
                
                customers.pop(idx)
            else:
                # Start new route if current one is not empty
                if current_route:
                    routes.append(current_route)
                    current_route = []
                    current_load = 0
                    current_time = 0
                    current_pos = 0
                else:
                    # If we can't start a new route, break
                    break
        
        # Add last route if not empty
        if current_route:
            routes.append(current_route)
        
        # Try to insert remaining customers into existing routes
        if customers:
            for cust_id, _, _, _, _ in customers:
                self._insert_into_best_route(cust_id, routes)
        
        return routes
    
    def update_velocity(self, w: float, c1: float, c2: float, global_best_position: np.ndarray):
        r1, r2 = np.random.random(2)
        
        # Constriction coefficient
        phi = c1 + c2
        chi = 2.0 / (abs(2.0 - phi - np.sqrt(phi * phi - 4.0 * phi)))
        
        # Calculate velocity components with constriction
        inertia = chi * w * self.velocity
        cognitive = chi * c1 * r1 * (self.best_position - self.position)
        social = chi * c2 * r2 * (global_best_position - self.position)
        
        # Add perturbation to escape local optima
        if np.random.random() < 0.1:  # 10% chance
            perturbation = np.random.normal(0, 0.1, size=len(self.velocity))
            self.velocity = inertia + cognitive + social + perturbation
        else:
            self.velocity = inertia + cognitive + social
            
        # Dynamic velocity clamping
        v_max = 0.5 * (1.0 - w)
        self.velocity = np.clip(self.velocity, -v_max, v_max)

    def update_position(self):
        # Simple position update
        self.position = self.position + self.velocity
        self.position = np.clip(self.position, 0, 1)

    def evaluate(self) -> float:
        routes = self._decode_position()
        if not routes:
            return float('inf')
            
        if len(routes) > self.problem.vehicles:
            return float('inf')
            
        # Calculate total distance
        total_distance = 0
        for route in routes:
            if not route:  # Skip empty routes
                continue
            prev = 0
            for customer in route:
                total_distance += self.distance_matrix[prev][customer]
                prev = customer
            total_distance += self.distance_matrix[prev][0]
        
        # Add penalties
        penalty = 0
        all_customers = set(range(1, len(self.problem.customers) + 1))
        served_customers = set(customer for route in routes for customer in route)
        unserved = len(all_customers - served_customers)
        
        if unserved > 0:
            penalty += unserved * 1000
            
        self.fitness = total_distance + penalty  # Store current fitness
        
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position.copy()
            self.best_routes = routes
            
        return self.fitness
    
    def _can_serve_customer(self, customer_id: int, current_route: list, current_time: float) -> bool:
        """Check if a customer can be served given current route and time"""
        if not current_route:
            # First customer in route - check direct from depot
            travel_time = self.time_matrix[0][customer_id]
            arrival_time = current_time + travel_time
            customer = self.problem.customers[customer_id-1]
            
            # Check if we can arrive before due time
            if arrival_time > customer.due_time:
                return False
                
            # Check if we can return to depot after service
            service_start = max(arrival_time, customer.ready_time)
            service_end = service_start + customer.service_time
            return_time = service_end + self.time_matrix[customer_id][0]
            
            return return_time <= self.problem.depot.due_time
        else:
            # Check from last customer in current route
            last_customer = current_route[-1]
            travel_time = self.time_matrix[last_customer][customer_id]
            arrival_time = current_time + travel_time
            customer = self.problem.customers[customer_id-1]
            
            # Check if we can arrive before due time
            if arrival_time > customer.due_time:
                return False
                
            # Check if we can return to depot after service
            service_start = max(arrival_time, customer.ready_time)
            service_end = service_start + customer.service_time
            return_time = service_end + self.time_matrix[customer_id][0]
            
            return return_time <= self.problem.depot.due_time
        

    def _insert_into_best_route(self, customer_id: int, routes: List[List[int]]) -> bool:
        """Try to insert a customer into the best position in any route"""
        best_route_idx = -1
        best_pos = -1
        best_cost_increase = float('inf')
        
        for r_idx, route in enumerate(routes):
            # Check capacity
            route_load = sum(self.problem.customers[c-1].demand for c in route)
            if route_load + self.problem.customers[customer_id-1].demand > self.problem.capacity:
                continue
                
            # Try each position
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [customer_id] + route[pos:]
                if self._is_time_feasible(new_route):
                    old_cost = self.calculate_route_distance(route)
                    new_cost = self.calculate_route_distance(new_route)
                    cost_increase = new_cost - old_cost
                    
                    if cost_increase < best_cost_increase:
                        best_cost_increase = cost_increase
                        best_route_idx = r_idx
                        best_pos = pos
        
        if best_route_idx >= 0:
            routes[best_route_idx].insert(best_pos, customer_id)
            return True
        return False