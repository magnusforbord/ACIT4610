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

    def _decode_position(self) -> List[List[int]]:
        routes: List[List[int]] = []
        available_vehicles = self.problem.vehicles
        unassigned = set(range(1, len(self.problem.customers) + 1))
        
        # Create clusters based on position values and geography
        customers = [(i+1, 
                    self.position[i],
                    self.problem.customers[i].x,
                    self.problem.customers[i].y,
                    self.problem.customers[i].due_time) 
                    for i in range(len(self.position))]
        
        # Calculate distance from depot for each customer
        depot_x, depot_y = self.problem.depot.x, self.problem.depot.y
        for i in range(len(customers)):
            dist_to_depot = np.sqrt((customers[i][2] - depot_x)**2 + 
                                (customers[i][3] - depot_y)**2)
            # Calculate angle from depot
            angle = np.arctan2(customers[i][3] - depot_y, 
                            customers[i][2] - depot_x)
            customers[i] = (*customers[i], dist_to_depot, angle)
        
        # Sort by position value, angle, and distance
        sorted_customers = sorted(customers, 
                                key=lambda x: (-x[1],  # Position value
                                            x[6],     # Angle from depot
                                            x[5]))    # Distance from depot
        sorted_customers = [x[0] for x in sorted_customers]
        
        while unassigned and available_vehicles > 0:
            current_route: List[int] = []
            current_capacity = self.problem.capacity
            current_time = 0
            current_pos = 0
            
            # Start with first unassigned customer
            for start_customer in sorted_customers:
                if start_customer in unassigned:
                    current_route.append(start_customer)
                    unassigned.remove(start_customer)
                    customer = self.problem.customers[start_customer-1]
                    current_capacity -= customer.demand
                    travel_time = self.time_matrix[current_pos][start_customer]
                    current_time = max(current_time + travel_time, 
                                    customer.ready_time) + customer.service_time
                    current_pos = start_customer
                    break
            
            # Add nearest feasible neighbors
            while True:
                best_next = None
                best_score = float('inf')
                
                for customer_id in unassigned:
                    customer = self.problem.customers[customer_id-1]
                    travel_time = self.time_matrix[current_pos][customer_id]
                    arrival_time = current_time + travel_time
                    
                    if (arrival_time <= customer.due_time and 
                        customer.demand <= current_capacity):
                        
                        # Score combines distance, time window, and angle
                        dist = self.distance_matrix[current_pos][customer_id]
                        time_slack = customer.due_time - arrival_time
                        
                        # Calculate angle difference
                        curr_x = self.problem.customers[current_pos-1].x
                        curr_y = self.problem.customers[current_pos-1].y
                        next_x = customer.x
                        next_y = customer.y
                        angle_diff = abs(np.arctan2(next_y - curr_y, 
                                                next_x - curr_x) - 
                                    np.arctan2(curr_y - depot_y, 
                                                curr_x - depot_x))
                        
                        score = (dist + 
                                0.3 * time_slack + 
                                50 * angle_diff)  # Weight angle difference heavily
                        
                        if score < best_score:
                            best_score = score
                            best_next = customer_id
                
                if best_next is None:
                    break
                    
                current_route.append(best_next)
                unassigned.remove(best_next)
                customer = self.problem.customers[best_next-1]
                current_capacity -= customer.demand
                travel_time = self.time_matrix[current_pos][best_next]
                current_time = max(current_time + travel_time, 
                                customer.ready_time) + customer.service_time
                current_pos = best_next
            
            if current_route:
                routes.append(current_route)
                available_vehicles -= 1
            else:
                break
                    
        return routes

    def update_velocity(self, w: float, c1: float, c2: float, global_best_position: np.ndarray):
        r1, r2 = np.random.random(2)
        
        # Calculate distance to personal and global best
        personal_distance = np.linalg.norm(self.best_position - self.position)
        global_distance = np.linalg.norm(global_best_position - self.position)
        
        # Adapt cognitive and social factors
        adapted_c1 = c1 + (2.5 - c1) * (personal_distance / (personal_distance + global_distance))
        adapted_c2 = c2 + (2.5 - c2) * (global_distance / (personal_distance + global_distance))
        
        # Calculate components
        inertia = w * self.velocity
        cognitive = adapted_c1 * r1 * (self.best_position - self.position)
        social = adapted_c2 * r2 * (global_best_position - self.position)
        
        # Add diversity factor based on best fitness
        if self.best_fitness > 1e-5:  # If we have a valid solution
            diversity = np.random.uniform(-0.1, 0.1, size=len(self.velocity))
            self.velocity = inertia + cognitive + social + diversity * (1.0 - w)  # Scale with inertia
        else:
            self.velocity = inertia + cognitive + social
            
        # Dynamic velocity clamping
        v_max = 0.5 * (1.0 - w)  # Decrease as inertia decreases
        self.velocity = np.clip(self.velocity, -v_max, v_max)
    def update_position(self):
        # Chaotic position update
        chaos_factor = 1 / (1 + np.exp(-self.velocity))  # Sigmoid function
        self.position = self.position + chaos_factor * self.velocity
        
        # Apply quantum-inspired position adjustment
        if np.random.random() < 0.1:  # 10% chance
            delta = np.random.uniform(-0.2, 0.2, size=len(self.position))
            mask = np.random.random(size=len(self.position)) < 0.3  # 30% of dimensions
            self.position[mask] += delta[mask]
        
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
            
        fitness = total_distance + penalty
        
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_position = self.position.copy()
            self.best_routes = routes
            
        return fitness