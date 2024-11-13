from typing import List, Tuple
from src.algorithms.pso.swarm import Swarm
from src.algorithms.base import BaseOptimizer, Solution
import time
import numpy as np

class PSOOptimizer(BaseOptimizer):
    def __init__(self,
                 problem,
                 distance_matrix,
                 time_matrix,
                 n_particles: int = 100,
                 w: float = 0.6,
                 c1: float = 1.8,
                 c2: float = 1.8):
        super().__init__(problem, distance_matrix, time_matrix)
        self.swarm = Swarm(
            problem=problem,
            n_particles=n_particles,
            distance_matrix=distance_matrix,
            time_matrix=time_matrix,
            w=w,
            c1=c1,
            c2=c2
        )

    def is_route_feasible(self, route: List[int]) -> bool:
        """Check if route satisfies all constraints"""
        if not route:
            return True
            
        # Check capacity
        total_demand = sum(self.problem.customers[c-1].demand for c in route)
        if total_demand > self.problem.capacity:
            return False
            
        # Check time windows
        current_time = 0
        current_pos = 0  # Start at depot
        
        for customer_id in route:
            customer = self.problem.customers[customer_id-1]
            
            # Add travel time
            travel_time = self.time_matrix[current_pos][customer_id]
            arrival_time = current_time + travel_time
            
            # Check if we arrived too late
            if arrival_time > customer.due_time:
                return False
                
            # Update current time (wait if arrived too early)
            service_start = max(arrival_time, customer.ready_time)
            current_time = service_start + customer.service_time
            current_pos = customer_id
            
        # Check return to depot
        final_time = current_time + self.time_matrix[current_pos][0]
        return final_time <= self.problem.depot.due_time

    def _apply_local_search(self, routes: List[List[int]]) -> List[List[int]]:
        if not routes:
            return routes
            
        # Only apply to a subset of routes each time
        n_routes_to_improve = min(5, len(routes))
        routes_to_improve = np.random.choice(len(routes), n_routes_to_improve, replace=False)
        
        for route_idx in routes_to_improve:
            # 2-opt improvement
            routes[route_idx] = self._apply_2opt(routes[route_idx])
            # Or-opt improvement 
            routes[route_idx] = self._apply_or_opt(routes[route_idx])
        
        return routes
    def _calculate_arrival_times(self, routes: List[List[int]]) -> List[List[float]]:
        """Calculate arrival times for all routes"""
        arrival_times = []
        for route in routes:
            times = []
            current_time = 0
            current_pos = 0
            
            for customer_id in route:
                customer = self.problem.customers[customer_id-1]
                travel_time = self.time_matrix[current_pos][customer_id]
                arrival_time = current_time + travel_time
                
                times.append(arrival_time)
                current_time = max(arrival_time, customer.ready_time) + customer.service_time
                current_pos = customer_id
                
            arrival_times.append(times)
        return arrival_times

    def _update_arrival_times(self, route: List[int]) -> List[float]:
        """Update arrival times for a single route"""
        times = []
        current_time = 0
        current_pos = 0
        
        for customer_id in route:
            customer = self.problem.customers[customer_id-1]
            travel_time = self.time_matrix[current_pos][customer_id]
            arrival_time = current_time + travel_time
            
            times.append(arrival_time)
            current_time = max(arrival_time, customer.ready_time) + customer.service_time
            current_pos = customer_id
            
        return times

    def _calculate_route_loads(self, routes: List[List[int]]) -> List[float]:
        """Calculate cumulative loads for all routes"""
        route_loads = []
        for route in routes:
            loads = [sum(self.problem.customers[c-1].demand for c in route[:i+1])
                    for i in range(len(route))]
            route_loads.append(loads)
        return route_loads

    def _update_capacity(self, route: List[int]) -> List[float]:
        """Update cumulative loads for a single route"""
        return [sum(self.problem.customers[c-1].demand for c in route[:i+1])
                for i in range(len(route))]

    def _is_insertion_feasible(self, 
                             customer: int,
                             route: List[int],
                             position: int,
                             arrival_times: List[float],
                             capacities: List[float]) -> bool:
        """Check if customer can feasibly be inserted at position"""
        # Check capacity
        customer_demand = self.problem.customers[customer-1].demand
        route_load = capacities[-1] if capacities else 0
        
        if route_load + customer_demand > self.problem.capacity:
            return False
            
        # Check time windows
        cust = self.problem.customers[customer-1]
        prev_pos = 0 if position == 0 else route[position-1]
        next_pos = 0 if position == len(route) else route[position]
        
        # Calculate arrival time at new customer
        if position == 0:
            arrival_time = self.time_matrix[0][customer]
        else:
            prev_customer = self.problem.customers[prev_pos-1]
            prev_departure = max(arrival_times[position-1], 
                               prev_customer.ready_time) + prev_customer.service_time
            arrival_time = prev_departure + self.time_matrix[prev_pos][customer]
            
        # Check if we arrive too late
        if arrival_time > cust.due_time:
            return False
            
        # Check if insertion delays subsequent customers too much
        departure_time = max(arrival_time, cust.ready_time) + cust.service_time
        next_arrival = departure_time + self.time_matrix[customer][next_pos]
        
        if position < len(route):
            next_customer = self.problem.customers[next_pos-1]
            if next_arrival > next_customer.due_time:
                return False
                
        return True
    

    def optimize(self, max_iterations: int) -> Solution:
        start_time = time.time()
        print(f"Starting PSO optimization with {max_iterations} iterations...")
        
        # Run multiple attempts
        n_attempts = 3
        best_solutions = []
        
        for attempt in range(n_attempts):
            print(f"\nAttempt {attempt + 1}/{n_attempts}")
            
            best_solution = None
            best_distance = float('inf')
            no_improvement = 0
            
            # Reset swarm for new attempt
            self.swarm = Swarm(
                problem=self.problem,
                n_particles=self.swarm.n_particles,
                distance_matrix=self.distance_matrix,
                time_matrix=self.time_matrix,
                w=0.95,
                c1=2.8,
                c2=1.2
            )
            
            for iteration in range(max_iterations):
                # Update inertia weight
                w = self.swarm.w - (self.swarm.w - 0.2) * (iteration / max_iterations) ** 0.8
                self.swarm.w = max(0.2, w)
                
                try:
                    self.swarm.optimize(1)
                    
                    current_distance = self.swarm.global_best_fitness
                    current_routes = self.swarm.global_best_routes
                    
                    if current_routes and len(current_routes) <= self.problem.vehicles:
                        if current_distance < best_distance:
                            best_distance = current_distance
                            best_solution = current_routes.copy()
                            no_improvement = 0
                            print(f"Iteration {iteration}: Distance = {best_distance:.2f}, Routes = {len(current_routes)}")
                    
                    # More frequent local search with intensity based on progress
                    if iteration % max(3, iteration//10) == 0 and best_solution:
                        improved_routes = self._apply_local_search(best_solution)
                        improved_distance = self.calculate_total_distance(improved_routes)
                        if improved_distance < best_distance:
                            best_distance = improved_distance
                            best_solution = improved_routes
                            print(f"Local search improved solution: {best_distance:.2f}")
                
                except Exception as e:
                    print(f"Error in iteration {iteration}: {str(e)}")
                    continue
            
            if best_solution:
                best_solutions.append((best_distance, best_solution))
        
        # Select best solution from all attempts
        if best_solutions:
            best_distance, best_solution = min(best_solutions, key=lambda x: x[0])
        else:
            best_distance, best_solution = float('inf'), []
        
        end_time = time.time()
        print(f"\nOptimization completed in {end_time - start_time:.2f} seconds")
        print(f"Best distance across attempts: {best_distance:.2f}")
        print(f"Routes: {len(best_solution) if best_solution else 0}/{self.problem.vehicles}")
        
        return Solution(
            routes=best_solution,
            total_distance=best_distance,
            feasible=bool(best_solution)
        )

    def calculate_total_distance(self, routes: List[List[int]]) -> float:
        """Calculate total distance for all routes"""
        if not routes:
            return float('inf')
            
        total_distance = 0
        for route in routes:
            if not route:
                continue
                
            # Add distance from depot to first customer
            total_distance += self.distance_matrix[0][route[0]]
            
            # Add distances between consecutive customers
            for i in range(len(route) - 1):
                total_distance += self.distance_matrix[route[i]][route[i + 1]]
                
            # Add distance from last customer back to depot
            total_distance += self.distance_matrix[route[-1]][0]
            
        return total_distance

    def calculate_route_distance(self, route: List[int]) -> float:
        """Calculate total distance for a single route"""
        if not route:
            return 0
            
        distance = self.distance_matrix[0][route[0]]  # Depot to first
        for i in range(len(route) - 1):
            distance += self.distance_matrix[route[i]][route[i + 1]]
        distance += self.distance_matrix[route[-1]][0]  # Last to depot
        return distance
    
    def _apply_2opt(self, route: List[int]) -> List[int]:
        """2-opt local search for single route"""
        if len(route) < 3:
            return route
            
        improved = True
        best_distance = self.calculate_route_distance(route)
        
        while improved:
            improved = False
            
            for i in range(len(route) - 1):
                for j in range(i + 2, len(route)):
                    # Create new route with 2-opt swap
                    new_route = route[:i+1] + route[j:i:-1] + route[j+1:]
                    
                    # Check feasibility and improvement
                    if self.is_route_feasible(new_route):
                        new_distance = self.calculate_route_distance(new_route)
                        if new_distance < best_distance:
                            route = new_route
                            best_distance = new_distance
                            improved = True
                            break
                if improved:
                    break
                    
        return route

    def _apply_or_opt(self, route: List[int]) -> List[int]:
        """Or-opt local search (relocate sequence of 1-3 customers)"""
        if len(route) < 2:
            return route
            
        improved = True
        best_distance = self.calculate_route_distance(route)
        
        while improved:
            improved = False
            
            # Try different sequence lengths
            for seq_length in range(1, min(4, len(route))):
                # Try each possible sequence
                for i in range(len(route) - seq_length + 1):
                    sequence = route[i:i+seq_length]
                    remaining = route[:i] + route[i+seq_length:]
                    
                    # Try inserting sequence at each position
                    for j in range(len(remaining) + 1):
                        new_route = remaining[:j] + sequence + remaining[j:]
                        
                        if self.is_route_feasible(new_route):
                            new_distance = self.calculate_route_distance(new_route)
                            if new_distance < best_distance:
                                route = new_route
                                best_distance = new_distance
                                improved = True
                                break
                                
                    if improved:
                        break
                if improved:
                    break
                    
        return route

    def _cross_exchange(self, route1: List[int], route2: List[int]) -> Tuple[List[int], List[int]]:
        """Cross-exchange between two routes"""
        if not route1 or not route2:
            return route1, route2
            
        best_route1, best_route2 = None, None
        best_total = self.calculate_route_distance(route1) + self.calculate_route_distance(route2)
        
        for i in range(len(route1)):
            for j in range(len(route2)):
                # Try swapping single customers
                new_route1 = route1[:i] + [route2[j]] + route1[i+1:]
                new_route2 = route2[:j] + [route1[i]] + route2[j+1:]
                
                if (self.is_route_feasible(new_route1) and 
                    self.is_route_feasible(new_route2)):
                    
                    total = self.calculate_route_distance(new_route1) + \
                            self.calculate_route_distance(new_route2)
                            
                    if total < best_total:
                        best_route1, best_route2 = new_route1, new_route2
                        best_total = total
        
        return best_route1 or route1, best_route2 or route2

    def _cross_operator(self, route1: List[int], route2: List[int]) -> Tuple[List[int], List[int]]:
        """CROSS operator (swap route segments)"""
        if len(route1) < 2 or len(route2) < 2:
            return route1, route2
            
        best_route1, best_route2 = None, None
        best_total = self.calculate_route_distance(route1) + self.calculate_route_distance(route2)
        
        # Try different segment lengths
        max_length = min(len(route1), len(route2), 3)  # Limit segment length to 3
        
        for length in range(1, max_length + 1):
            for i in range(len(route1) - length + 1):
                for j in range(len(route2) - length + 1):
                    # Extract and swap segments
                    seg1 = route1[i:i+length]
                    seg2 = route2[j:j+length]
                    
                    new_route1 = route1[:i] + seg2 + route1[i+length:]
                    new_route2 = route2[:j] + seg1 + route2[j+length:]
                    
                    if (self.is_route_feasible(new_route1) and 
                        self.is_route_feasible(new_route2)):
                        
                        total = self.calculate_route_distance(new_route1) + \
                                self.calculate_route_distance(new_route2)
                                
                        if total < best_total:
                            best_route1, best_route2 = new_route1, new_route2
                            best_total = total
        
        return best_route1 or route1, best_route2 or route2