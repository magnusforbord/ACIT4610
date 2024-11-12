from typing import List
from src.algorithms.pso.swarm import Swarm
from src.algorithms.base import BaseOptimizer, Solution
import time

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
        """Apply local search focusing on shortest route"""
        if not routes:
            return routes

        # Create copies to avoid modifying original routes
        routes = [route.copy() for route in routes]
            
        # Find shortest non-empty route
        non_empty_routes = [(i, r) for i, r in enumerate(routes) if r]
        if not non_empty_routes:
            return routes
            
        shortest_route_idx, shortest_route = min(non_empty_routes, 
                                               key=lambda x: len(x[1]))
        
        # Store original route for restoration if needed
        original_routes = [route.copy() for route in routes]
        customers_to_move = shortest_route.copy()
        
        for customer in customers_to_move:
            inserted = False
            
            # Try each route except the shortest one
            for i in range(len(routes)):
                if i == shortest_route_idx:
                    continue
                    
                target_route = routes[i]
                
                # Try each position in the target route
                for pos in range(len(target_route) + 1):
                    # Create temporary route with customer inserted
                    temp_route = target_route[:pos] + [customer] + target_route[pos:]
                    
                    # Check feasibility
                    if self.is_route_feasible(temp_route):
                        # Update route
                        routes[i] = temp_route
                        shortest_route.remove(customer)
                        routes[shortest_route_idx] = shortest_route
                        inserted = True
                        break
                
                if inserted:
                    break
            
            if not inserted:
                # Restore original routes if we couldn't insert
                return original_routes
        
        # Remove empty routes
        return [route for route in routes if route]

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
        
        best_solution = None
        best_distance = float('inf')
        no_improvement = 0
        
        for iteration in range(max_iterations):
            print(f"Starting iteration {iteration}")
            
            # Update inertia weight
            w = self.swarm.w - (self.swarm.w - 0.4) * iteration / max_iterations
            self.swarm.w = max(0.4, w)
            
            try:
                self.swarm.optimize(1)
                
                current_distance = self.swarm.global_best_fitness
                current_routes = self.swarm.global_best_routes
                                
                if current_routes and len(current_routes) <= self.problem.vehicles:
                    if current_distance < best_distance:
                        best_distance = current_distance
                        best_solution = current_routes.copy()
                        no_improvement = 0
                        print(f"Iteration {iteration}: New best distance = {best_distance:.2f}, "
                            f"Routes = {len(current_routes)}, "
                            f"Vehicles used = {len(current_routes)}/{self.problem.vehicles}")
                
                # Apply local search periodically
                if iteration % 5 == 0 and best_solution:
                    print("Applying local search...")
                    improved_routes = self._apply_local_search(best_solution)
                    improved_distance = self.calculate_total_distance(improved_routes)
                    if improved_distance < best_distance:
                        best_distance = improved_distance
                        best_solution = improved_routes
                        print(f"Local search improved solution: {best_distance:.2f}")
                        
                        # Update swarm's global best with local search improvement
                        best_particle = min(self.swarm.particles, key=lambda p: p.best_fitness)
                        best_particle.best_routes = improved_routes.copy()
                        best_particle.best_fitness = improved_distance
                        self.swarm.global_best_routes = improved_routes.copy()
                        self.swarm.global_best_fitness = improved_distance

            except Exception as e:
                print(f"Error in iteration {iteration}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        end_time = time.time()
        print(f"\nOptimization completed in {end_time - start_time:.2f} seconds")
        print(f"Final best distance: {best_distance:.2f}")
        print(f"Number of routes: {len(best_solution) if best_solution else 0}")
        print(f"Vehicles used: {len(best_solution) if best_solution else 0}/{self.problem.vehicles}")
        #print("Routes:", [f"Vehicle {i+1}: {route}" for i, route in enumerate(current_routes)])
        return Solution(
            routes=best_solution if best_solution else [],
            total_distance=best_distance if best_solution else float('inf'),
            feasible=True if best_solution else False
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