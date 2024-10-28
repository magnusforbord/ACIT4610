import random 
import numpy as np 
from distance_calc import euclidean_distance
import matplotlib.pyplot as plt

class AntColonyOptimizer:
    def __init__(self, vehicles, customers, depot, num_ants, num_iterations, alpha=1, beta=1, rho=0.5, Q=100):
        self.vehicles = vehicles  # List of Vehicle instances
        self.vehicle_count = len(vehicles)  # Total number of vehicles
        self.vehicle_capacity = vehicles[0].capacity if vehicles else 0  # Assume all vehicles have the same capacity
        self.customers = customers  # List of Customer instances (excluding depot)
        self.depot = depot  # Separate Depot instance
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q

        self.distance_matrix = self.calculate_distance_matrix()
        self.pheromone_matrix = np.ones((len(self.customers) + 1, len(self.customers) + 1))  # +1 to include depot
        self.best_routes = []
        self.best_distance = float('inf')

        self.history = []
        self.pheromone_matrix_history = []
        self.iteration_penalty_multiplier = 1

    def calculate_distance_matrix(self):
        num_customers = len(self.customers) + 1  # Including depot
        distance_matrix = np.zeros((num_customers, num_customers))

        # Calculate distances between depot and each customer
        for j, customer in enumerate([self.depot] + self.customers):
            for k, other_customer in enumerate([self.depot] + self.customers):
                distance_matrix[j][k] = euclidean_distance(customer, other_customer)
        
        return distance_matrix
    
    def heuristic_info(self, current_customer, next_customer, current_time):
        distance = self.distance_matrix[current_customer][next_customer]
        if distance == 0:
            distance = 1e-10  # Avoid division by zero

        distance_heuristic = 1 / distance

        customer = self.customers[next_customer - 1]
        time_window = customer.due_date - customer.ready_time
        time_remaining = customer.due_date - current_time

        if time_remaining <= 0:
            time_heuristic = 0
        else:
            time_heuristic = (time_remaining / time_window) ** 2  # Exponentially increase urgency

        return distance_heuristic * time_heuristic

        # Combine heuristics: higher priority to closer customers with tight time windows
        return distance_heuristic * time_heuristic
    def probability(self, current_customer, next_customer, current_time):
        pheromone = self.pheromone_matrix[current_customer][next_customer] ** self.alpha
        heuristic = self.heuristic_info(current_customer, next_customer, current_time) ** self.beta
        return pheromone * heuristic
    
    def solution(self):
        routes = [[] for _ in range(self.vehicle_count)]
        visited = set()
        remaining_customers = set(range(1, len(self.customers) + 1))

        for vehicle_index in range(self.vehicle_count):
            current_customer = 0  # Start at depot
            current_capacity = 0
            current_time = 0
            routes[vehicle_index].append(current_customer)

            while remaining_customers:
                feasible_customers = [
                    j for j in remaining_customers 
                    if (current_capacity + self.customers[j - 1].demand <= self.vehicles[vehicle_index].capacity)
                ]
                
                if not feasible_customers:
                    # Return to depot and move to next vehicle if there are remaining customers
                    routes[vehicle_index].append(0)
                    if remaining_customers and vehicle_index + 1 < self.vehicle_count:
                        vehicle_index += 1
                        routes[vehicle_index].append(0)
                        current_capacity = 0  # Reset capacity for new vehicle
                    break

                probabilities = [
                    self.probability(current_customer, j, current_time) if j in feasible_customers else 0
                    for j in range(1, len(self.customers) + 1)
                ]
                total_prob = sum(probabilities)

                if total_prob == 0:
                    break

                probabilities = [p / total_prob for p in probabilities]
                next_customer = np.random.choice(range(1, len(self.customers) + 1), p=probabilities)

                # Calculate arrival time and update current time and capacity
                travel_time = self.distance_matrix[current_customer][next_customer]
                arrival_time = current_time + travel_time
                current_time = max(arrival_time, self.customers[next_customer - 1].ready_time) + self.customers[next_customer - 1].service_time

                routes[vehicle_index].append(next_customer)
                visited.add(next_customer)
                remaining_customers.remove(next_customer)
                current_capacity += self.customers[next_customer - 1].demand
                current_customer = next_customer

                if current_capacity >= self.vehicles[vehicle_index].capacity:
                    break

            routes[vehicle_index].append(0)  # Return to depot
            if not remaining_customers:
                break

        return routes
    
    def update_pheromones(self, solutions):
        self.pheromone_matrix *= (1 - self.rho)

        # Sort solutions by total distance and take the top 10% best solutions
        solutions = sorted(solutions, key=lambda x: x[1])
        top_solutions = solutions[:max(1, len(solutions) // 10)]  # Top 10%

        for routes, total_distance in top_solutions:
            if total_distance > 0:
                pheromone_deposit = self.Q / total_distance
                for route in routes:
                    for i in range(len(route) - 1):
                        self.pheromone_matrix[route[i]][route[i + 1]] += pheromone_deposit
        self.pheromone_matrix_history.append(np.copy(self.pheromone_matrix))  # For animation

    def evaluate_routes(self, routes):
        total_distance = 0
        penalty_multiplier = 5000  # You can adjust this value

        for route in routes:
            route_distance = 0
            current_time = 0
            current_capacity = 0

            for i in range(len(route) - 1):
                customer_id = route[i]
                next_customer_id = route[i + 1]

                if customer_id == 0:
                    customer = self.depot
                else:
                    customer = self.customers[customer_id - 1]

                if next_customer_id == 0:
                    next_customer = self.depot
                else:
                    next_customer = self.customers[next_customer_id - 1]

                travel_time = self.distance_matrix[customer_id][next_customer_id]
                arrival_time = current_time + travel_time

                # Only apply penalties for customers (not depot)
                if next_customer_id != 0:
                    # Early arrival penalty
                    if arrival_time < next_customer.ready_time:
                        waiting_time = next_customer.ready_time - arrival_time
                        route_distance += waiting_time * penalty_multiplier
                        current_time = next_customer.ready_time + next_customer.service_time
                    else:
                        current_time = arrival_time + next_customer.service_time

                    # Late arrival penalty
                    if arrival_time > next_customer.due_date:
                        late_penalty = (arrival_time - next_customer.due_date) * penalty_multiplier
                        route_distance += late_penalty

                    # Capacity check
                    current_capacity += next_customer.demand
                    if current_capacity > self.vehicle_capacity:
                        capacity_penalty = (current_capacity - self.vehicle_capacity) * penalty_multiplier
                        route_distance += capacity_penalty
                        current_capacity = self.vehicle_capacity  # Cap capacity

                else:
                    current_time = arrival_time  # Update time when returning to depot

                route_distance += travel_time  # Add travel distance

            total_distance += route_distance

        return total_distance
            
    def optimize(self):
        for iteration in range(self.num_iterations):
            self.iteration_penalty_multiplier = 1 + iteration / self.num_iterations  # Dynamic penalty

            all_solutions = []
            for ant in range(self.num_ants):
                routes = self.solution()
                total_distance = self.evaluate_routes(routes)

                if total_distance < self.best_distance:
                    self.best_distance = total_distance
                    self.best_routes = routes.copy()

                all_solutions.append((routes, total_distance))
            
            self.update_pheromones(all_solutions)
            self.history.append((self.best_routes, self.best_distance))
            print(f"Iteration {iteration + 1}: Best Distance: {self.best_distance}")

    def plot_convergence(self):
        distances = [dist for _, dist in self.history]
        plt.plot(distances, marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('Best Distance')
        plt.title('Convergence Plot')
        plt.show()

    def validate_solution(self):
        """Verify solution constraints and penalties."""
        for vehicle_index, route in enumerate(self.best_routes):
            current_time = 0
            current_capacity = 0
            total_route_distance = 0
            penalties = {
                'capacity_violations': 0,
                'late_delivery_violations': 0,
                'early_delivery_penalties': 0
            }

            print(f"\nVehicle {vehicle_index + 1} Route: {route}")

            for i in range(len(route) - 1):
                customer_id = route[i]

                # Determine if current point is the depot or a customer
                if customer_id == 0:
                    customer = self.depot  # Use depot if ID is 0
                else:
                    customer = self.customers[customer_id - 1]  # Adjust for zero-based indexing

                # Calculate travel time from previous location
                travel_time = self.distance_matrix[route[i]][route[i + 1]]
                current_time += travel_time

                # Only check penalties and capacity for customers (not depot)
                if customer_id != 0:
                    # Check for early delivery penalty
                    if current_time < customer.ready_time:
                        penalties['early_delivery_penalties'] += customer.ready_time - current_time
                        current_time = customer.ready_time  # Update to wait until ready time

                    # Check for late delivery penalty
                    if current_time > customer.due_date:
                        penalties['late_delivery_violations'] += current_time - customer.due_date

                    # Update time with service time
                    current_time += customer.service_time
                    current_capacity += customer.demand

                    # Check for capacity violation
                    if current_capacity > self.vehicle_capacity:
                        penalties['capacity_violations'] += current_capacity - self.vehicle_capacity
                        current_capacity = self.vehicle_capacity  # Cap at vehicle's max capacity

                # Add travel distance
                total_route_distance += travel_time

            print(f"Total Route Distance: {total_route_distance}")
            print("Penalties:", penalties)

    def vehicle_utilization(self):
        """Check vehicle utilization efficiency."""
        # Count vehicles that have routes longer than the depot-only route
        used_vehicles = sum(1 for route in self.best_routes if len(route) > 2)
        print(f"Total Vehicles Used: {used_vehicles} / {self.vehicle_count}")
        
        # Print under-utilization message if any vehicles are unused
        if used_vehicles < self.vehicle_count:
            print(f"{self.vehicle_count - used_vehicles} vehicles were unused, suggesting under-utilization.")

        # Calculate and print load for each vehicle
        for vehicle_index, route in enumerate(self.best_routes):
            if len(route) > 2:  # Ignore routes with only depot visits
                total_load = sum(self.customers[customer - 1].demand for customer in route if customer != 0)
                print(f"Vehicle {vehicle_index + 1} Load: {total_load} / {self.vehicle_capacity}")
