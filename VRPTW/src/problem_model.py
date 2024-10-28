class Depot:
    def __init__(self, customer_no, x_coord, y_coord):
        self.customer_no = customer_no
        self.x_coord = x_coord
        self.y_coord = y_coord

class Customer: 
    def __init__(self, customer_no, x_coord, y_coord, demand, ready_time, due_date, service_time):
        self.customer_no = customer_no
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.demand = demand
        self.ready_time = ready_time
        self.due_date = due_date
        self.service_time = service_time

class Vehicle:
    def __init__(self, vehicle_no, capacity):
        self.vehicle_no = vehicle_no
        self.capacity = capacity
        self.current_load = 0
        self.route = []  # To store the route taken by the vehicle