from problem_model import Customer, Depot, Vehicle

def dataset_parser(file_path):
    with open(file_path, 'r') as file:
        all_lines = file.readlines()
    
    vehicle_data = []
    customer_data = []

    vehicle_section = False 
    customer_section = False
    for line in all_lines:
        line = line.strip()

        if not line: 
            continue
        
        if line.startswith("VEHICLE"):
            vehicle_section = True
            customer_section = False
            continue
        if line.startswith("CUSTOMER"):
            vehicle_section = False
            customer_section = True
            continue

        if vehicle_section: 
            if "CAPACITY" in line:
                continue
            else:
                parts = line.split()
                if len(parts) == 2:
                    num_vehicles = int(parts[0])
                    capacity = int(parts[1])
                    vehicle_data = [Vehicle(vehicle_no=i, capacity= capacity ) for i in range(num_vehicles)]

        if customer_section:
            parts = line.split()
            if len(parts) == 7:
                customer = Customer(
                    customer_no = int(parts[0]),
                    x_coord = float(parts[1]),
                    y_coord = float(parts[2]),
                    demand = int(parts[3]),
                    ready_time = int(parts[4]),
                    due_date = int(parts[5]),
                    service_time = int(parts[6])
                )
                customer_data.append(customer)
    depot = Depot(
        customer_no=0,
        x_coord=customer_data[0].x_coord,
        y_coord=customer_data[0].y_coord
    )
    customer_data = customer_data[1:]  # Exclude the depot from the customers list

    return vehicle_data, customer_data, depot