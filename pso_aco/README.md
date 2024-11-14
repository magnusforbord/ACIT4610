# Vehicle Routing Problem with Time Windows (VRPTW) Optimizer

## Overview
This project implements two nature-inspired algorithms (Ant Colony Optimization and Particle Swarm Optimization) to solve the Vehicle Routing Problem with Time Windows (VRPTW). It optimizes delivery routes for a fleet of vehicles while respecting time windows and capacity constraints.

## Features
- ACO (Ant Colony Optimization) implementation
- PSO (Particle Swarm Optimization) implementation
- Support for Solomon VRPTW benchmark instances
- Real-time visualization of solution progress
- Multiple route improvement techniques (2-opt, Or-opt)
- Constraint handling for time windows and vehicle capacity

## Requirements
```
numpy>=1.26.0
pandas>=2.1.0
matplotlib>=3.8.0
seaborn>=0.12.0
pytest>=7.4.3
```

## Installation
1. Clone the repository:
```bash
git clone <repository-url>
cd vrptw-optimizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
### Running ACO Solver
```python
python run_aco.py
```

### Running PSO Solver
```python
python run_pso.py
```

### Example Code
```python
from src.utils.data_loader import load_solomon_instance
from src.utils.distance_util import create_distance_matrix, create_time_matrix
from src.algorithms.aco.aco_optimizer import ACOOptimizer

# Load problem instance
problem = load_solomon_instance('data/c101.txt')

# Create matrices
dist_matrix = create_distance_matrix(problem.customers, problem.depot)
time_matrix = create_time_matrix(problem, dist_matrix)

# Initialize and run ACO
aco = ACOOptimizer(
    problem=problem,
    distance_matrix=dist_matrix,
    time_matrix=time_matrix,
    n_ants=50,
    alpha=1.0,
    beta=2.0,
    rho=0.1
)

solution, distances, times = aco.optimize(max_iterations=100)
```

## Project Structure
```
├── src/
│   ├── algorithms/
│   │   ├── aco/                 # Ant Colony Optimization
│   │   │   ├── ant.py
│   │   │   ├── colony.py
│   │   │   └── aco_optimizer.py
│   │   └── pso/                 # Particle Swarm Optimization
│   │       ├── particle.py
│   │       ├── swarm.py
│   │       └── pso_optimizer.py
│   └── utils/
│       ├── data_loader.py       # Problem instance loader
│       ├── distance_util.py     # Distance calculations
│       ├── validator.py         # Solution validation
│       ├── aco_visualizer.py    # ACO visualization
│       └── pso_visualizer.py    # PSO visualization
├── data/                        # Solomon benchmark instances
├── results/                     # Generated visualizations
├── requirements.txt
└── README.md
```

## Algorithm Parameters

### ACO Parameters
- `n_ants`: Number of ants in colony
- `alpha`: Pheromone influence factor
- `beta`: Heuristic information influence
- `rho`: Pheromone evaporation rate

### PSO Parameters
- `n_particles`: Swarm size
- `w`: Inertia weight
- `c1`: Cognitive component
- `c2`: Social component

## Visualization
The project generates several visualizations:
- Pheromone evolution (ACO)
- Route construction progress
- Convergence graphs
- Final solution routes

Results are saved in the `results/` directory.

## Solution Quality
Solutions are evaluated based on:
- Total distance traveled
- Number of vehicles used
- Time window constraint satisfaction
- Vehicle capacity constraint satisfaction

