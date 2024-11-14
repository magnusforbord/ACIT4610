from src.utils.data_loader import load_solomon_instance
from src.utils.distance_util import create_distance_matrix, create_time_matrix
from Comparison_manager import ExperimentManager, run_experiment

def main():
    # Load problem
    print("Loading problem instance...")
    problem = load_solomon_instance('data/c101.txt')
    print(f"Loaded problem with {len(problem.customers)} customers")
    
    # Create matrices
    print("Creating distance and time matrices...")
    dist_matrix = create_distance_matrix(problem.customers, problem.depot)
    time_matrix = create_time_matrix(problem, dist_matrix)
    
    # Run experiment
    print("\nStarting experiment...")
    run_experiment(
        problem=problem,
        distance_matrix=dist_matrix,
        time_matrix=time_matrix,
        n_runs=20,        # Number of independent runs
        n_iterations=100  # Iterations per run
    )
    
    print("\nExperiment completed. Check 'experiment_results' directory for analysis.")

if __name__ == "__main__":
    main()