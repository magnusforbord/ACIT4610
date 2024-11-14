import json
import pandas as pd
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
from typing import Dict, List

from src.algorithms.aco.aco_optimizer import ACOOptimizer
from src.algorithms.pso.pso_optimizer import PSOOptimizer
from src.utils.data_loader import Problem

class ExperimentManager:
    def __init__(self, experiment_name="vrptw_comparison"):
        self.experiment_name = experiment_name
        self.results_dir = "experiment_results"
        self.ensure_directories()
        
    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        dirs = [
            self.results_dir,
            f"{self.results_dir}/raw",
            f"{self.results_dir}/processed",
            f"{self.results_dir}/plots"
        ]
        for dir_path in dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

    def save_run_result(self, algorithm: str, run_number: int, data: dict):
        """Save individual run results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.results_dir}/raw/{algorithm}_run_{run_number}_{timestamp}.json"
        
        run_data = {
            "algorithm": algorithm,
            "run_number": run_number,
            "timestamp": timestamp,
            "distances": data["distances"],
            "times": data["times"],
            "final_distance": data["final_distance"],
            "feasible": data["feasible"],
            "n_routes": data["n_routes"],
            "total_time": data["total_time"]
        }
        
        with open(filename, 'w') as f:
            json.dump(run_data, f, indent=2)
            
    def load_all_results(self) -> pd.DataFrame:
        """Load all results into a pandas DataFrame"""
        all_data = []
        
        for filename in os.listdir(f"{self.results_dir}/raw"):
            if filename.endswith(".json"):
                with open(f"{self.results_dir}/raw/{filename}", 'r') as f:
                    data = json.load(f)
                    all_data.append(data)
        
        return pd.DataFrame(all_data)
    
    def generate_analysis(self):
        """Generate comprehensive analysis of all runs"""
        df = self.load_all_results()
        
        # Basic statistics per algorithm
        stats = df.groupby('algorithm').agg({
            'final_distance': ['mean', 'std', 'min', 'max'],
            'total_time': ['mean', 'std'],
            'n_routes': ['mean', 'min', 'max']
        }).round(2)
        
        # Save statistics
        stats.to_csv(f"{self.results_dir}/processed/algorithm_statistics.csv")
        
        # Generate plots
        self.generate_plots(df)
            
    def generate_plots(self, df: pd.DataFrame):
        """Generate and save comparison plots"""
        # 1. Convergence Plot
        plt.figure(figsize=(10, 6))
        for algo in df['algorithm'].unique():
            algo_data = df[df['algorithm'] == algo]
            
            # Debug print
            print(f"\nProcessing {algo}:")
            print(f"Number of runs: {len(algo_data)}")
            
            try:
                # Convert string representation of list to actual list if needed
                distances_list = [
                    json.loads(d) if isinstance(d, str) else d 
                    for d in algo_data['distances']
                ]
                
                # Ensure all distance lists are the same length
                min_length = min(len(d) for d in distances_list)
                distances_matrix = np.array([d[:min_length] for d in distances_list])
                
                mean_distances = np.mean(distances_matrix, axis=0)
                std_distances = np.std(distances_matrix, axis=0)
                
                iterations = range(len(mean_distances))
                plt.plot(iterations, mean_distances, label=algo, linewidth=2)
                plt.fill_between(iterations, 
                            mean_distances - std_distances,
                            mean_distances + std_distances,
                            alpha=0.2)
                    
                print(f"Processed distances shape: {distances_matrix.shape}")
                print(f"Mean distance range: {mean_distances.min():.2f} - {mean_distances.max():.2f}")
                
            except Exception as e:
                print(f"Error processing {algo} data: {str(e)}")
                print("Sample data:", algo_data['distances'].iloc[0][:5])
                continue
                
        plt.title('Convergence Comparison')
        plt.xlabel('Iteration')
        plt.ylabel('Distance')
        plt.legend()
        plt.grid(True)
        
        # Ensure proper axis scaling
        plt.margins(x=0.02)
        
        # Add padding to y-axis
        if plt.ylim()[0] < plt.ylim()[1]:  # Only if we have valid data
            ymin, ymax = plt.ylim()
            plt.ylim(ymin * 0.95, ymax * 1.05)
        
        plt.savefig(f"{self.results_dir}/plots/convergence_comparison.png")
        plt.close()
        
        
        # 2. Box Plot of Final Distances
        plt.figure(figsize=(8, 6))
        plt.boxplot([df[df['algorithm'] == algo]['final_distance'] 
                    for algo in df['algorithm'].unique()],
                    labels=df['algorithm'].unique())
        plt.title('Final Solution Quality Distribution')
        plt.ylabel('Distance')
        plt.grid(True)
        plt.savefig(f"{self.results_dir}/plots/solution_quality_boxplot.png")
        plt.close()
        
        # 3. Computation Time Comparison
        plt.figure(figsize=(8, 6))
        plt.boxplot([df[df['algorithm'] == algo]['total_time'] 
                    for algo in df['algorithm'].unique()],
                    labels=df['algorithm'].unique())
        plt.title('Computation Time Distribution')
        plt.ylabel('Time (seconds)')
        plt.grid(True)
        plt.savefig(f"{self.results_dir}/plots/computation_time_boxplot.png")
        plt.close()

def run_experiment(problem, distance_matrix, time_matrix, n_runs=30, n_iterations=200):
    """Run full experiment with both algorithms"""
    experiment = ExperimentManager()
    
    for run in range(n_runs):
        print(f"\nRun {run + 1}/{n_runs}")
        
        # ACO Run
        aco = ACOOptimizer(problem, distance_matrix, time_matrix, 50, 1.0, 2.0, 0.1)
        start_time = time.time()
        solution, distances, times = aco.optimize(n_iterations)
        total_time = time.time() - start_time
        
        experiment.save_run_result(
            algorithm="ACO",
            run_number=run,
            data={
                "distances": distances,
                "times": times,
                "final_distance": solution.total_distance,
                "feasible": solution.feasible,
                "n_routes": len(solution.routes),
                "total_time": total_time
            }
        )
        
        # PSO Run
        pso = PSOOptimizer(problem, distance_matrix, time_matrix, 200, 0.9, 2.5, 1.5)
        start_time = time.time()
        solution, distances, times = pso.optimize(n_iterations)
        total_time = time.time() - start_time
        
        experiment.save_run_result(
            algorithm="PSO",
            run_number=run,
            data={
                "distances": distances,
                "times": times,
                "final_distance": solution.total_distance,
                "feasible": solution.feasible,
                "n_routes": len(solution.routes),
                "total_time": total_time
            }
        )
    
    # Generate analysis after all runs
    experiment.generate_analysis()