
Solving a Real World Problem Using Reinforcement Learning

This project develops an autonomous agent to operate within OpenAI Gym’s Taxi-v3 environment, where a taxi must navigate a grid, pick up passengers, and drop them off at specific locations. The goal is to train the agent to complete tasks with minimal steps and maximum rewards, simulating an efficient taxi service within a restricted environment.

The notebook implements three reinforcement learning (RL) algorithms:
Q-learning, SARSA, and REINFORCE, and evaluates their performance on the Taxi-v3 environment. The notebook is divided into three main parts: Q-learning Implementation and Training, Agent Evaluation, and Optimization with Alternative Algorithms.

Project Structure

The notebook contains three main sections:
	1.	Q-learning Agent Implementation and Training: Trains the primary agent in the Taxi-v3 environment using Q-learning.
	2.	Agent Evaluation: Evaluates the Q-learning agent’s performance by comparing it with a random policy and a heuristic policy.
	3.	Optimization with Alternative Algorithms: Compares the performance of SARSA and REINFORCE with Q-learning.

Requirements

To run this notebook, you’ll need the following packages:
	•	Python 3.x
	•	OpenAI Gym (for the Taxi-v3 environment)
	•	NumPy (for numerical operations)
	•	Matplotlib (for plotting results)
	•	PyTorch (for implementing the REINFORCE algorithm)

Setting Up a Virtual Environment

It’s recommended to set up a virtual environment to manage dependencies. Follow these steps:
	1.	Create a virtual environment:

python3 -m venv taxi_env


	2.	Activate the virtual environment:
	•	On macOS/Linux:

source taxi_env/bin/activate


	•	On Windows:

taxi_env\Scripts\activate


	3.	Install the required packages:

pip install gym numpy matplotlib torch



Code Overview

	1.	Q-learning Agent Implementation and Training:
This section implements and trains a Q-learning agent, which uses an epsilon-greedy policy to balance exploration and exploitation over 10,000 episodes. The resulting Q-table is saved for evaluation.
	2.	Agent Evaluation:
The trained Q-learning agent is evaluated against:
	•	A random policy: Baseline policy where actions are selected at random.
	•	A heuristic policy: Simple rules to guide the taxi toward the passenger or destination.
The evaluation compares the policies based on:
	•	Total Rewards: Effectiveness of each policy in achieving high rewards.
	•	Steps Taken: Efficiency of each policy in minimizing steps.
	3.	Optimization with Alternative Algorithms:
Additional experiments use SARSA (an on-policy method) and REINFORCE (a policy gradient method):
	•	SARSA: Updates based on the agent’s actual actions, leading to a more conservative policy.
	•	REINFORCE: Directly optimizes the policy based on cumulative rewards. Reward shaping is applied to give additional feedback when the taxi moves closer to the passenger or destination, aiding the learning process in a sparse reward environment.

Running the Notebook

	1.	Run the Notebook: Open the notebook in Jupyter Notebook or another environment and run each section in order:
	•	Q-learning Agent Implementation and Training: First, train the Q-learning agent.
	•	Agent Evaluation: Evaluate and compare the Q-learning agent with random and heuristic policies.
	•	Optimization with Alternative Algorithms: Compare Q-learning with SARSA and REINFORCE.
	2.	Results and Visualizations: The notebook generates and saves comparison plots in the results folder:
	•	Cumulative Rewards Comparison: Shows the total rewards achieved by each policy.
	•	Steps Taken Comparison: Illustrates the efficiency of each policy in terms of steps per episode.

Approach Explanation

	•	Q-learning: Learns an optimal action-value function (Q-table) for each state-action pair, balancing exploration and exploitation using an epsilon-greedy policy.
	•	SARSA: Updates based on the agent’s actual actions, resulting in a cautious, on-policy approach.
	•	REINFORCE: A policy gradient algorithm that optimizes the policy directly. Reward shaping is applied to provide extra feedback for actions that bring the taxi closer to the passenger or destination, making it more effective in a sparse-reward setting.

Results

The notebook includes visualizations comparing:
	1.	Q-learning, Random, and Heuristic Policies: Cumulative rewards and steps taken.
	2.	SARSA and REINFORCE: Comparative plots to show their performance alongside Q-learning, highlighting total rewards and steps taken per episode.

The results are saved as .png files in the results folder.

