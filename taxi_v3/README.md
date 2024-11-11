Taxi-v3 Autonomous Agent

This project develops an autonomous agent to operate within the OpenAI Gym’s Taxi-v3 environment, a grid-based simulation where a taxi must pick up and drop off passengers at specific locations. The goal is to train the agent to complete tasks with minimal steps and maximum rewards, simulating an efficient taxi service within a restricted environment.

The notebook includes implementations of three reinforcement learning algorithms: Q-learning, SARSA and REINFORCE, and evaluates their performance on the Taxi-v3 environment. The notebook is divided into three main parts: Q-learning Implementation and Training, Agent Evaluation, and Optimization with Alternative Algorithms.

Project Structure

The project notebook contains three main sections:
	1.	Q-learning Agent Implementation and Training: The primary RL algorithm used to train the Taxi-v3 agent.
	2.	Agent Evaluation: Code to evaluate the Q-learning agent’s performance and compare it with a random policy and a heuristic policy.
	3.	Optimization with Alternative Algorithms: Additional experiments with SARSA (an on-policy method) and REINFORCE (a policy gradient method) for comparison.

Requirements

To run this notebook, you’ll need the following packages:
	•	Python 3.x
	•	OpenAI Gym for the Taxi-v3 environment
	•	NumPy for numerical operations
	•	Matplotlib for plotting results
	•	PyTorch for implementing the REINFORCE algorithm

Install these packages using pip:

pip install gym numpy matplotlib torch

Code Overview and Approach

The notebook is structured as follows:

1. Q-learning Agent Implementation and Training

The first section of the notebook implements and trains the Q-learning agent, an off-policy algorithm that iteratively updates a Q-table to maximize cumulative rewards. The key hyperparameters tuned for optimal performance include:
	•	alpha (learning rate): Controls how much new information overrides previous Q-values.
	•	gamma (discount factor): Balances immediate and future rewards.
	•	epsilon (exploration rate): Controls the balance between exploration and exploitation, gradually decaying over time.

In this section, the agent uses an epsilon-greedy policy to explore and learn optimal actions over 10,000 episodes. The resulting Q-table is saved at the end of training for use in evaluation.

2. Agent Evaluation

This section evaluates the trained Q-learning agent by comparing it with two other approaches:
	1.	Random Policy: A baseline policy where actions are selected at random.
	2.	Heuristic-based Policy: A simple, rule-based approach that tries to move toward the passenger or destination directly.

The evaluation is performed over 100 episodes, and the following metrics are calculated:
	•	Total rewards: Measures the effectiveness of each policy in achieving high rewards.
	•	Steps taken: Measures the efficiency of each policy in minimizing steps.

Plots are generated to visualize cumulative rewards and steps taken across episodes for each policy.

3. Optimization with Alternative Algorithms (SARSA and REINFORCE)

In this section, two alternative RL algorithms are implemented:
	•	SARSA: An on-policy algorithm that updates the Q-table based on the agent’s actual actions, leading to a more cautious policy compared to Q-learning.
	•	REINFORCE: A policy gradient method that directly optimizes the policy based on cumulative rewards. To address the challenges posed by sparse rewards in the Taxi-v3 environment, reward shaping is applied to give additional feedback when the taxi moves closer to the passenger or destination.

Reward shaping provides incremental rewards for actions that move the taxi closer to the goal, helping the REINFORCE algorithm learn more effectively in a sparse reward setting.

Running the Notebook

	1.	Run the Notebook: Open the notebook in your preferred environment and run each section in order:
	•	Start with the Q-learning Agent Implementation and Training section to train the Q-learning agent.
	•	Proceed to the Agent Evaluation section to evaluate and compare the Q-learning agent with the random and heuristic policies.
	•	Lastly, run the Optimization with Alternative Algorithms section to compare Q-learning with SARSA and REINFORCE.
	2.	Save Results: The notebook automatically saves comparison plots in a results directory, including:
	•	Cumulative Rewards Comparison: Shows the total rewards achieved by each policy across episodes.
	•	Steps Taken Comparison: Illustrates the efficiency of each policy in terms of steps taken per episode.

Approach Explanation

Q-learning

Q-learning is an off-policy RL algorithm that learns the optimal action-value function for each state-action pair. It updates its Q-table using the Bellman equation to maximize expected cumulative rewards, balancing immediate and long-term goals. The epsilon-greedy policy decays over time, allowing the agent to explore initially and gradually shift toward exploitation as it learns an optimal policy.

SARSA

SARSA is an on-policy RL algorithm that updates the Q-table based on the agent’s actual experience, leading to a more cautious policy. Unlike Q-learning, which can learn from hypothetical actions, SARSA relies on the sequence of actions taken by the agent. This approach is useful for understanding how a conservative, on-policy algorithm performs in the Taxi-v3 environment.

REINFORCE

REINFORCE is a policy gradient algorithm that optimizes the policy directly based on cumulative rewards. Since the sparse rewards in Taxi-v3 make it challenging for REINFORCE to learn efficiently, reward shaping is applied to give feedback on movements that bring the taxi closer to the target. This provides more frequent signals, making learning more feasible in a sparse environment.

Reward Shaping

In the REINFORCE implementation, reward shaping provides extra feedback for actions that decrease the distance to the passenger or destination. This helps guide the agent’s behavior, especially in the early stages of learning.

Results and Visualizations

	1.	Q-learning, Random, and Heuristic Policies: The evaluation section includes plots comparing the Q-learning agent with random and heuristic policies on cumulative rewards and steps taken.
	2.	SARSA and REINFORCE: Individual and comparative plots are included to visualize the performance of SARSA and REINFORCE algorithms alongside Q-learning. Each algorithm’s total rewards and steps taken per episode are displayed to illustrate their learning progress and efficiency.

The results are saved as .png files in the results folder.
