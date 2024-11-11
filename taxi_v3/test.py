import gym
import pygame
import time  # To add delays between each movement

# Create the Taxi-v3 environment with render_mode="human"
env = gym.make('Taxi-v3', render_mode="human")
state = env.reset()

# Render the initial state of the environment
env.render()

# Add a simple event loop to keep the Pygame window open
running = True

# Number of steps to simulate (you can adjust this)
steps = 100

for step in range(steps):
    if not running:
        break
    
    # Take a random action (you can replace this with your Q-learning policy if needed)
    action = env.action_space.sample()  # Replace with a learned policy action if needed
    next_state, reward, terminated, truncated, info = env.step(action)
    
    # Render the updated state
    env.render()

    # Add a delay to slow down the visualization (0.5 seconds)
    time.sleep(0.5)

    # Check if the episode is over
    if terminated or truncated:
        print(f"Episode ended after {step + 1} steps")
        break

    # Check for Pygame events (quit or escape key)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

# Close the environment properly
env.close()
pygame.quit()