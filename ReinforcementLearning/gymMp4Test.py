import gym
import os

print(os.getcwd())
path_dir = os.path.join(os.getcwd(), 'ReinforcementLearning/records')
env = gym.make("Breakout-v0")
env = gym.wrappers.Monitor(env, path_dir, force=True)
observation = env.reset()
for _ in range(512):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        observation = env.reset()

env.close()

