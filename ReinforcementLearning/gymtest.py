import gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation
env_name = "Breakout-v0"
env = gym.make(env_name)

frames = []
env.reset()
done = False
for _ in range(300):
    frames.append(env.render(mode='rgb_array'))
    obs, reward, done, _ = env.step(env.action_space.sample())
    if done:
        break

patch = plt.imshow(frames[0])
plt.axis('off')
def animate(i):
    patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=10)
    anim.save("random_agent.gif", writer='imagemagick')

