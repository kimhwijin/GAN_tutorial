import imageio
import numpy as np
import gym
import tensorflow as tf
import os

images = []
record_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'records')
model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_model/my_model')
model = tf.keras.models.load_model(model_dir)


#mp4
env = gym.make("CartPole-v0")
env = gym.wrappers.Monitor(env, record_dir, force=True)

done = False
observation = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    observation, _, done, _ = env.step(action)
    if done:
        observation = env.reset()

env.reset()
env.close()


#make gif file
'''

obs = env.reset()
img = env.render(mode='rgb_array')

while not done:
    images.append(img)
    obs = np.expand_dims(obs, 0)
    action = np.argmax(model.predict(obs))
    obs, _, done ,_ = env.step(action)
    img = env.render(mode='rgb_array')

imageio.mimsave('CartPole-v0.gif', [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)
env.close()
'''

