import imageio
import numpy as np
import gym
import tensorflow as tf
images = []
model = tf.keras.models.load_model('ReinforcementLearning/saved_model/my_model')

env = gym.make('CartPole-v0')

obs = env.reset()
img = env.render(mode='rgb_array')

done = False

while not done:
    images.append(img)
    obs = np.expand_dims(obs, 0)
    action = np.argmax(model.predict(obs))
    obs, _, done ,_ = env.step(action)
    img = env.render(mode='rgb_array')

imageio.mimsave('lander_a2c.gif', [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)
env.close()
