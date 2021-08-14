import random
import gym
#import math
import numpy as np
from collections import deque
import tensorflow as tf
import tensorflow.keras as K
import os

def preprocessing_state(state):
    return np.reshape(state, [1, 4])

def choose_action(state, epsilon):
    if np.random.random() <= epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(model.predict(state))


model = tf.keras.models.load_model('saved_model/my_model')
env = gym.make('CartPole-v0')


state = env.reset()
state = preprocessing_state(state)
done = False

for _ in range(1000):
    action = choose_action(state, 2)
    next_state, reward, done, _ = env.step(action)
    next_state = preprocessing_state(next_state)
    state = next_state
    env.render()

'''
for _ in range(1000):

    state, _, _, _ = env.step(tf.math.argmax(model.predict(state)))
    state = preprocessing_state(state)
    #env.step(env.action_space.sample())
    env.render()
'''
env.close()  # https://github.com/openai/gym/issues/893