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

def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

def replay(batch_size):
    x_batch, y_batch = [], []
    minibatch = random.sample(memory, min(len(memory), batch_size))
    for state, action, reward, next_state, done in minibatch:
        y_target = model.predict(state)
        y_target[0][action] = reward if done else reward + gamma * np.max(model.predict(next_state)[0])
        x_batch.append(state[0])
        y_batch.append(y_target[0])

    model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)



EPOCHS = 1000
THRESHOLD = 45
MONITOR = True
batch_size = 64
gamma = 1.0
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995


env_string = 'CartPole-v0'
env = gym.make(env_string)   
input_size = env.observation_space.shape[0]
action_size = env.action_space.n
print(input_size, action_size)

memory = deque(maxlen=100000)


model = K.models.Sequential()
model.add(K.layers.Dense(24, input_dim=input_size, activation='tanh'))
model.add(K.layers.Dense(48, activation='tanh'))
model.add(K.layers.Dense(action_size, activation='linear'))
model.compile(loss='mse', optimizer=K.optimizers.Adam(learning_rate=0.01, decay=0.01))




scores = deque(maxlen=100)
avg_scores = []

for e in range(EPOCHS):
    print(e)
    state = env.reset()
    state = preprocessing_state(state)
    done = False
    i = 0
    while not done:
        action = choose_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        next_state = preprocessing_state(next_state)
        remember(state, action, reward, next_state, done)
        state = next_state
        epsilon = max(epsilon_min, epsilon_decay * epsilon) # decrease epsilon
        i += 1

    scores.append(i)
    mean_score = np.mean(scores)
    avg_scores.append(mean_score)
    if mean_score >= THRESHOLD and e >= 100:
        print('Ran {} episodes. Solved after {} trials ✔'.format(e, e - 100))
        break
    if e % 100 == 0:
        print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))
        model.save(os.path.join('C:/Codes/TensorflowWithKeras/ReinforcementLearning','saved_model/my_model'))
    replay(batch_size)

print('Did not solve after {} episodes 😞'.format(e))

import matplotlib.pyplot as plt
plt.plot(avg_scores)
plt.show()

model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_model/my_model')
model.save(model_dir)
env.close()