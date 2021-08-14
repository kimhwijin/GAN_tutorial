import random
import gym
import math
import numpy as np
from collections import deque
import tensorflow as tf
import tensorflow.keras as K

#최대 에피소드
EPOCHS = 1000
#임계값
THRESHOLD = 45
#기록 여부
MONITOR = True

class DQN():
    def __init__(self, env_string, batch_size=64):
        #경험 재생 버퍼 <S , A , R , S' > 저장
        self.memory = deque(maxlen=100000)
        self.env = gym.make(env_string)
        self.input_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        self.batch_size = batch_size
        self.gamma = 1.0
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        alpha = 0.01
        alpha_decay = 0.01
        if MONITOR: self.env = gym.wrappers.Monitor(self.env, '../data/' + env_string, force=True)

        self.model = K.models.Sequential()
        self.model.add(K.layers.Dense(24, input_dim=self.input_size, activation='tanh'))
        self.model.add(K.layers.Dense(48, activation='tanh'))
        self.model.add(K.layers.Dense(self.action_size, activation='linear'))
        self.model.compile(loss='mse', optimizer=K.optimizers.Adam(learning_rate=alpha, decay=alpha_decay))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        
        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict(state)
            y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])

        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)

    def choose_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.model.predict(state))
    
    def preprocess_state(self, state):
        return np.reshape(state, [1, self.input_size])

    def train(self):
        print("Train")
        scores = deque(maxlen=100)
        avg_scores = []

        for e in range(EPOCHS):
            print(e)
            state = self.env.reset()
            state = self.preprocess_state(state)
            done = False

            i = 0
            while not done:
                action = self.choose_action(state, self.epsilon)
                next_state, reward, done, _ = self.env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                i += 1

            scores.append(i)
            mean_score = np.mean(scores)
            avg_scores.append(mean_score)
            if mean_score >= THRESHOLD and e >= 100:
                print('Ran {} episodes. Solved after {} trials?'.format(e, e - 100))
            
            if e % 100 == 0:
                print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks'.format(e, mean_score))
        
        self.replay(self.batch_size)
        print('Did not solve after {} episodes : ('.format(e))
        return avg_scores


env_string = "CartPole-v0"
agent = DQN(env_string)
scores = agent.train()