from gym import make
import numpy as np
import torch
import copy
from collections import deque
import random

GAMMA = 0.98
GRID_SIZE_X = 30
GRID_SIZE_Y = 30


# Simple discretization
def transform_state(state):
    state = (np.array(state) + np.array((1.2, 0.07))) / np.array((1.8, 0.14))
    x = min(int(state[0] * GRID_SIZE_X), GRID_SIZE_X - 1)
    y = min(int(state[1] * GRID_SIZE_Y), GRID_SIZE_Y - 1)
    return x + GRID_SIZE_X * y


class QLearning:
    def __init__(self, state_dim, action_dim, lr):
        self._qlearning_estimate = np.zeros((state_dim, action_dim)) + 2.
        self._lr = lr

    # Updating Q-function
    def update(self, transition):
        state, action, next_state, reward = transition
        difference = (reward + GAMMA * np.max(self._qlearning_estimate[next_state])) - self._qlearning_estimate[
            state, action]
        self._qlearning_estimate[state, action] += self._lr * difference

    # Getting greedy optimal action
    def act(self, state):
        return np.argmax(self._qlearning_estimate[state])

    def save(self, path):
        np.savez(path, self._qlearning_estimate)


def evaluate_policy(agent, episodes=5):
    env = make("MountainCar-v0")
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.

        while not done:
            state, reward, done, _ = env.step(agent.act(transform_state(state)))
            total_reward += reward
        returns.append(total_reward)
    return returns


def train(env, ql, transitions=5_000_000, eps=0.3):
    reduction = eps / transitions
    state = env.reset()
    state_t = transform_state(state)
    trajectory = []
    for i in range(transitions):
        # Epsilon-greedy policy
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = ql.act(state_t)

        next_state, reward, done, _ = env.step(action)
        modified_reward = reward + 300 * (GAMMA * abs(next_state[1]) - abs(state[1]))
        # reward += abs(next_state[1]) / 0.07
        next_state_t = transform_state(next_state)

        trajectory.append((state_t, action, next_state_t, modified_reward))

        # Decay Epsilon
        if eps > 0:
            eps -= reduction

        if done:
            for transition in reversed(trajectory):
                ql.update(transition)
            trajectory = []

        state = next_state if not done else env.reset()
        state_t = next_state_t if not done else transform_state(state)

        if (i + 1) % (transitions // 100) == 0:
            rewards = evaluate_policy(ql, 5)
            mean_reward, std_reward = np.mean(rewards), np.std(rewards)
            print(f"Step: {i + 1}, Reward mean: {mean_reward}, Reward std: {std_reward}")
            ql.save("./agent.npz")


if __name__ == "__main__":
    env = make("MountainCar-v0")
    ql = QLearning(state_dim=GRID_SIZE_X * GRID_SIZE_Y, action_dim=3, lr=0.1)
    env.seed(0)
    np.random.seed(0)
    random.seed(0)

    train(env, ql)
