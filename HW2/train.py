from gym import make
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from collections import deque, namedtuple
from utils import device
import random
import copy
from plot import show_result

SEED = 0
GAMMA = 0.99
INITIAL_STEPS = 1024
BUFFER_SIZE = 65_536
TRANSITIONS = 1_000_000
STEPS_PER_UPDATE = 4
STEPS_PER_TARGET_UPDATE = STEPS_PER_UPDATE * 1000
BATCH_SIZE = 64
LEARNING_RATE = 5e-4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class DQN:
    def __init__(self, state_dim: int, action_dim: int):
        self.steps = 0  # Do not change
        self.replay_buffer = deque(maxlen=BUFFER_SIZE)
        self.experience = namedtuple("Experience", field_names=["state", "action", "next_state", "reward", "done"])
        self.state_dim = state_dim
        self.policy_net = nn.Sequential(
            nn.Linear(in_features=state_dim, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=action_dim)
        ).to(device)
        self.target_net = copy.deepcopy(self.policy_net).to(device)
        self.optimizer = Adam(self.policy_net.parameters(), lr=LEARNING_RATE)

    def consume_transition(self, transition):
        # Add transition to a replay buffer.
        # Hint: use deque with specified maxlen. It will remove old experience automatically.
        state, action, next_state, reward, done = transition
        experience = self.experience(state, action, next_state, reward, done)
        self.replay_buffer.append(experience)

    def sample_batch(self):
        # Sample batch from a replay buffer.
        # Hints:
        # 1. Use random.randint
        # 2. Turn your batch into a numpy.array before turning it to a Tensor. It will work faster
        experiences = random.sample(self.replay_buffer, BATCH_SIZE)
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)

        return states, actions, next_states, rewards, dones

    def train_step(self, batch: np.ndarray):
        # Use batch to update DQN's network.
        states, actions, next_states, rewards, dones = batch
        state_action_values = self.policy_net(states).gather(1, actions)
        next_state_actions = self.policy_net(next_states).argmax(1, keepdim=True).long().detach()
        # next_state_values_target = self.target_net(next_states).max(1, keepdim=True)[0].detach()
        next_state_values = self.target_net(next_states).gather(1, next_state_actions).detach()
        expected_state_action_values = (GAMMA * next_state_values * (1 - dones)) + rewards
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        # Update weights of a target Q-network here. You may use copy.deepcopy to do this or
        # assign a values of network parameters via PyTorch methods.
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def act(self, state, target=False):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        state = torch.from_numpy(state).float().to(device)
        model = self.target_net if target else self.policy_net

        model.eval()
        with torch.no_grad():
            Q_value_actions = model(state)
        model.train()

        action = Q_value_actions.cpu().data.numpy().argmax()
        return action

    def update(self, transition):
        # You don't need to change this
        self.consume_transition(transition)
        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            self.train_step(batch)
        if self.steps % STEPS_PER_TARGET_UPDATE == 0:
            self.update_target_network()
        self.steps += 1

    def save(self):
        torch.save(self.policy_net, "agent.pkl")


def evaluate_policy(agent, episodes=5):
    env = make("LunarLander-v2")
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.

        while not done:
            state, reward, done, _ = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)
    return returns


def take_initial_steps(env, dqn):
    state = env.reset()
    for _ in range(INITIAL_STEPS):
        steps = 0
        action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action)
        dqn.consume_transition((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()

    return state


def print_policy_evaluation(dqn, step):
    rewards = evaluate_policy(dqn, 20)
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"Step: {step}, Reward mean: {mean_reward}, Reward std: {std_reward}")
    dqn.save()
    return mean_reward, std_reward


def train_agent(env, dqn, eps=0.5):
    means_ls, stds_ls, steps_ls = [], [], []
    state = take_initial_steps(env, dqn)
    decay = 0.99

    episode_long = 0
    for i in range(TRANSITIONS):
        steps = 0

        # Epsilon-greedy policy
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = dqn.act(state)

        next_state, reward, done, _ = env.step(action)
        dqn.update((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()
        eps *= decay if done else 1

        if (i + 1) % (TRANSITIONS // 100) == 0:
            mean_reward, std_reward = print_policy_evaluation(dqn, i + 1)
            print(f'Current epsilon: {eps}')
            steps_ls.append(i + 1)
            means_ls.append(mean_reward)
            stds_ls.append(std_reward)

    show_result(np.array(steps_ls), np.array(means_ls), np.array(stds_ls))


if __name__ == "__main__":
    print(f'Current device: {device}')
    env = make("LunarLander-v2")
    dqn = DQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    env.seed(SEED)
    train_agent(env, dqn)
