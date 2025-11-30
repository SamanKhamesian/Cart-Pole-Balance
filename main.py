# Author : Saman Khamesian
# ASU ID : 1229248581
# Course : EEE 598 (Reinforcement Learning in Robotics)
# Teacher: Prof. Jennie Si
# Subject: Assignment 3


import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from dm_control import suite

from config import TD3Config, SEED_TRAIN, SEED_TEST
from td3_agent import TD3Agent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class CartpoleEnv:
    def __init__(self, seed=0):
        self.env = suite.load(domain_name="cartpole", task_name="balance")
        set_seed(seed)

    @staticmethod
    def _flatten_obs(obs_dict):
        return np.concatenate([v.ravel() for v in obs_dict.values()])

    def reset(self):
        time_step = self.env.reset()
        return self._flatten_obs(time_step.observation)

    def step(self, action):
        time_step = self.env.step(action)
        obs = self._flatten_obs(time_step.observation)
        reward = time_step.reward
        done = time_step.last()
        return obs, reward, done


def train_agent(s, env, agent, episodes=200):
    rewards_per_episode = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            total_reward += reward
        rewards_per_episode.append(total_reward)

        if (episode + 1) % 10 == 0 or episode == 0:
            print(f"Seed {s} | Episode {episode + 1}/{episodes} | Reward: {total_reward:.2f}")

    return rewards_per_episode


def run_train_agent():
    all_rewards = []
    agent = None

    for s in SEED_TRAIN:
        # Create environment
        env = CartpoleEnv(seed=s)

        # Extract dimensions from environment
        state_dim = env.reset().shape[0]
        action_dim = env.env.action_spec().shape[0]
        max_action = float(env.env.action_spec().maximum[0])

        # Create config
        config = TD3Config()
        config.max_action = max_action

        # Create agent
        agent = TD3Agent(state_dim, action_dim, config)

        # Train
        rewards = train_agent(s, env, agent, episodes=500)
        all_rewards.append(rewards)

        print(f"Seed {s} finished.")

    return np.array(all_rewards), agent


def evaluate_agent(agent, episodes=100, render=False):
    rewards_per_episode = []
    env = CartpoleEnv(seed=SEED_TEST)

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            state = next_state
            total_reward += reward
            if render:
                env.env.physics.render()

        rewards_per_episode.append(total_reward)
        print(f"Evaluation Episode {ep+1}/{episodes} | Reward: {total_reward:.2f}")

    return np.array(rewards_per_episode)


def show_training_result(train_rewards):
    mean_rewards = np.mean(train_rewards, axis=0)
    std_rewards = np.std(train_rewards, axis=0)

    episodes = np.arange(1, train_rewards.shape[1] + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, mean_rewards, label="Reward (Mean)")
    plt.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3, label="± STD")
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Total Reward", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(np.arange(0, 1100, 200), fontsize=14)
    plt.title("TD3 Learning Curve on DMC Cartpole-Balance\n", fontsize=16)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig("td3_learning_curve.png")
    plt.show()


def show_eval_result(eval_rewards):
    episodes = np.arange(1, len(eval_rewards) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, eval_rewards, label="Evaluation Reward", color="tab:orange")
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Total Reward", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(np.arange(0, 1100, 200), fontsize=14)
    plt.title("TD3 Evaluation on DMC Cartpole-Balance\n", fontsize=16)
    plt.legend(fontsize=14, loc="lower right")
    plt.tight_layout()
    plt.savefig("td3_eval_curve.png")
    plt.show()


if __name__ == "__main__":
    train_reward_list, agent = run_train_agent()
    show_training_result(train_reward_list)

    test_reward_list = evaluate_agent(agent)
    show_eval_result(test_reward_list)

