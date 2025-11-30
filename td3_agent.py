import torch
import torch.nn.functional as F

from actor_critic import Actor, Critic
from config import TD3Config
from replay_buffer import ReplayBuffer


class TD3Agent:
    def __init__(self, state_dim, action_dim, config: TD3Config):
        self.cfg = config

        self.actor = Actor(state_dim, action_dim, config.max_action)
        self.actor_target = Actor(state_dim, action_dim, config.max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        self.replay_buffer = ReplayBuffer(state_dim, action_dim, config.buffer_capacity)

        self.total_it = 0
        self.action_dim = action_dim
        self.max_action = config.max_action

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        action = self.actor(state).detach().numpy().flatten()
        return action

    def store_transition(self, *args):
        self.replay_buffer.add(*args)

    def update(self):
        if self.replay_buffer.size < self.cfg.batch_size:
            return  # Not enough samples

        self.total_it += 1

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.cfg.batch_size)

        with torch.no_grad():
            # Add noise to actions for target policy smoothing
            noise = (torch.randn_like(actions) * self.cfg.policy_noise).clamp(-self.cfg.noise_clip, self.cfg.noise_clip)

            next_actions = (self.actor_target(next_states) + noise).clamp(-self.max_action, self.max_action)

            # Compute target Q-values
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.cfg.gamma * target_q

        # Current Q estimates
        current_q1, current_q2 = self.critic(states, actions)

        # Critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.cfg.policy_freq == 0:
            actor_loss = -self.critic.Q1(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data)
