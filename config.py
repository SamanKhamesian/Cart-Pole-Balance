SEED_TRAIN = [0, 1, 2]
SEED_TEST = 10


class TD3Config:
    def __init__(self):
        # TD3 hyperparameters
        self.gamma = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2

        # Learning rates
        self.actor_lr = 3e-4
        self.critic_lr = 3e-4

        # Replay buffer settings
        self.buffer_capacity = 1_000_00
        self.batch_size = 256

        # For action scaling
        self.max_action = 1.0
